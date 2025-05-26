# Empire Framework Configuration Plan

## Overview

This document outlines how Empire Framework components, especially Principles and their hierarchies, will be configured and loaded by the Adaptive Bridge Builder agent. The configuration approach follows a layered strategy with clear precedence rules, multiple configuration methods, and robust validation.

## Configuration Location Hierarchy

The agent will look for configuration in the following locations, in order of precedence (highest to lowest):

1. **Command Line Arguments** - For one-time overrides and testing
2. **Environment Variables** - For deployment-specific configuration
3. **Configuration Files** - For persistent, version-controlled settings
4. **Default Values** - Fallback for unconfigured values

## Directory Structure

```
/resources/
  /empire_framework_schemas/   # JSON schemas for validation
    principle_schema.json
    profile_schema.json
    growth_journal_schema.json
  /principles/                # Principle definition files
    fairness.json
    adaptability.json
    harmony.json
    empathy.json
    resilience.json
    domain_specific/
      technical_principles.json
      social_principles.json
  /profiles/                  # Profile configurations
    adaptive_hero.json
    technical_mediator.json
  /config/                    # Configuration files
    empire_framework.json     # Main configuration
    principle_weights.json    # Weights for different contexts
    environment_configs/      # Environment-specific configurations
      development.json
      production.json
      testing.json
```

## Configuration File Formats

### Main Configuration File (empire_framework.json)

```json
{
  "profile": "adaptive_hero",
  "active_principles": ["fairness", "adaptability", "harmony", "empathy", "resilience"],
  "principle_directory": "resources/principles",
  "growth_journal_path": "data/growth_journal.json",
  "reflection_schedule": {
    "frequency": "daily",
    "deep_reflection_frequency": "weekly"
  },
  "logging": {
    "principle_decisions": true,
    "min_log_level": "INFO",
    "log_file": "logs/empire_framework.log"
  },
  "context_detection": {
    "enabled": true,
    "context_samples": 5
  }
}
```

### Principle File (example: fairness.json)

```json
{
  "id": "fairness",
  "name": "Fairness as a Fundamental Truth",
  "description": "Every interaction must uphold equity, avoiding bias while acknowledging and accommodating differences.",
  "weight": 0.9,
  "category": "core",
  "evaluation_criteria": [
    {
      "id": "bias_avoidance",
      "description": "Avoids introducing or perpetuating bias",
      "weight": 0.4,
      "evaluation_function": "evaluate_bias_avoidance"
    },
    {
      "id": "equal_consideration",
      "description": "Gives equal consideration to all relevant parties",
      "weight": 0.3,
      "evaluation_function": "evaluate_equal_consideration"
    },
    {
      "id": "difference_accommodation",
      "description": "Accommodates relevant differences appropriately",
      "weight": 0.3,
      "evaluation_function": "evaluate_difference_accommodation"
    }
  ],
  "relationships": [
    {
      "principle_id": "harmony",
      "relationship_type": "supports",
      "strength": 0.8
    },
    {
      "principle_id": "empathy",
      "relationship_type": "prerequisite_for",
      "strength": 0.7
    }
  ]
}
```

### Profile File (example: adaptive_hero.json)

```json
{
  "id": "adaptive_hero",
  "name": "Empire of the Adaptive Hero",
  "description": "A profile focused on adaptability, fairness, and harmony",
  "principle_weights": {
    "adaptability": 1.0,
    "fairness": 0.9,
    "harmony": 0.8,
    "empathy": 0.7,
    "resilience": 0.7
  },
  "context_specific_weights": {
    "crisis": {
      "adaptability": 1.0,
      "resilience": 0.9,
      "fairness": 0.8
    },
    "long_term_planning": {
      "harmony": 1.0,
      "fairness": 0.9,
      "empathy": 0.8
    }
  }
}
```

## Environment Variables

Environment variables follow the pattern `EMPIRE_[COMPONENT]_[SETTING]`. For example:

- `EMPIRE_PROFILE=adaptive_hero` - Set the active profile
- `EMPIRE_PRINCIPLE_WEIGHTS=fairness:0.9,adaptability:1.0` - Override principle weights
- `EMPIRE_CONFIG_PATH=/custom/path/empire_config.json` - Use custom config path
- `EMPIRE_LOGGING_LEVEL=DEBUG` - Set logging level
- `EMPIRE_GROWTH_JOURNAL_PATH=/data/growth_journal.json` - Custom growth journal location
- `EMPIRE_CONTEXT=production` - Set the context (which may affect principle weights)

## Command Line Arguments

Command line arguments follow a similar pattern and override environment variables:

```
--empire-profile adaptive_hero
--empire-principle-weights fairness:0.9,adaptability:1.0
--empire-config-path /custom/path/empire_config.json
--empire-logging-level DEBUG
--empire-growth-journal-path /data/growth_journal.json
--empire-context production
```

## Loading Process

### Initialization Sequence

1. **Parse Command Line Arguments**
   - Extract Empire Framework settings
   - Set initial configuration context

2. **Load Environment Variables**
   - Check for all EMPIRE_* variables
   - Apply settings not already set by command line

3. **Determine Base Configuration Path**
   - Use path from command line or environment if provided
   - Fall back to default location (resources/config/empire_framework.json)

4. **Load Base Configuration File**
   - Parse and validate against schema
   - Apply settings not already set

5. **Determine Environment-Specific Configuration**
   - Check for environment override (production, development, testing)
   - Load and merge environment-specific configuration if present

6. **Load Profile Configuration**
   - Based on the resolved profile (adaptive_hero by default)
   - Apply profile settings (especially principle weights)

7. **Load Principle Definitions**
   - Load all principles from the principles directory
   - Apply any weight overrides from profile or explicit configuration
   - Build principle relationships graph

8. **Initialize Principle Engine**
   - Register all principles with their weights
   - Register evaluation functions
   - Initialize relationship graph for conflict resolution

9. **Initialize Growth Journal**
   - Connect to specified journal path
   - Ensure journal structure is valid
   - Load previous experiences and learnings

10. **Initialize Reflection System**
    - Set up reflection schedules
    - Connect to orchestration analytics

## Principle Hierarchy and Relationships

Principles form a directed graph where relationships define how principles interact with each other:

- **supports**: Principles that align with and reinforce each other
- **conflicts_with**: Principles that may be in tension, requiring resolution
- **prerequisite_for**: Principles that must be satisfied before another
- **dependent_on**: Principles that rely on others for full expression

The Principle Engine uses this graph for:

1. **Conflict Resolution**: When principles conflict, resolution strategies consider:
   - Principle weights (higher weighted principles take precedence)
   - Relationship strengths between principles
   - Context-specific weight adjustments
   - Predefined resolution strategies for known conflicts

2. **Alignment Scoring**: When evaluating actions:
   - Primary principles are scored directly
   - Relationship effects are propagated through the graph
   - Final alignment score accounts for direct and indirect effects

3. **Explanation Generation**: When explaining decisions:
   - Principle relationships help generate coherent explanations
   - Tracing paths through the relationship graph provides rationale

## Context-Aware Configuration

The agent can dynamically adjust principle weights based on detected context:

1. **Context Detection**
   - Analyze current interaction type (crisis, planning, social, technical)
   - Consider previous interactions and established patterns
   - Use explicit context settings if provided

2. **Weight Adjustment**
   - Apply context-specific weights from profile definition
   - Temporary adjustments don't modify base configuration
   - Log weight adjustments for later reflection

3. **Context Transitions**
   - Smooth transitions between contexts
   - Gradually adjust weights rather than abrupt changes
   - Track context changes in growth journal

## Configuration Validation

All configuration is validated before use:

1. **Schema Validation**
   - Validate all JSON files against their schemas
   - Report detailed validation errors for fixing

2. **Relationship Consistency**
   - Ensure principle relationships are bi-directional where appropriate
   - Check for circular dependencies
   - Verify relationship strengths are normalized

3. **Evaluation Function Validation**
   - Verify all referenced evaluation functions exist
   - Check function signatures match expected patterns

4. **Weight Normalization**
   - Ensure weights are properly normalized
   - Apply rescaling if necessary

## Implementation Approach

### Phase 1: Core Configuration Loading

1. Create basic configuration loader
2. Implement command line and environment variable parsing
3. Add JSON schema validation
4. Build basic principle loading

### Phase 2: Principle Relationship Graph

1. Build relationship graph data structure
2. Implement relationship consistency validation
3. Create utilities for traversing the graph
4. Add visualization tools for debugging

### Phase 3: Context-Aware Configuration

1. Implement context detection mechanisms
2. Create weight adjustment strategies
3. Build context transition handling
4. Add logging for context changes

### Phase 4: Integration with PrincipleEngine

1. Connect configuration to PrincipleEngine
2. Implement dynamic principle adjustment
3. Add decision explanation based on relationships
4. Integrate with GrowthJournal

## Testing Approach

1. **Unit Tests**
   - Test configuration loading from all sources
   - Verify precedence rules are followed
   - Test validation of malformed configurations
   - Verify relationship graph building

2. **Integration Tests**
   - Test PrincipleEngine with full configuration
   - Verify context-aware weight adjustment
   - Test decision making across different profiles
   - Verify configuration changes are reflected in behavior

3. **Configuration Tests**
   - Validate all bundled configurations
   - Test with minimal and maximal configurations
   - Verify defaults are applied correctly
   - Test environment-specific configuration loading

## Deployment Considerations

1. **Configuration Management**
   - Store base configurations in version control
   - Use environment variables for deployment-specific settings
   - Document all configuration options

2. **Security**
   - Don't store sensitive information in principle definitions
   - Use secure storage for growth journal with sensitive insights
   - Apply appropriate file permissions

3. **Monitoring**
   - Log configuration changes
   - Monitor principle weight adjustments
   - Track decision alignment scores over time
   - Alert on configuration validation failures

4. **Backup and Recovery**
   - Regularly backup growth journal and learned adjustments
   - Maintain configuration history
   - Support rollback to previous configurations

## Conclusion

This configuration plan provides a flexible, robust approach to managing Empire Framework components, especially Principles and their relationships. The layered configuration strategy allows for:

- Easy deployment across different environments
- Quick testing and experimentation with command line overrides
- Consistent configuration validation
- Dynamic adaptation to different contexts
- Clear precedence rules when settings conflict

By following this plan, the Adaptive Bridge Builder agent will maintain consistent, principled behavior while adapting to different usage scenarios and learning from experience.
