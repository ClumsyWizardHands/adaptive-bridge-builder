-- Principle Database Schema
-- This schema defines the database structure for storing the agent's operational principles
-- It includes tables for principles, categories, evaluation criteria, importance levels, tags,
-- and decision points, along with their relationships.

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- -----------------------------------------------------
-- Table: principle_categories
-- Description: Categories for organizing principles
-- -----------------------------------------------------
CREATE TABLE principle_categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    parent_category_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_category_id) REFERENCES principle_categories(id) ON DELETE SET NULL
);

-- -----------------------------------------------------
-- Table: importance_levels
-- Description: Predefined importance levels for principles
-- -----------------------------------------------------
CREATE TABLE importance_levels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    value INTEGER NOT NULL UNIQUE,  -- Numeric value for sorting/comparison
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------
-- Table: principles
-- Description: Core principles table containing the main attributes
-- -----------------------------------------------------
CREATE TABLE principles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    category_id INTEGER NOT NULL,
    importance_level_id INTEGER NOT NULL,
    short_name TEXT UNIQUE,  -- For programmatic reference
    version TEXT DEFAULT '1.0',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT,
    updated_by TEXT,
    FOREIGN KEY (category_id) REFERENCES principle_categories(id) ON DELETE RESTRICT,
    FOREIGN KEY (importance_level_id) REFERENCES importance_levels(id) ON DELETE RESTRICT
);

-- -----------------------------------------------------
-- Table: evaluation_types
-- Description: Types of evaluation criteria (LLM prompt, rules, etc.)
-- -----------------------------------------------------
CREATE TABLE evaluation_types (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    requires_llm BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------
-- Table: evaluation_criteria
-- Description: Detailed criteria for evaluating principles
-- -----------------------------------------------------
CREATE TABLE evaluation_criteria (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    principle_id INTEGER NOT NULL,
    type_id INTEGER NOT NULL,
    content TEXT NOT NULL,  -- Can be LLM prompt text or structured rule as JSON
    parameters TEXT,  -- JSON format for additional parameters
    version TEXT DEFAULT '1.0',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT,
    updated_by TEXT,
    FOREIGN KEY (principle_id) REFERENCES principles(id) ON DELETE CASCADE,
    FOREIGN KEY (type_id) REFERENCES evaluation_types(id) ON DELETE RESTRICT
);

-- -----------------------------------------------------
-- Table: tags
-- Description: Tags for organizing and grouping principles
-- -----------------------------------------------------
CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------
-- Table: principle_tags
-- Description: Junction table for many-to-many relationship between principles and tags
-- -----------------------------------------------------
CREATE TABLE principle_tags (
    principle_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (principle_id, tag_id),
    FOREIGN KEY (principle_id) REFERENCES principles(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
);

-- -----------------------------------------------------
-- Table: decision_points
-- Description: System locations where principles are evaluated
-- -----------------------------------------------------
CREATE TABLE decision_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    component TEXT NOT NULL,  -- E.g., "A2ATaskHandler", "ConflictResolver"
    context_schema TEXT,  -- JSON schema defining expected context structure
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------
-- Table: principle_decision_points
-- Description: Junction table linking principles to applicable decision points
-- -----------------------------------------------------
CREATE TABLE principle_decision_points (
    principle_id INTEGER NOT NULL,
    decision_point_id INTEGER NOT NULL,
    alignment_threshold REAL DEFAULT 0.7,  -- Value between 0 and 1
    priority INTEGER DEFAULT 0,  -- Higher value = higher priority
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (principle_id, decision_point_id),
    FOREIGN KEY (principle_id) REFERENCES principles(id) ON DELETE CASCADE,
    FOREIGN KEY (decision_point_id) REFERENCES decision_points(id) ON DELETE CASCADE
);

-- -----------------------------------------------------
-- Table: principle_evaluations
-- Description: Historical record of principle evaluations
-- -----------------------------------------------------
CREATE TABLE principle_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    principle_id INTEGER NOT NULL,
    decision_point_id INTEGER NOT NULL,
    action_description TEXT,
    context_summary TEXT,  -- JSON
    aligned BOOLEAN NOT NULL,
    overall_score REAL,  -- Value between 0 and 1
    evaluation_details TEXT,  -- JSON with full evaluation details
    was_modified BOOLEAN DEFAULT FALSE,
    modification_details TEXT,  -- JSON describing modifications
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (principle_id) REFERENCES principles(id) ON DELETE CASCADE,
    FOREIGN KEY (decision_point_id) REFERENCES decision_points(id) ON DELETE CASCADE
);

-- -----------------------------------------------------
-- Table: principle_dependencies
-- Description: Relationships between principles
-- -----------------------------------------------------
CREATE TABLE principle_dependencies (
    principle_id INTEGER NOT NULL,
    dependent_principle_id INTEGER NOT NULL,
    relationship_type TEXT NOT NULL,  -- 'supports', 'conflicts', 'requires', etc.
    strength REAL,  -- Value between 0 and 1
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (principle_id, dependent_principle_id),
    FOREIGN KEY (principle_id) REFERENCES principles(id) ON DELETE CASCADE,
    FOREIGN KEY (dependent_principle_id) REFERENCES principles(id) ON DELETE CASCADE,
    CHECK (principle_id != dependent_principle_id)  -- Prevent self-dependency
);

-- -----------------------------------------------------
-- Table: audit_log
-- Description: Track changes to principles and evaluation criteria
-- -----------------------------------------------------
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    record_id INTEGER NOT NULL,
    action TEXT NOT NULL,  -- 'INSERT', 'UPDATE', 'DELETE'
    old_values TEXT,  -- JSON format of old values
    new_values TEXT,  -- JSON format of new values
    user_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------
-- Indexes for better query performance
-- -----------------------------------------------------
CREATE INDEX idx_principles_category ON principles(category_id);
CREATE INDEX idx_principles_importance ON principles(importance_level_id);
CREATE INDEX idx_evaluation_criteria_principle ON evaluation_criteria(principle_id);
CREATE INDEX idx_principle_evaluations_principle ON principle_evaluations(principle_id);
CREATE INDEX idx_principle_evaluations_decision_point ON principle_evaluations(decision_point_id);
CREATE INDEX idx_principle_evaluations_timestamp ON principle_evaluations(timestamp);
CREATE INDEX idx_audit_log_table_record ON audit_log(table_name, record_id);

-- -----------------------------------------------------
-- Initial data for key tables
-- -----------------------------------------------------
-- Importance levels
INSERT INTO importance_levels (name, value, description) 
VALUES 
('Critical', 100, 'Fundamental principles that must never be violated'),
('High', 75, 'Very important principles that should rarely be compromised'),
('Medium', 50, 'Important principles that guide general behavior'),
('Low', 25, 'Principles that provide guidance but allow flexibility');

-- Evaluation types
INSERT INTO evaluation_types (name, description, requires_llm) 
VALUES 
('LLM Prompt', 'Natural language prompt for evaluation by LLM', TRUE),
('Rule-Based', 'Structured rules that can be evaluated programmatically', FALSE),
('Hybrid', 'Combines rule-based pre-filtering with LLM evaluation', TRUE),
('Example-Based', 'Provides examples of aligned and misaligned actions', TRUE);

-- Categories
INSERT INTO principle_categories (name, description) 
VALUES 
('Ethical', 'Principles related to ethical behavior and moral standards'),
('Operational', 'Principles governing system operations and behavior'),
('Relational', 'Principles for managing relationships and interactions'),
('Epistemic', 'Principles related to knowledge, truth, and understanding');

-- Sample decision points
INSERT INTO decision_points (name, description, component) 
VALUES 
('a2a_task_response_generation', 'Evaluates and potentially modifies responses generated by the A2A Task Handler', 'A2ATaskHandler'),
('orchestrator_task_assignment', 'Evaluates and potentially modifies task assignments made by the Orchestrator Engine', 'OrchestratorEngine'),
('conflict_resolution_generation', 'Evaluates and potentially modifies conflict resolutions generated by the Conflict Resolver', 'ConflictResolver'),
('agent_registry_selection', 'Evaluates and potentially modifies agent selections made by the Agent Registry', 'AgentRegistry'),
('agent_connector_request_formation', 'Evaluates and potentially modifies requests formed by the Universal Agent Connector', 'UniversalAgentConnector');
