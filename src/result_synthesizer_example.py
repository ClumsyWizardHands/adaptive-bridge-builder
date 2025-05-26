"""
Result Synthesizer Example

This module demonstrates how to use the ResultSynthesizer to collect, validate, and
synthesize outputs from multiple agents into cohesive final products with proper
attribution and quality control.
"""

import logging
import json
from datetime import datetime, timezone
from typing import Dict, List, Any

from result_synthesizer import (
    ResultSynthesizer, ContentType, QualityDimension, SynthesisStrategy,
    ResolutionStrategy, AttributionFormat
)
from conflict_resolver import ConflictResolver
from principle_engine import PrincipleEngine
from content_handler import ContentHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ResultSynthesizerExample")


def setup_synthesizer() -> ResultSynthesizer:
    """
    Set up and configure a ResultSynthesizer.
    
    Returns:
        Configured ResultSynthesizer instance
    """
    # Configure attribution format for acknowledging contributions
    attribution_format = AttributionFormat(
        template="This work was collaboratively created by: {contributors}",
        detailed_template=(
            "This {content_type} represents a synthesis of contributions from multiple agents:\n"
            "{detailed_contributions}\n\n"
            "Created through the 'Growth as a Shared Journey' principle."
        ),
        include_timestamps=True,
        include_confidence=True
    )
    
    # Create synthesizer with supporting components
    synthesizer = ResultSynthesizer(
        conflict_resolver=ConflictResolver(),
        principle_engine=PrincipleEngine(),
        content_handler=ContentHandler(),
        default_resolution_strategy=ResolutionStrategy.WEIGHTED_VOTE,
        default_synthesis_strategy=SynthesisStrategy.COMPLEMENTARY,
        attribution_format=attribution_format
    )
    
    return synthesizer


def example_text_synthesis() -> None:
    """
    Example of synthesizing text contributions from multiple agents.
    """
    synthesizer = setup_synthesizer()
    task_id = "market-report-task"
    
    # Add contribution from research agent
    research_content = """
    # Market Research Report: Cloud Computing Sector
    
    ## Industry Overview
    
    The cloud computing sector has experienced significant growth over the past decade, with 
    a compound annual growth rate (CAGR) of 17.5% from 2020 to 2025. The global market size 
    reached $371.4 billion in 2024, with projections to exceed $832.1 billion by 2030.
    
    ## Key Players
    
    The market remains dominated by major technology companies:
    - Amazon Web Services (AWS): 32% market share
    - Microsoft Azure: 21% market share
    - Google Cloud Platform: 9% market share
    - Alibaba Cloud: 6% market share
    - Other providers: 32% market share
    
    ## Recent Trends
    
    The most significant trends in cloud computing include:
    1. Increased adoption of multi-cloud strategies
    2. Growing demand for edge computing solutions
    3. Rising concerns around data sovereignty and compliance
    """
    
    synthesizer.add_contribution(
        agent_id="research-agent",
        content=research_content,
        content_type=ContentType.TEXT,
        task_id=task_id,
        confidence=0.92,
        attribution="Market data sourced from industry reports and verified against quarterly earnings statements.",
        metadata={"section": "market_research", "expertise": "industry_analysis"}
    )
    
    # Add contribution from analysis agent
    analysis_content = """
    # Analysis of Cloud Computing Market Dynamics
    
    ## Competitive Analysis
    
    The cloud services market exhibits oligopolistic characteristics, with the top three providers 
    controlling approximately 62% of the market. Barriers to entry remain high due to the 
    massive infrastructure investments required.
    
    ## Growth Drivers
    
    Key factors accelerating cloud adoption include:
    - Digital transformation initiatives accelerated by the COVID-19 pandemic
    - Increasing deployment of AI/ML workloads requiring scalable computing resources
    - Cost optimization strategies favoring OpEx over CapEx models
    - Shortage of skilled IT staff leading organizations to outsource infrastructure management
    
    ## Challenges and Constraints
    
    Despite strong growth prospects, the industry faces several headwinds:
    - Growing regulatory scrutiny in multiple jurisdictions
    - Data privacy concerns leading to fragmented compliance requirements
    - Environmental sustainability concerns related to energy consumption
    - Increasing cost of customer acquisition as the market matures
    """
    
    synthesizer.add_contribution(
        agent_id="analysis-agent",
        content=analysis_content,
        content_type=ContentType.TEXT,
        task_id=task_id,
        confidence=0.85,
        attribution="Analysis based on industry trends and economic indicators",
        metadata={"section": "competitive_analysis", "expertise": "business_strategy"}
    )
    
    # Add contribution from finance agent
    finance_content = """
    # Financial Outlook for Cloud Computing Sector
    
    ## Investment Trends
    
    Venture capital and private equity investments in cloud technologies reached $32.8 billion 
    in 2024, a 12% increase from the previous year. Early-stage funding has been particularly 
    strong in:
    - Serverless computing platforms
    - Cloud security solutions
    - Industry-specific cloud services (e.g., healthcare, finance)
    
    ## Valuation Metrics
    
    Public cloud providers are currently trading at the following multiples:
    - Price/Sales: 8.2x (average)
    - Price/Earnings: 42.3x (average)
    - EV/EBITDA: 22.7x (average)
    
    These valuations represent a premium of approximately 35% compared to the broader 
    technology sector, reflecting strong growth expectations.
    
    ## Profitability Analysis
    
    While revenue growth remains impressive, profitability varies significantly across 
    the sector:
    - Hyperscale providers achieve operating margins of 25-40%
    - Mid-tier providers typically see margins of 15-22%
    - Specialized niche providers can achieve margins of 30-45% in some cases
    """
    
    synthesizer.add_contribution(
        agent_id="finance-agent",
        content=finance_content,
        content_type=ContentType.TEXT,
        task_id=task_id,
        confidence=0.95,
        attribution="Financial data compiled from public filings, investor presentations, and market analyses.",
        metadata={"section": "financial_analysis", "expertise": "investment_analysis"}
    )
    
    # Synthesize the contributions
    result = synthesizer.synthesize_results(
        task_id=task_id,
        strategy=SynthesisStrategy.SECTION_BASED,
        resolution_strategy=ResolutionStrategy.WEIGHTED_VOTE,
        required_dimensions=[
            QualityDimension.ACCURACY,
            QualityDimension.COHERENCE,
            QualityDimension.ATTRIBUTION
        ],
        quality_threshold=0.8,
        metadata={"report_type": "market_analysis", "sector": "cloud_computing"}
    )
    
    # Print the synthesized result
    logger.info(f"Synthesized content type: {result.content_type.value}")
    logger.info(f"Quality scores: {[(d.value, s) for d, s in result.quality_scores.items()]}")
    logger.info(f"Contributors: {result.contributors}")
    logger.info(f"Attribution: {result.attribution_text}")
    logger.info(f"Content length: {len(result.content)} characters")
    logger.info(f"Conflicts resolved: {len(result.conflicts_resolved)}")
    logger.info("\nSynthesized Content Preview (first 500 chars):")
    logger.info(result.content[:500] + "...")


def example_data_synthesis() -> None:
    """
    Example of synthesizing structured data from multiple agents.
    """
    synthesizer = setup_synthesizer()
    task_id = "product-metrics-task"
    
    # Add contribution from metrics agent
    usage_metrics = {
        "user_statistics": {
            "total_users": 274500,
            "active_users": 189732,
            "new_users": 12450,
            "churned_users": 8320
        },
        "engagement_metrics": {
            "daily_active_users": 78500,
            "monthly_active_users": 189732,
            "average_session_duration": 18.7,  # minutes
            "sessions_per_user": 4.3
        },
        "platform_breakdown": {
            "web": 0.42,
            "ios": 0.38,
            "android": 0.20
        },
        "timestamp": "2025-05-15T00:00:00Z",
        "data_source": "analytics_platform"
    }
    
    synthesizer.add_contribution(
        agent_id="metrics-agent",
        content=usage_metrics,
        content_type=ContentType.STRUCTURED_DATA,
        task_id=task_id,
        confidence=0.98,
        attribution="Compiled from internal analytics database",
        metadata={"domain": "usage_metrics", "time_period": "monthly"}
    )
    
    # Add contribution from customer agent
    customer_metrics = {
        "customer_satisfaction": {
            "nps_score": 42,
            "csat": 4.2,  # out of 5
            "survey_responses": 3840
        },
        "support_metrics": {
            "tickets_opened": 12432,
            "average_response_time": 4.2,  # hours
            "average_resolution_time": 22.8,  # hours
            "first_contact_resolution_rate": 0.68
        },
        "feedback_themes": [
            {"theme": "user_interface", "sentiment": 0.65, "mention_count": 845},
            {"theme": "performance", "sentiment": 0.72, "mention_count": 756},
            {"theme": "features", "sentiment": 0.48, "mention_count": 1232},
            {"theme": "pricing", "sentiment": 0.35, "mention_count": 689}
        ],
        "timestamp": "2025-05-15T00:00:00Z",
        "data_source": "crm_system"
    }
    
    synthesizer.add_contribution(
        agent_id="customer-agent",
        content=customer_metrics,
        content_type=ContentType.STRUCTURED_DATA,
        task_id=task_id,
        confidence=0.94,
        attribution="Aggregated from customer support system and feedback surveys",
        metadata={"domain": "customer_experience", "time_period": "monthly"}
    )
    
    # Add contribution from revenue agent
    revenue_metrics = {
        "financial_metrics": {
            "mrr": 2850000,  # Monthly Recurring Revenue
            "arr": 34200000,  # Annual Recurring Revenue
            "revenue_growth": 0.23,  # Year-over-year
            "customer_acquisition_cost": 872,
            "customer_lifetime_value": 4350,
            "average_revenue_per_user": 15.2
        },
        "conversion_metrics": {
            "trial_conversion_rate": 0.22,
            "upgrade_rate": 0.18,
            "downgrade_rate": 0.07,
            "churn_rate": 0.042
        },
        "billing_efficiency": {
            "payment_success_rate": 0.965,
            "refund_rate": 0.023,
            "invoice_dispute_rate": 0.008
        },
        "timestamp": "2025-05-12T00:00:00Z",  # Note: slight inconsistency in date
        "data_source": "billing_system"
    }
    
    synthesizer.add_contribution(
        agent_id="revenue-agent",
        content=revenue_metrics,
        content_type=ContentType.STRUCTURED_DATA,
        task_id=task_id,
        confidence=0.97,
        attribution="Compiled from billing system and financial reports",
        metadata={"domain": "financial_metrics", "time_period": "monthly"}
    )
    
    # Synthesize the contributions
    result = synthesizer.synthesize_results(
        task_id=task_id,
        strategy=SynthesisStrategy.COMPLEMENTARY,
        resolution_strategy=ResolutionStrategy.EVIDENCE_BASED,
        required_dimensions=[
            QualityDimension.ACCURACY,
            QualityDimension.CONSISTENCY
        ],
        quality_threshold=0.9,
        metadata={"report_type": "executive_dashboard", "quarter": "Q2-2025"}
    )
    
    # Print the synthesized result
    logger.info(f"Synthesized data metrics with {len(result.conflicts_resolved)} resolved conflicts")
    logger.info(f"Contributors: {', '.join(result.contributors)}")
    logger.info(f"Attribution: {result.attribution_text}")
    
    # Print key sections from synthesized data
    synthesized_data = result.content
    logger.info("\nSynthesized Data Overview:")
    
    if isinstance(synthesized_data, dict):
        # Show top-level keys
        logger.info(f"Top-level keys: {', '.join(synthesized_data.keys())}")
        
        # Example of date conflict resolution if it exists
        if "timestamp" in synthesized_data:
            logger.info(f"Resolved timestamp: {synthesized_data['timestamp']}")
        
        # Show a summary of combined metrics
        combined_metrics = []
        for domain in ["user_statistics", "customer_satisfaction", "financial_metrics"]:
            if domain in synthesized_data:
                combined_metrics.append(domain)
        
        logger.info(f"Successfully combined domains: {', '.join(combined_metrics)}")
    else:
        logger.info(f"Unexpected data format: {type(synthesized_data)}")


def example_code_synthesis() -> None:
    """
    Example of synthesizing code from multiple agents.
    """
    synthesizer = setup_synthesizer()
    task_id = "data-processing-module"
    
    # Add contribution from architecture agent
    architecture_code = """
    '''
    Data Processing Module
    
    This module handles data processing operations for the analytics pipeline.
    It provides classes and functions for data extraction, transformation, and loading.
    '''
    
    import pandas as pd
    import numpy as np
    from typing import Dict, List, Any, Optional, Union
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("DataProcessor")
    
    
    class DataProcessor:
        '''Main class for data processing operations.'''
        
        def __init__(self, config: Dict[str, Any]):
            '''
            Initialize the data processor.
            
            Args:
                config: Configuration dictionary with processing parameters
            '''
            self.config = config
            self.data_sources = config.get('data_sources', [])
            self.transformations = config.get('transformations', [])
            self.output_format = config.get('output_format', 'csv')
            logger.info(f"Initialized DataProcessor with {len(self.data_sources)} sources")
        
        def extract_data(self, source_id: Optional[str] = None) -> pd.DataFrame:
            '''
            Extract data from the specified source.
            
            Args:
                source_id: Optional ID of the source to extract from
                
            Returns:
                DataFrame with extracted data
            '''
            # TODO: Implement data extraction logic
            pass
        
        def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
            '''
            Apply transformations to the data.
            
            Args:
                data: DataFrame to transform
                
            Returns:
                Transformed DataFrame
            '''
            # TODO: Implement transformation logic
            pass
        
        def load_data(self, data: pd.DataFrame, destination: str) -> bool:
            '''
            Load data to the specified destination.
            
            Args:
                data: DataFrame to load
                destination: Destination to load to
                
            Returns:
                Whether the load was successful
            '''
            # TODO: Implement data loading logic
            pass
        
        def run_pipeline(self) -> bool:
            '''
            Run the complete ETL pipeline.
            
            Returns:
                Whether the pipeline ran successfully
            '''
            # TODO: Implement pipeline logic
            pass
    """
    
    synthesizer.add_contribution(
        agent_id="architecture-agent",
        content=architecture_code,
        content_type=ContentType.CODE,
        task_id=task_id,
        confidence=0.9,
        attribution="Initial architecture design by architecture-agent",
        metadata={"language": "python", "module": "data_processor", "focus": "structure"}
    )
    
    # Add contribution from implementation agent
    implementation_code = """
    '''
    Data Processing Module Implementation
    
    This module contains the implementation of data processing operations.
    '''
    
    import pandas as pd
    import numpy as np
    from typing import Dict, List, Any, Optional, Union
    import logging
    import json
    import os
    from datetime import datetime, timezone
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("DataProcessor")
    
    
    class DataProcessor:
        '''Main class for data processing operations.'''
        
        def __init__(self, config: Dict[str, Any]):
            '''
            Initialize the data processor.
            
            Args:
                config: Configuration dictionary with processing parameters
            '''
            self.config = config
            self.data_sources = config.get('data_sources', [])
            self.transformations = config.get('transformations', [])
            self.output_format = config.get('output_format', 'csv')
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"Initialized DataProcessor with {len(self.data_sources)} sources")
        
        def extract_data(self, source_id: Optional[str] = None) -> pd.DataFrame:
            '''
            Extract data from the specified source.
            
            Args:
                source_id: Optional ID of the source to extract from
                
            Returns:
                DataFrame with extracted data
            '''
            if source_id is None and not self.data_sources:
                raise ValueError("No data source specified")
                
            if source_id is not None:
                sources = [s for s in self.data_sources if s.get('id') == source_id]
                if not sources:
                    raise ValueError(f"Data source {source_id} not found")
                source = sources[0]
            else:
                source = self.data_sources[0]
                
            source_type = source.get('type', 'csv')
            source_path = source.get('path', '')
            
            if source_type == 'csv':
                return pd.read_csv(source_path)
            elif source_type == 'json':
                return pd.read_json(source_path)
            elif source_type == 'database':
                # This would need database connection details
                raise NotImplementedError("Database extraction not implemented")
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
        
        def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
            '''
            Apply transformations to the data.
            
            Args:
                data: DataFrame to transform
                
            Returns:
                Transformed DataFrame
            '''
            result = data.copy()
            
            for transform in self.transformations:
                transform_type = transform.get('type', '')
                
                if transform_type == 'filter':
                    column = transform.get('column', '')
                    condition = transform.get('condition', '')
                    value = transform.get('value')
                    
                    if condition == 'equals':
                        result = result[result[column] == value]
                    elif condition == 'not_equals':
                        result = result[result[column] != value]
                    elif condition == 'greater_than':
                        result = result[result[column] > value]
                    elif condition == 'less_than':
                        result = result[result[column] < value]
                
                elif transform_type == 'map':
                    column = transform.get('column', '')
                    mapping = transform.get('mapping', {})
                    result[column] = result[column].map(mapping)
                
                elif transform_type == 'aggregate':
                    group_by = transform.get('group_by', [])
                    aggs = transform.get('aggregations', {})
                    result = result.groupby(group_by).agg(aggs).reset_index()
                    
            return result
        
        def load_data(self, data: pd.DataFrame, destination: str) -> bool:
            '''
            Load data to the specified destination.
            
            Args:
                data: DataFrame to load
                destination: Destination to load to
                
            Returns:
                Whether the load was successful
            '''
            try:
                output_dir = os.path.dirname(destination)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                if self.output_format == 'csv':
                    data.to_csv(destination, index=False)
                elif self.output_format == 'json':
                    data.to_json(destination, orient='records')
                elif self.output_format == 'parquet':
                    data.to_parquet(destination, index=False)
                else:
                    raise ValueError(f"Unsupported output format: {self.output_format}")
                    
                logger.info(f"Successfully saved data to {destination}")
                return True
            except Exception as e:
                logger.error(f"Failed to save data: {str(e)}")
                return False
        
        def run_pipeline(self) -> bool:
            '''
            Run the complete ETL pipeline.
            
            Returns:
                Whether the pipeline ran successfully
            '''
            try:
                # Extract data from all sources and combine
                all_data = []
                for source in self.data_sources:
                    source_id = source.get('id')
                    logger.info(f"Extracting data from source: {source_id}")
                    df = self.extract_data(source_id)
                    all_data.append(df)
                
                if not all_data:
                    logger.warning("No data extracted")
                    return False
                    
                # Combine all data
                if len(all_data) > 1:
                    combined_data = pd.concat(all_data, ignore_index=True)
                else:
                    combined_data = all_data[0]
                
                # Transform combined data
                logger.info(f"Transforming data with {len(self.transformations)} transformations")
                transformed_data = self.transform_data(combined_data)
                
                # Load data to destination
                output_path = self.config.get('output_path', f'output_{self.timestamp}.{self.output_format}')
                success = self.load_data(transformed_data, output_path)
                
                return success
            except Exception as e:
                logger.error(f"Pipeline execution failed: {str(e)}")
                return False
    """
    
    synthesizer.add_contribution(
        agent_id="implementation-agent",
        content=implementation_code,
        content_type=ContentType.CODE,
        task_id=task_id,
        confidence=0.85,
        attribution="Implementation details by implementation-agent",
        metadata={"language": "python", "module": "data_processor", "focus": "implementation"}
    )
    
    # Add contribution from optimization agent
    optimization_code = """
    '''
    Data Processing Module - Optimized Version
    
    This module contains optimized implementations of data processing operations.
    '''
    
    import pandas as pd
    import numpy as np
    from typing import Dict, List, Any, Optional, Union, Callable
    import logging
    import json
    import os
    from datetime import datetime, timezone
    import concurrent.futures
    from functools import partial
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("DataProcessor")
    
    # Performance monitoring decorator
    def monitor_performance(func: Callable) -> Callable:
        '''Decorator to monitor function performance.'''
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"{func.__name__} executed in {duration:.2f} seconds")
            return result
        return wrapper
    
    
    class DataProcessor:
        '''Main class for data processing operations.'''
        
        def __init__(self, config: Dict[str, Any]):
            '''
            Initialize the data processor.
            
            Args:
                config: Configuration dictionary with processing parameters
            '''
            self.config = config
            self.data_sources = config.get('data_sources', [])
            self.transformations = config.get('transformations', [])
            self.output_format = config.get('output_format', 'csv')
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.max_workers = config.get('max_workers', os.cpu_count())
            logger.info(f"Initialized DataProcessor with {len(self.data_sources)} sources")
            
        def _extract_from_source(self, source: Dict[str, Any]) -> pd.DataFrame:
            '''Extract data from a single source.'''
            source_type = source.get('type', 'csv')
            source_path = source.get('path', '')
            source_id = source.get('id', 'unknown')
            
            logger.info(f"Extracting from source {source_id} ({source_type})")
            
            try:
                if source_type == 'csv':
                    return pd.read_csv(source_path)
                elif source_type == 'json':
                    return pd.read_json(source_path)
                elif source_type == 'parquet':
                    return pd.read_parquet(source_path)
                else:
                    raise ValueError(f"Unsupported source type: {source_type}")
            except Exception as e:
                logger.error(f"Error extracting from {source_id}: {str(e)}")
                return pd.DataFrame()  # Return empty dataframe on error
        
        @monitor_performance
        def extract_data(self, source_id: Optional[str] = None) -> pd.DataFrame:
            '''
            Extract data from the specified source.
            
            Args:
                source_id: Optional ID of the source to extract from
                
            Returns:
                DataFrame with extracted data
            '''
            if source_id is None and not self.data_sources:
                raise ValueError("No data source specified")
                
            if source_id is not None:
                sources = [s for s in self.data_sources if s.get('id') == source_id]
                if not sources:
                    raise ValueError(f"Data source {source_id} not found")
                return self._extract_from_source(sources[0])
            else:
                # Process all sources in parallel using asyncio
                import asyncio
                loop = asyncio.get_event_loop()
                tasks = [loop.run_in_executor(None, self._extract_from_source, source) for source in self.data_sources]
                results = loop.run_until_complete(asyncio.gather(*tasks))
                
                # Combine results
                valid_results = [df for df in results if not df.empty]
                if not valid_results:
                    return pd.DataFrame()
                return pd.concat(valid_results, ignore_index=True)
        
        def _apply_transformation(
            self, data: pd.DataFrame, transform: Dict[str, Any]
        ) -> pd.DataFrame:
            '''Apply a single transformation to the data.'''
            transform_type = transform.get('type', '')
            
            try:
                if transform_type == 'filter':
                    column = transform.get('column', '')
                    condition = transform.get('condition', '')
                    value = transform.get('value')
                    
                    if condition == 'equals':
                        return data[data[column] == value]
                    elif condition == 'not_equals':
                        return data[data[column] != value]
                    elif condition == 'greater_than':
                        return data[data[column] > value]
                    elif condition == 'less_than':
                        return data[data[column] < value]
                
                elif transform_type == 'map':
                    column = transform.get('column', '')
                    mapping = transform.get('mapping', {})
                    result = data.copy()
                    result[column] = result[column].map(mapping)
                    return result
                
                elif transform_type == 'aggregate':
                    group_by = transform.get('group_by', [])
                    aggs = transform.get('aggregations', {})
                    return data.groupby(group_by).agg(aggs).reset_index()
                
                elif transform_type == 'create_column':
                    column = transform.get('column', '')
                    expression = transform.get('expression', '')
                    result = data.copy()
                    # Use eval for simple expressions
                    result[column] = result.eval(expression)
                    return result
                
                elif transform_type == 'drop_columns':
                    columns = transform.get('columns', [])
                    return data.drop(columns=columns)
                
                elif transform_type == 'rename_columns':
                    mapping = transform.get('mapping', {})
                    return data.rename(columns=mapping)
                
                else:
                    logger.warning(f"Unknown transformation type: {transform_type}")
                    return data
            except Exception as e:
                logger.error(f"Error applying transformation {transform_type}: {str(e)}")
                return data  # Return original data on error
        
        @monitor_performance
        def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
            '''
            Apply transformations to the data.
            
            Args:
                data: DataFrame to transform
                
            Returns:
                Transformed DataFrame
            '''
            if data.empty:
                logger.warning("Cannot transform empty dataset")
                return data
                
            result = data.copy()
            
            # Apply each transformation sequentially
            for transform in self.transformations:
                transform_type = transform.get('type', '')
                logger.info(f"Applying transformation: {transform_type}")
                result = self._apply_transformation(result, transform)
                
            return result
        
        @monitor_performance
        def load_data(self, data: pd.DataFrame, destination: str) -> bool:
            '''
            Load data to the specified destination.
            
            Args:
                data: DataFrame to load
                destination: Destination to load to
                
            Returns:
                Whether the load was successful
            '''
            if data.empty:
                logger.warning("Cannot save empty dataset")
                return False
                
            try:
                output_dir = os.path.dirname(destination)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                # Use appropriate save method based on format
                if self.output_format == 'csv':
                    data.to_csv(destination, index=False)
                elif self.output_format == 'json':
                    data.to_json(destination, orient='records')
                elif self.output_format == 'parquet':
                    data.to_parquet(destination, index=False)
                elif self.output_format == 'excel':
                    data.to_excel(destination, index=False)
                else:
                    raise ValueError(f"Unsupported output format: {self.output_format}")
                    
                logger.info(f"Successfully saved data to {destination} ({len(data)} records)")
                return True
            except Exception as e:
                logger.error(f"Failed to save data: {str(e)}")
                return False
                
        @monitor_performance
        def run_pipeline(self) -> bool:
            '''
            Run the complete ETL pipeline.
            
            Returns:
                Whether the pipeline ran successfully
            '''
            try:
                # Extract data
                logger.info(f"Starting data extraction from {len(self.data_sources)} sources")
                combined_data = self.extract_data()
                
                if combined_data.empty:
                    logger.warning("No data extracted, pipeline terminated")
                    return False
                    
                logger.info(f"Extracted {len(combined_data)} records")
                
                # Transform data
                logger.info(f"Applying {len(self.transformations)} transformations")
                transformed_data = self.transform_data(combined_data)
                
                if transformed_data.empty:
                    logger.warning("Transformation resulted in empty dataset, pipeline terminated")
                    return False
                    
                logger.info(f"Transformation complete, {len(transformed_data)} records remaining")
                
                # Load data
                output_path = self.config.get('output_path', f'output_{self.timestamp}.{self.output_format}')
                logger.info(f"Loading data to {output_path}")
                success = self.load_data(transformed_data, output_path)
                
                return success
            except Exception as e:
                logger.error(f"Pipeline execution failed: {str(e)}")
                return False
    """
    
    synthesizer.add_contribution(
        agent_id="optimization-agent",
        content=optimization_code,
        content_type=ContentType.CODE,
        task_id=task_id,
        confidence=0.92,
        attribution="Performance optimizations by optimization-agent",
        metadata={"language": "python", "module": "data_processor", "focus": "optimization"}
    )
    
    # Synthesize the contributions
    result = synthesizer.synthesize_results(
        task_id=task_id,
        strategy=SynthesisStrategy.BEST_ELEMENTS,
        resolution_strategy=ResolutionStrategy.MOST_RECENT,
        required_dimensions=[
            QualityDimension.FUNCTIONALITY,
            QualityDimension.EFFICIENCY,
            QualityDimension.READABILITY
        ],
        quality_threshold=0.85,
        metadata={"version": "1.0.0", "language": "python"}
    )
    
    # Print the synthesized result
    logger.info(f"Synthesized code with {len(result.conflicts_resolved)} resolved conflicts")
    logger.info(f"Contributors: {', '.join(result.contributors)}")
    logger.info(f"Attribution: {result.attribution_text}")
    logger.info(f"Code length: {len(result.content)} characters")
    
    # Show selected features from the synthesized code
    code_lines = result.content.split("\n")
    logger.info(f"\nCode Structure Preview ({len(code_lines)} lines total):")
    
    # Look for specific code elements
    performance_monitoring = "monitor_performance" in result.content
    parallel_processing = "concurrent.futures" in result.content
    error_handling = "try:" in result.content and "except" in result.content
    
    logger.info(f"Performance monitoring included: {performance_monitoring}")
    logger.info(f"Parallel processing included: {parallel_processing}")
    logger.info(f"Error handling included: {error_handling}")
    
    # Print sample of function signatures
    function_lines = [line for line in code_lines if line.strip().startswith("def ")]
    if function_lines:
        logger.info("\nFunction signatures:")
        for i, line in enumerate(function_lines[:5]):  # Show first 5 functions
            logger.info(f"  {line.strip()}")
        if len(function_lines) > 5:
            logger.info(f"  ... and {len(function_lines) - 5} more")


def main() -> None:
    """Run the examples."""
    logger.info("Starting Result Synthesizer Examples")
    
    logger.info("\n\n=== TEXT SYNTHESIS EXAMPLE ===\n")
    example_text_synthesis()
    
    logger.info("\n\n=== DATA SYNTHESIS EXAMPLE ===\n")
    example_data_synthesis()
    
    logger.info("\n\n=== CODE SYNTHESIS EXAMPLE ===\n")
    example_code_synthesis()
    
    logger.info("\nExamples completed successfully")


if __name__ == "__main__":
    main()
