from typing import Dict, List, Tuple
import pandas as pd
from sfn_blueprint import SFNAgent
from sfn_blueprint import Task
from sfn_blueprint import SFNAIHandler
from sfn_blueprint import SFNPromptManager

class FeatureClassificationAgent(SFNAgent):
    """Agent responsible for categorizing features and generating metadata"""
    
    def __init__(self, llm_provider='openai'):
        super().__init__(name="Feature Classification", role="Feature Classifier")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        
    def execute_task(self, task: Task) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """
        Classifies features and generates metadata
        """
        if not isinstance(task.data, pd.DataFrame):
            raise ValueError("Task data must be a pandas DataFrame")
        
        df = task.data
        print(f"DataFrame info:")
        print(df.info())
        print("\nDataFrame head:")
        print(df.head())
        
        feature_metadata = {}
        feature_classifications = {}
        
        for column in df.columns:
            try:
                print(f"\nProcessing column: {column}")
                print(f"Column dtype: {df[column].dtype}")
                print(f"Null values: {df[column].isnull().sum()}")
                print(f"Sample values: {df[column].head()}")
                
                total_rows = len(df)
                series = df[column]
                
                # Basic metadata
                metadata = {
                    "null_percentage": (series.isnull().sum() / total_rows * 100),
                    "unique_ratio": (series.nunique() / total_rows),
                    "cardinality": series.nunique(),
                    "constant_flag": series.nunique() <= 1,
                    "data_type": str(series.dtype),
                    "zero_percentage": 0.0  # Default value
                }
                
                # Only calculate zero percentage for numeric columns
                if pd.api.types.is_numeric_dtype(series):
                    try:
                        zero_count = (series == 0).sum()
                        metadata["zero_percentage"] = (zero_count / total_rows * 100)
                    except:
                        print(f"Error calculating zero percentage for column {column}")
                
                print(f"Generated metadata: {metadata}")
                feature_metadata[column] = metadata
                
                # Classify feature
                category = self._determine_category(series)
                print(f"Determined category: {category}")
                
                feature_classifications[column] = {
                    "category": category,
                    "data_type": str(series.dtype),
                    "description": self._generate_category_description(category, metadata)
                }
                print(f"Generated classification: {feature_classifications[column]}")
                
            except Exception as e:
                print(f"Error processing column {column}: {str(e)}")
                # Safe default values
                feature_metadata[column] = {
                    "null_percentage": 0.0,
                    "unique_ratio": 0.0,
                    "cardinality": 0,
                    "constant_flag": False,
                    "data_type": "unknown",
                    "zero_percentage": 0.0
                }
                feature_classifications[column] = {
                    "category": "Unknown",
                    "data_type": "unknown",
                    "description": "Feature type could not be determined"
                }
        
        return feature_classifications, feature_metadata
        
    def _determine_category(self, series: pd.Series) -> str:
        """Determines if a feature is Numerical, Categorical, or DateTime"""
        try:
            if pd.api.types.is_datetime64_any_dtype(series):
                return "DateTime"
            elif pd.api.types.is_numeric_dtype(series):
                return "Numerical"
            else:
                return "Categorical"
        except Exception as e:
            print(f"Error determining category: {str(e)}")
            return "Unknown"
            
    def _generate_category_description(self, category, metadata):
        """Generate a description for a feature category with safe null handling"""
        try:
            print(f"Generating description for category {category} with metadata {metadata}")
            
            if metadata.get('constant_flag', False):
                return "Constant value feature - consider removing"
            
            if category == "Numerical":
                return f"Numerical feature with {metadata['null_percentage']:.1f}% missing values, {metadata['zero_percentage']:.1f}% zeros"
            elif category == "Categorical":
                return f"Categorical feature with {metadata['cardinality']} unique values ({metadata['unique_ratio']:.2%} unique ratio)"
            elif category == "DateTime":
                return f"Temporal feature with {metadata['null_percentage']:.1f}% missing values"
            elif category == "Text":
                return "Text feature containing string data"
            else:
                return f"Feature of type {category}"
                
        except Exception as e:
            print(f"Error generating description: {str(e)}")
            return f"Feature of type {category}"

    def get_validation_params(self, response, task):
        """Get parameters for validation"""
        if not isinstance(task.data, pd.DataFrame):
            raise ValueError("Task data must be a pandas DataFrame")

        # Get validation prompts from prompt manager
        prompts = self.prompt_manager.get_prompt(
            agent_type='feature_classifier',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response,
            columns=task.data.columns.tolist()
        )
        return prompts 