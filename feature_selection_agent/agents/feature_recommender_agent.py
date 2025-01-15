from typing import Dict, List
import pandas as pd
from sfn_blueprint import SFNAgent
from sfn_blueprint import Task
from sfn_blueprint import SFNAIHandler
import os
from sfn_blueprint import SFNPromptManager
from feature_selection_agent.config.model_config import MODEL_CONFIG, DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER
import json
from sfn_blueprint import SFNSessionManager

class SFNFeatureRecommenderAgent(SFNAgent):
    """Agent responsible for providing final feature recommendations with explanations"""
    
    def __init__(self, llm_provider='openai'):
        super().__init__(name="Feature Recommender", role="Feature Advisor")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["feature_recommender"]
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        method_config_path = os.path.join(parent_path, 'config', 'method_configs.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)
        
        # Load method configurations for thresholds
        with open(method_config_path, 'r') as f:
            self.method_configs = json.load(f)
        
    def execute_task(self, task: Task) -> Dict:
        """
        Provide feature recommendations based on test results
        
        :param task: Task object containing:
            - data: Dict containing:
                - dataframe: pandas DataFrame
                - test_results: Dict of test results for each feature
                - metadata: Dict of feature metadata
                - target_column: str or None
                - feature_count: int or None
        :return: Dictionary with feature recommendations
        """
        # Check if data is a dictionary and contains a dataframe
        if not isinstance(task.data, dict) or not isinstance(task.data.get('dataframe'), pd.DataFrame):
            raise ValueError("Task data must be a dictionary containing a pandas DataFrame under 'dataframe' key")
            
        test_results = task.data.get('test_results', {})
        metadata = task.data.get('metadata', {})
        target_column = task.data.get('target_column')
        
        total_features = task.data.get('dataframe').shape[1]
        feature_count = task.data.get('feature_count', int(total_features * 0.6))
        print(f">>>>>Total features: {total_features}, Feature count: {feature_count}")
        recommendations = self._get_recommendations(
            test_results=test_results,
            metadata=metadata,
            target_column=target_column,
            feature_count=feature_count
        )
        return recommendations

    def _get_recommendations(self, test_results: Dict, metadata: Dict, 
                           target_column: str, feature_count: int) -> Dict:
        """Get feature recommendations using LLM"""
        
        # Get prompts using PromptManager
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='feature_recommender',
            llm_provider=self.llm_provider,
            prompt_type='main',
            test_results=json.dumps(test_results, indent=2),
            metadata=json.dumps(metadata, indent=2),
            target_column=target_column,
            feature_count=feature_count
        )

        # Get provider config
        provider_config = self.model_config.get(self.llm_provider, {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.3,
            "max_tokens": 2000,
            "n": 1,
            "stop": None
        })
        
        # Prepare configuration
        configuration = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": provider_config["temperature"],
            "max_tokens": provider_config["max_tokens"],
            "n": provider_config["n"],
            "stop": provider_config["stop"]
        }

        # Use AI handler
        response, token_cost_summary = self.ai_handler.route_to(
            llm_provider=self.llm_provider,
            configuration=configuration,
            model=provider_config['model']
        )

        # Handle response
        try:
            if isinstance(response, dict):  # For Cortex
                content = response['choices'][0]['message']['content']
            elif hasattr(response, 'choices'):  # For OpenAI
                content = response.choices[0].message.content
            else:  # For other providers
                content = response
                
            # Clean the response string to extract only the JSON content
            cleaned_str = content.strip()
            # Remove markdown code block markers if present
            cleaned_str = cleaned_str.replace('```json', '').replace('```', '')
            # Find the actual JSON content between { }
            start_idx = cleaned_str.find('{')
            end_idx = cleaned_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                cleaned_str = cleaned_str[start_idx:end_idx + 1]
            
            recommendations = json.loads(cleaned_str)
            return self._validate_recommendations(recommendations, feature_count)
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            print(">>> Error parsing response:", str(e))  # temp
            return {
                "recommendations": [],
                "summary": {
                    "total_features": len(test_results),
                    "selected_count": 0,
                    "selection_criteria": "Error in generating recommendations"
                }
            }

    def _validate_recommendations(self, recommendations: Dict, desired_count: int) -> Dict:
        """Validate and normalize recommendations"""
        if not isinstance(recommendations, dict) or "recommendations" not in recommendations:
            return {
                "recommendations": [],
                "summary": {
                    "total_features": 0,
                    "selected_count": 0,
                    "selection_criteria": "Invalid recommendation format"
                }
            }
            
        # Validate each recommendation
        valid_recommendations = []
        for rec in recommendations["recommendations"]:
            if self._is_valid_recommendation(rec):
                valid_recommendations.append(rec)
        
        # Update summary
        recommendations["recommendations"] = valid_recommendations
        recommendations["summary"]["selected_count"] = sum(
            1 for rec in valid_recommendations if rec.get("selected", False)
        )
        
        return recommendations

    def _is_valid_recommendation(self, recommendation: Dict) -> bool:
        """Validate individual recommendation"""
        required_keys = ["feature_name", "status", "explanation", "priority", "selected"]
        if not all(key in recommendation for key in required_keys):
            return False
            
        # Validate status
        if recommendation["status"] not in ["R", "G", "Y"]:
            return False
            
        # Validate priority
        if recommendation["priority"] not in ["high", "medium", "low"]:
            return False
            
        return True

    def get_validation_params(self, response, task):
        """
        Get parameters for validation
        
        :param response: The response to validate
        :param task: Task object containing:
            - data: Dict containing:
                - dataframe: pandas DataFrame
                - test_results: Dict of test results
                - metadata: Dict of feature metadata
                - target_column: str or None
                - feature_count: int or None
        """
        # Check if data is a dictionary and contains a dataframe
        if not isinstance(task.data, dict) or not isinstance(task.data.get('dataframe'), pd.DataFrame):
            raise ValueError("Task data must be a dictionary containing a pandas DataFrame under 'dataframe' key")

        test_results = task.data.get('test_results', {})
        feature_count = task.data.get('feature_count')

        # Get validation prompts
        prompts = self.prompt_manager.get_prompt(
            agent_type='feature_recommender',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response,
            test_results=test_results,
            feature_count=feature_count
        )
        return prompts 