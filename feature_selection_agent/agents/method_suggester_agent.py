from typing import Dict, List
import pandas as pd
from sfn_blueprint import SFNAgent
from sfn_blueprint import Task
from sfn_blueprint import SFNAIHandler
import os
from sfn_blueprint import SFNPromptManager
from feature_selection_agent.config.model_config import MODEL_CONFIG, DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER
import json

class SFNMethodSuggesterAgent(SFNAgent):
    """Agent responsible for suggesting appropriate feature selection methods based on dataset characteristics"""
    
    def __init__(self, llm_provider='openai'):
        super().__init__(name="Method Suggester", role="Method Advisor")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["method_suggester"]
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        method_config_path = os.path.join(parent_path, 'config', 'method_configs.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)
        
        # Load method configurations
        with open(method_config_path, 'r') as f:
            self.method_pool = json.load(f)
        
    def execute_task(self, task: Task) -> Dict:
        """
        Suggests feature selection methods based on dataset characteristics
        
        :param task: Task object containing:
            - data: Dict containing:
                - dataframe: pandas DataFrame
                - metadata: Dict of feature metadata
                - target_column: str or None
                - target_type: str or None
        :return: Dictionary with suggested methods
        """
        # Check if data is a dictionary and contains a dataframe
        if not isinstance(task.data, dict) or not isinstance(task.data.get('dataframe'), pd.DataFrame):
            raise ValueError("Task data must be a dictionary containing a pandas DataFrame under 'dataframe' key")
            
        metadata = task.data.get('metadata', {})
        target_column = task.data.get('target_column')
        
        suggestions = self._suggest_methods(
            metadata=metadata,
            has_target=target_column is not None,
            target_type=task.data.get('target_type')
        )
        return suggestions

    def _suggest_methods(self, metadata: Dict, has_target: bool, target_type: str = None) -> Dict:
        """
        Suggest appropriate feature selection methods
        """
        # Get prompts using PromptManager
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='method_suggester',
            llm_provider=self.llm_provider,
            prompt_type='main',
            metadata=json.dumps(metadata, indent=2),
            has_target=has_target,
            target_type=target_type or "None",
            method_pool=json.dumps(self.method_pool, indent=2)
        )

        # Get provider config or use default if not found
        provider_config = self.model_config.get(self.llm_provider, {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.3,
            "max_tokens": 1000,
            "n": 1,
            "stop": None
        })
        
        # Prepare the configuration for the API call
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

        # Use the AI handler to route the request
        response, token_cost_summary = self.ai_handler.route_to(
            llm_provider=self.llm_provider,
            configuration=configuration,
            model=provider_config['model']
        )
        print('>>>>>>response of method suggester agent', response) #temp
        # Handle response based on provider
        try:
            if isinstance(response, dict):  # For Cortex
                content = response['choices'][0]['message']['content']
            elif hasattr(response, 'choices'):  # For OpenAI
                content = response.choices[0].message.content
            else:  # For other providers or direct string response
                content = response
            print('>>>>>>content of method suggester agent', content) #temp                
            # Clean the response string to extract only the JSON content
            cleaned_str = content.strip()
            # Remove markdown code block markers if present
            cleaned_str = cleaned_str.replace('```json', '').replace('```', '')
            # Find the actual JSON content between { }
            start_idx = cleaned_str.find('{')
            end_idx = cleaned_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                cleaned_str = cleaned_str[start_idx:end_idx + 1]
            
            suggestions = json.loads(cleaned_str)
            print('>>>>>>cleaned suggestions of method suggester agent', suggestions) #temp
            return self._validate_suggestions(suggestions, has_target)
        except (json.JSONDecodeError, AttributeError, KeyError):
            return {"suggested_methods": []}

    def _validate_suggestions(self, suggestions: Dict, has_target: bool) -> Dict:
        """
        Validate and normalize the suggested methods
        """
        if not isinstance(suggestions, dict) or "suggested_methods" not in suggestions:
            return {"suggested_methods": []}
            
        validated_methods = []
        for method in suggestions["suggested_methods"]:
            if self._is_valid_method(method, has_target):
                validated_methods.append(method)
                
        return {"suggested_methods": validated_methods}
        
    def _is_valid_method(self, method: Dict, has_target: bool) -> bool:
        """
        Validate individual method suggestion
        """
        required_keys = ["method_name", "category", "reason", "priority"]
        if not all(key in method for key in required_keys):
            return False
            
        # Check if method exists in method pool
        method_category = "with_target" if has_target else "without_target"
        available_methods = []
        for category in self.method_pool[method_category].values():
            available_methods.extend(category.keys())
            
        return method["method_name"] in available_methods

    def get_validation_params(self, response, task):
        """
        Get parameters for validation
        
        :param response: The response to validate
        :param task: Task object containing:
            - data: Dict containing:
                - dataframe: pandas DataFrame
                - metadata: Dict of feature metadata
                - target_column: str or None
                - target_type: str or None
        """
        # Check if data is a dictionary and contains a dataframe
        if not isinstance(task.data, dict) or not isinstance(task.data.get('dataframe'), pd.DataFrame):
            raise ValueError("Task data must be a dictionary containing a pandas DataFrame under 'dataframe' key")

        metadata = task.data.get('metadata', {})
        target_column = task.data.get('target_column')

        # Get validation prompts from prompt manager
        prompts = self.prompt_manager.get_prompt(
            agent_type='method_suggester',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response,
            metadata=metadata,
            has_target=target_column is not None,
            target_type=task.data.get('target_type') or "None"
        )
        return prompts 