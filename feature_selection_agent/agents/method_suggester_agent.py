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
        try:
            with open(method_config_path, 'r') as f:
                self.method_pool = json.load(f)
            print(f">>> Successfully loaded method pool with {len(self.method_pool.get('methods_without_target', []))} methods without target")
            print(f">>> and {len(self.method_pool.get('methods_with_target', []))} methods with target")
        except Exception as e:
            print(f">>> Error loading method configs: {str(e)}")
            raise
        
    def execute_task(self, task: Task) -> Dict:
        """
        Suggests feature selection methods based on dataset characteristics
        """
        print(">>> Starting execute_task in MethodSuggesterAgent")  # Debug log
        print(f">>> Task data keys: {task.data.keys() if isinstance(task.data, dict) else 'Not a dict'}")  # Debug log
        
        # Check if data is a dictionary and contains a dataframe
        if not isinstance(task.data, dict) or not isinstance(task.data.get('dataframe'), pd.DataFrame):
            raise ValueError("Task data must be a dictionary containing a pandas DataFrame under 'dataframe' key")
            
        metadata = task.data.get('metadata', {})
        target_column = task.data.get('target_column')
        
        print(f">>> Metadata: {metadata}")  # Debug log
        print(f">>> Target column: {target_column}")  # Debug log
        
        suggestions = self._suggest_methods(
            metadata=metadata,
            has_target=target_column is not None,
            target_type=task.data.get('target_type')
        )
        print(f">>> Generated suggestions: {suggestions}")  # Debug log
        return suggestions

    def _suggest_methods(self, metadata: Dict, has_target: bool, target_type: str = None) -> Dict:
        """
        Suggest appropriate feature selection methods
        """
        print(">>> Starting _suggest_methods")  # Debug log
        print(f">>> Input params - metadata: {metadata}, has_target: {has_target}, target_type: {target_type}")  # Debug log
        
        # Get prompts using PromptManager
        try:
            system_prompt, user_prompt = self.prompt_manager.get_prompt(
                agent_type='method_suggester',
                llm_provider=self.llm_provider,
                prompt_type='main',
                metadata=json.dumps(metadata, indent=2),
                has_target=has_target,
                target_type=target_type or "None",
                method_pool=json.dumps(self.method_pool, indent=2)
            )
            print(">>> Successfully got prompts")  # Debug log
            print(f">>> System prompt: {system_prompt}")  # Debug log
            print(f">>> User prompt: {user_prompt}")  # Debug log
        except Exception as e:
            print(f">>> Error getting prompts: {str(e)}")  # Debug log
            raise

        # Get provider config or use default if not found
        provider_config = self.model_config.get(self.llm_provider, {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.3,
            "max_tokens": 1000,
            "n": 1,
            "stop": None
        })
        print(f">>> Using provider config: {provider_config}")  # Debug log
        
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
        print(">>> Starting validation")  # Debug log
        print(f">>> Suggestions to validate: {suggestions}")  # Debug log
        
        if not isinstance(suggestions, dict) or "suggested_methods" not in suggestions:
            print(">>> Invalid suggestions format")  # Debug log
            return {"suggested_methods": []}
            
        validated_methods = []
        method_pool_key = "methods_with_target" if has_target else "methods_without_target"
        available_methods = self.method_pool.get(method_pool_key, [])
        
        # Create a map of available method names for quick lookup
        available_method_names = {method["name"]: method for method in available_methods}
        print(f">>> Available methods: {list(available_method_names.keys())}")  # Debug log
        
        for method in suggestions["suggested_methods"]:
            try:
                # Check required keys
                if not all(key in method for key in ["method_name", "category", "reason", "priority"]):
                    print(f">>> Missing required keys in method: {method}")  # Debug log
                    continue
                    
                # Check if method exists in method pool
                if method["method_name"] in available_method_names:
                    # Verify the category matches
                    pool_method = available_method_names[method["method_name"]]
                    if method["category"] == pool_method["category"]:
                        validated_methods.append(method)
                        print(f">>> Validated method: {method['method_name']}")  # Debug log
                    else:
                        print(f">>> Category mismatch for method: {method['method_name']}")  # Debug log
                else:
                    print(f">>> Method not found in pool: {method['method_name']}")  # Debug log
                    
            except Exception as e:
                print(f">>> Error validating method {method}: {str(e)}")  # Debug log
                continue
        
        print(f">>> Validation complete. Valid methods: {len(validated_methods)}")  # Debug log
        return {"suggested_methods": validated_methods}

    def _is_valid_method(self, method: Dict, has_target: bool) -> bool:
        """
        Validate individual method suggestion
        """
        # This method is now handled within _validate_suggestions
        return True

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