from typing import Dict, List
import pandas as pd
from sfn_blueprint import SFNAgent
from sfn_blueprint import Task
from sfn_blueprint import SFNAIHandler
import os
from sfn_blueprint import SFNPromptManager
from feature_selection_agent.config.model_config import MODEL_CONFIG, DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER
import json

class SFNFieldMappingAgent(SFNAgent):
    """Agent responsible for identifying critical fields like CUST_ID, BILLING_DATE, TARGET, REVENUE"""
    
    def __init__(self, llm_provider='openai'):
        super().__init__(name="Field Mapping", role="Field Identifier")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["field_mapper"]
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)
        
    def execute_task(self, task: Task) -> Dict[str, str]:
        """
        Maps dataset columns to critical fields
        Returns: Dictionary mapping critical fields to dataset columns
        """
        if not isinstance(task.data, pd.DataFrame):
            raise ValueError("Task data must be a pandas DataFrame")

        columns = task.data.columns.tolist()
        field_mappings = self._identify_fields(columns)
        return field_mappings

    def _identify_fields(self, columns: List[str]) -> Dict[str, str]:
        """
        Identify critical fields from the column names
        
        :param columns: List of column names in the dataset
        :return: Dictionary with mapped fields
        """
        # Get prompts using PromptManager
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='field_mapper', 
            llm_provider=self.llm_provider,
            prompt_type='main',
            columns=columns
        )

        # Get provider config or use default if not found
        provider_config = self.model_config.get(self.llm_provider, {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.5,
            "max_tokens": 500,
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
        print(f">>>>>>>Response: {response}")
        
        # Handle response based on provider
        try:
            if isinstance(response, dict):  # For Cortex
                content = response['choices'][0]['message']['content']
            elif hasattr(response, 'choices'):  # For OpenAI
                content = response.choices[0].message.content
            else:  # For other providers or direct string response
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
            
            mappings = json.loads(cleaned_str)
            return self._validate_mappings(mappings)
        except (json.JSONDecodeError, AttributeError, KeyError):
            return {
                "customer_id": None,
                "date_mapping": None,
                "revenue_mapping": None,
                "target": None
            }

    def _validate_mappings(self, mappings: Dict[str, str]) -> Dict[str, str]:
        """
        Validate and normalize the mappings returned by the model
        
        :param mappings: Dictionary of field mappings
        :return: Validated and normalized mappings
        """
        required_keys = ["customer_id", "date_mapping", "revenue_mapping", "target"]
        validated_mappings = {}
        
        for key in required_keys:
            validated_mappings[key] = mappings.get(key)
            
        return validated_mappings

    def get_validation_params(self, response, task):
        """
        Get parameters for validation
        :param response: The response from execute_task to validate (field mappings dictionary)
        :param task: The validation task containing the DataFrame
        :return: Dictionary with validation parameters
        """
        if not isinstance(task.data, pd.DataFrame):
            raise ValueError("Task data must be a pandas DataFrame")

        # Get validation prompts from prompt manager
        prompts = self.prompt_manager.get_prompt(
            agent_type='field_mapper',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response,
            columns=task.data.columns.tolist()
        )
        return prompts 