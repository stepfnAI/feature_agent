import os
from typing import List, Dict, Any
import pandas as pd
from sfn_blueprint import SFNAgent
from sfn_blueprint import Task
from sfn_blueprint import SFNAIHandler
from sfn_blueprint import SFNPromptManager
from feature_selection_agent.config.model_config import MODEL_CONFIG, DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER

class SFNFeatureSuggestionAgent(SFNAgent):
    def __init__(self, llm_provider='openai'):
        super().__init__(name="Feature Suggestion", role="To generate feature suggestions")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["feature_suggester"]
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)

    def execute_task(self, task: Task) -> List[str]:
        """
        Execute the feature suggestion task.
        
        :param task: Task object containing the data and category
        :return: List of feature suggestions
        """
        if not isinstance(task.data, dict) or 'df' not in task.data or 'category' not in task.data:
            raise ValueError("Task data must be a dictionary containing 'df' and 'category' keys")

        df = task.data['df']
        category = task.data['category']

        columns = df.columns.tolist()
        sample_records = df.head(3).to_dict(orient='records')
        describe_dict = df.describe().to_dict()

        suggestions = self._generate_suggestions(columns, sample_records, describe_dict, category)
        return suggestions

    def _generate_suggestions(self, columns: List[str], sample_records: List[Dict[str, Any]], 
                            describe_dict: Dict[str, Dict[str, float]], category: str) -> List[str]:
        """
        Generate feature suggestions based on the data and category.
        """
        kwargs = {
            'columns': columns,
            'samples': sample_records,
            'describe': describe_dict,
            'category': category
        }
        # Get prompts using PromptManager
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='feature_suggester', 
            llm_provider=self.llm_provider,
            **kwargs
        )
        
        # Get provider config or use default if not found
        provider_config = self.model_config.get(self.llm_provider, {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.3,
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
        
        # Handle response based on provider
        try:
            if isinstance(response, dict):  # For Cortex
                content = response['choices'][0]['message']['content']
            elif hasattr(response, 'choices'):  # For OpenAI
                content = response.choices[0].message.content
            else:  # For other providers or direct string response
                content = response
                
            suggestions_text = content.strip()
            return self._parse_suggestions(suggestions_text)
        except Exception as e:
            print(f"Error processing response: {e}")
            return []

    def _parse_suggestions(self, suggestions_text: str) -> List[str]:
        """
        Parse the suggestions text into a list of individual suggestions.
        
        :param suggestions_text: Raw text of suggestions from the OpenAI model
        :return: List of individual suggestions
        """
        # Split the text by newlines and remove any empty lines
        suggestions = [line.strip() for line in suggestions_text.split('\n') if line.strip()]
        
        # Remove numbering from each suggestion
        suggestions = [suggestion.split('. ', 1)[-1] for suggestion in suggestions]
        
        return suggestions

    def get_validation_params(self, response, task):
        """
        Get parameters for validation
        :param response: The response from execute_task to validate (list of suggestions)
        :param task: The validation task containing the DataFrame and category
        :return: Dictionary with validation parameters
        """
        if not isinstance(task.data, dict) or 'df' not in task.data or 'category' not in task.data:
            raise ValueError("Task data must be a dictionary containing 'df' and 'category' keys")

        df = task.data['df']
        category = task.data['category']

        # Get validation prompts from prompt manager
        prompts = self.prompt_manager.get_prompt(
            agent_type='feature_suggester',
            llm_provider='openai',
            prompt_type='validation',
            actual_output=response,
            columns=df.columns.tolist(),
            category=category,
            samples=df.head(3).to_dict(orient='records'),
            describe=df.describe().to_dict()
        )
        return prompts