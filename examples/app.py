import sys
import os
import streamlit as st
from sfn_blueprint import SFNAgent, Task, SFNSessionManager
from sfn_blueprint import SFNDataLoader, setup_logger, SFNDataPostProcessor
from sfn_blueprint import SFNValidateAndRetryAgent
from feature_selection_agent.config.model_config import DEFAULT_LLM_PROVIDER
from feature_selection_agent.agents.field_mapping_agent import SFNFieldMappingAgent
from feature_selection_agent.agents.feature_classification_agent import FeatureClassificationAgent
from feature_selection_agent.agents.method_suggester_agent import SFNMethodSuggesterAgent
from feature_selection_agent.agents.feature_selection_executor_agent import SFNFeatureSelectionExecutorAgent
from feature_selection_agent.agents.feature_recommender_agent import SFNFeatureRecommenderAgent
from feature_selection_agent.views.streamlit_views import StreamlitView

class FeatureSelectionApp:
    def __init__(self):
        self.view = StreamlitView(title="Feature Selection App")
        self.session = SFNSessionManager()
        self.logger, self.handler = setup_logger()
        
    def run_app(self):
        """Main application flow"""
        # Initialize UI
        self._initialize_ui()
        
        # Step 1: Data Upload and Preview
        if not self.session.get('data_upload_complete'):
            self._handle_data_upload()
            return
        
        # Display summary of completed steps
        self._display_step_summary(current_step=2)
        
        # Step 2: Field Mapping and Classification
        if not self.session.get('field_mapping_complete'):
            self._handle_field_mapping()
            return
        
        # Display summary of completed steps
        self._display_step_summary(current_step=3)
        
        # Step 3: Feature Selection Methods
        if not self.session.get('method_selection_complete'):
            self._handle_method_selection()
            return
        
        # Display summary of completed steps
        self._display_step_summary(current_step=4)
        
        # Step 4: Final Feature Selection Interface
        if not self.session.get('feature_selection_complete'):
            self._handle_feature_selection()
            return
        
        # Display summary of completed steps
        self._display_step_summary(current_step=5)
        
        # Step 5: Post Processing
        if not self.session.get('post_processing_complete'):
            self._handle_post_processing()
            return

    def _initialize_ui(self):
        """Initialize the UI components"""
        col1, col2 = self.view.create_columns([7, 1])
        with col1:
            self.view.display_title()
        with col2:
            if self.view.display_button("🔄", key="reset_button"):
                self.session.clear()
                self.view.rerun_script()

    def _handle_data_upload(self):
        """Step 1: Handle data upload and preview"""
        self.view.display_header("Step 1: Data Upload and Preview")
        self.view.display_markdown("---")
        
        # Always show data preview if data is loaded
        df = self.session.get('df')
        if df is not None:
            self.view.display_subheader("Data Preview")
            self.view.display_dataframe(df.head())
            
        
        # Show file uploader only if no data is loaded
        uploaded_file = self.view.file_uploader(
            "Choose a CSV or Excel file", 
            accepted_types=["csv", "xlsx", "json", "parquet"],
            key="data_upload"
        )

        if uploaded_file is not None:
            with self.view.display_spinner('Loading data...'):
                try:
                    data_loader = SFNDataLoader()
                    file_path = self.view.save_uploaded_file(uploaded_file)
                    load_task = Task("Load the uploaded file", data=uploaded_file, path=file_path)
                    
                    df = data_loader.execute_task(load_task)
                    self.view.delete_uploaded_file(file_path)
                    
                    self.session.set('df', df)
                    self.view.show_message(f"✅ Data loaded successfully. Shape: {df.shape}", "success")
                    self.view.display_subheader("Data Preview")
                    self.view.display_dataframe(df.head())
                    if not self.session.get('data_upload_complete'):
                        if self.view.display_button("Confirm and Continue", key="confirm_data"):
                            self.session.set('data_upload_complete', True)
                            self.view.rerun_script()
                            return
                    
                except Exception as e:
                    self.logger.error(f"Error loading file: {e}")
                    self.view.show_message(f"❌ Error loading file: {str(e)}", "error")

    def _handle_field_mapping(self):
        """Step 2: Handle field mapping and classification"""
        self.view.display_header("Step 2: Field Mapping and Classification")
        self.view.display_markdown("---")
        
        df = self.session.get('df')
        
        # First, handle field mapping if not done
        if not self.session.get('field_mapping_done'):
            with self.view.display_spinner('🤖 AI is mapping critical fields...'):
                # try:
                mapping_agent = SFNFieldMappingAgent()
                mapping_task = Task("Map fields", data=df)
                validation_task = Task("Validate field mapping", data=df)
                
                validate_and_retry_agent = SFNValidateAndRetryAgent(
                    llm_provider=DEFAULT_LLM_PROVIDER,
                    for_agent='field_mapper'
                )
                
                field_mappings, validation_message, is_valid = validate_and_retry_agent.complete(
                    agent_to_validate=mapping_agent,
                    task=mapping_task,
                    validation_task=validation_task,
                    method_name='execute_task',
                    get_validation_params='get_validation_params',
                    max_retries=2,
                    retry_delay=3.0
                )
                print(f">>>>>>>Field mappings: {field_mappings}")
                if is_valid:
                    self.session.set('suggested_mappings', field_mappings)  # Store original suggestions
                    self.session.set('field_mappings', field_mappings)
                    self.session.set('field_mapping_done', True)
                else:
                    self.view.show_message("❌ AI couldn't generate valid field mappings.", "error")
                    self.view.show_message(validation_message, "warning")
                    return

        # Display mapping interface for user verification/modification
        if not self.session.get('mapping_verified'):
            self._display_mapping_interface()
            return

        # Only proceed with classification after mapping is verified
        if not self.session.get('classification_done'):
            with self.view.display_spinner('Classifying features...'):
                classifier = FeatureClassificationAgent()
                # Get verified critical fields
                critical_fields = [v for v in self.session.get('field_mappings').values() if v is not None]
                # Filter DataFrame to exclude critical fields
                features_for_classification = [col for col in df.columns if col not in critical_fields]
                classification_task = Task("Classify features", 
                                        data=df[features_for_classification])
                classifications, metadata = classifier.execute_task(classification_task)
                
                self.session.set('feature_classifications', classifications)
                self.session.set('feature_metadata', metadata)
                self.session.set('classification_done', True)

        if self.session.get('mapping_verified') and self.session.get('classification_done'):
            if self.view.display_button("Continue to Next Step"):
                self.session.set('field_mapping_complete', True)
                self.view.rerun_script()

    def _display_mapping_interface(self):
        """Display interface for verifying and modifying field mappings"""
        self.view.display_subheader("AI Suggested Critical Field Mappings")
        
        # Get current mappings and available columns
        suggested_mappings = self.session.get('suggested_mappings')
        current_mappings = self.session.get('field_mappings')
        all_columns = list(self.session.get('df').columns)
        
        # Format message similar to join agent style
        message = "🎯 AI Suggested Mappings:\n"
        for field, mapped_col in suggested_mappings.items():
            message += f"- {field}:  **{mapped_col or 'Not Found'}**\n"
        
        # Display using show_message with info type
        self.view.show_message(message, "info")
        
        self.view.display_markdown("---")
        
        # Then show options to proceed
        action = self.view.radio_select(
            "How would you like to proceed?",
            options=[
                "Use AI Recommended Mappings",
                "Select Columns Manually"
            ],
            key="mapping_choice"
        )
        
        if action == "Use AI Recommended Mappings":
            if self.view.display_button("Confirm Mappings"):
                self.session.set('mapping_verified', True)
                self.view.rerun_script()
            
        else:  # Select Columns Manually
            # Required fields
            required_fields = ["CUST_ID", "BILLING_DATE", "REVENUE"]
            optional_fields = ["TARGET", "PRODUCT"]
            
            modified_mappings = {}
            
            # Handle required fields
            self.view.display_subheader("Required Fields")
            for field in required_fields:
                current_value = current_mappings.get(field)
                modified_mappings[field] = self.view.select_box(
                    f"Select column for {field}",
                    options=[""] + all_columns,
                    index=all_columns.index(current_value) + 1 if current_value in all_columns else 0
                )
            
            # Handle optional fields
            self.view.display_subheader("Optional Fields")
            for field in optional_fields:
                current_value = current_mappings.get(field)
                value = self.view.select_box(
                    f"Select column for {field} (optional)",
                    options=[""] + all_columns,
                    index=all_columns.index(current_value) + 1 if current_value in all_columns else 0
                )
                if value:  # Only add if a column was selected
                    modified_mappings[field] = value
            
            # Additional critical fields
            if self.view.display_button("+ Add Critical Field"):
                num_additional = self.session.get('additional_fields', 0) + 1
                self.session.set('additional_fields', num_additional)
            
            # Display additional field inputs if any
            num_additional = self.session.get('additional_fields', 0)
            if num_additional > 0:
                self.view.display_subheader("Additional Critical Fields")
                for i in range(num_additional):
                    value = self.view.select_box(
                        f"Select additional critical column #{i+1}",
                        options=[""] + [col for col in all_columns 
                                      if col not in modified_mappings.values()],
                    )
                    if value:  # Only add if a column was selected
                        modified_mappings[f"CRITICAL_{i+1}"] = value
            
            # Confirm modified mappings
            if self.view.display_button("Confirm Modified Mappings"):
                # Validate that required fields are mapped
                missing_required = [f for f in required_fields if not modified_mappings.get(f)]
                if missing_required:
                    self.view.show_message(
                        f"❌ Please map required fields: {', '.join(missing_required)}", 
                        "error"
                    )
                else:
                    self.session.set('field_mappings', modified_mappings)
                    self.session.set('mapping_verified', True)
                    self.view.rerun_script()

    def _handle_method_selection(self):
        """Step 3: Handle feature selection method suggestions and selection"""
        self.view.display_header("Step 3: Feature Selection Methods")
        self.view.display_markdown("---")
        
        # Get method suggestions if not already done
        if not self.session.get('methods_suggested'):
            with self.view.display_spinner('🤖 AI is suggesting feature selection methods...'):
                # try:
                suggester = SFNMethodSuggesterAgent()
                metadata = self.session.get('feature_metadata')
                field_mappings = self.session.get('field_mappings')
                
                suggest_task = Task("Suggest methods", data={
                    'dataframe': self.session.get('df'),
                    'metadata': metadata,
                    'target_column': field_mappings.get('target')
                })
                
                validation_task = Task("Validate method suggestions", data={
                    'dataframe': self.session.get('df'),
                    'metadata': metadata,
                    'target_column': field_mappings.get('target')
                })
                
                # Store metadata and target info in session for agent to access
                self.session.set('current_metadata', metadata)
                self.session.set('current_target', field_mappings.get('target'))
                
                validate_and_retry_agent = SFNValidateAndRetryAgent(
                    llm_provider=DEFAULT_LLM_PROVIDER,
                    for_agent='method_suggester'
                )
                
                suggested_methods, validation_message, is_valid = validate_and_retry_agent.complete(
                    agent_to_validate=suggester,
                    task=suggest_task,
                    validation_task=validation_task,
                    method_name='execute_task',
                    get_validation_params='get_validation_params',
                    max_retries=2,
                    retry_delay=3.0
                )
                
                if is_valid:
                    self.session.set('suggested_methods', suggested_methods)
                    self.session.set('methods_suggested', True)
                else:
                    self.view.show_message("❌ AI couldn't generate valid method suggestions.", "error")
                    self.view.show_message(validation_message, "warning")
                    return
                        
                # except Exception as e:
                #     self.logger.error(f"Error in method suggestion: {e}")
                #     self.view.show_message("❌ Error suggesting methods", "error")
                #     return
        
        # Display method selection interface
        self._display_method_selection_interface()
        
        if self.view.display_button("Confirm Selected Methods"):
            with self.view.display_spinner('Running feature selection methods...'):
                # try:
                selected_method_names = self.session.get('selected_methods')
                suggested_methods = self.session.get('suggested_methods')
                
                # Get full method information for selected methods
                selected_method_info = [
                    method for method in suggested_methods["suggested_methods"] 
                    if method["method_name"] in selected_method_names
                ]
                
                print(">>> Selected methods with full info:", selected_method_info)  # temp
                
                executor = SFNFeatureSelectionExecutorAgent()
                execution_task = Task("Execute methods", data={
                    'dataframe': self.session.get('df'),
                    'methods': selected_method_info,  # Pass the full method info
                    'field_mappings': self.session.get('field_mappings'),
                    'metadata': self.session.get('feature_metadata')
                })
                
                print(">>> Executing feature selection methods with task:", execution_task.data)  # temp
                test_results = executor.execute_task(execution_task)
                print(">>> Feature selection test results:", test_results)  # temp
                
                self.session.set('test_results', test_results)
                self.view.show_message("✅ Feature selection methods executed successfully", "success")
                
                # Now we can proceed to the next step
                self.session.set('method_selection_complete', True)
                self.view.rerun_script()
                # except Exception as e:
                #     print(">>> Error in feature selection execution:", str(e))  # temp
                #     self.logger.error(f"Error executing feature selection methods: {e}")
                #     self.view.show_message("❌ Error executing feature selection methods", "error")
                #     return

    def _handle_feature_selection(self):
        """Step 4: Handle final feature selection interface"""
        self.view.display_header("Step 4: Feature Selection")
        self.view.display_markdown("---")
        
        # Get feature recommendations if not done
        if not self.session.get('recommendations_generated'):
            with self.view.display_spinner('🤖 AI is analyzing results...'):
                # try:
                print(">>> Starting feature recommendation process")  # temp
                print(">>> Test results from session:", self.session.get('test_results'))  # temp
                
                recommender = SFNFeatureRecommenderAgent()
                recommend_task = Task("Get recommendations", data={
                    'dataframe': self.session.get('df'),
                    'test_results': self.session.get('test_results'),
                    'metadata': self.session.get('feature_metadata'),
                    'target_column': self.session.get('field_mappings').get('target')
                })
                
                print(">>> Recommendation task data:", recommend_task.data)  # temp
                
                validation_task = Task("Validate recommendations", data={
                    'dataframe': self.session.get('df'),
                    'test_results': self.session.get('test_results'),
                    'metadata': self.session.get('feature_metadata'),
                    'target_column': self.session.get('field_mappings').get('target')
                })
                
                # Store additional data in session
                self.session.set('current_test_results', self.session.get('test_results'))
                self.session.set('current_metadata', self.session.get('feature_metadata'))
                self.session.set('current_target', self.session.get('field_mappings').get('target'))
                
                validate_and_retry_agent = SFNValidateAndRetryAgent(
                    llm_provider=DEFAULT_LLM_PROVIDER,
                    for_agent='feature_recommender'
                )
                
                recommendations, validation_message, is_valid = validate_and_retry_agent.complete(
                    agent_to_validate=recommender,
                    task=recommend_task,
                    validation_task=validation_task,
                    method_name='execute_task',
                    get_validation_params='get_validation_params',
                    max_retries=2,
                    retry_delay=3.0
                )
                
                print(">>> Recommendations received:", recommendations)  # temp
                print(">>> Validation message:", validation_message)  # temp
                print(">>> Is valid:", is_valid)  # temp
                
                if is_valid:
                    self.session.set('recommendations', recommendations)
                    self.session.set('recommendations_generated', True)
                else:
                    self.view.show_message("❌ AI couldn't generate valid recommendations.", "error")
                    self.view.show_message(validation_message, "warning")
                    return
                        
                # except Exception as e:
                #     self.logger.error(f"Error generating recommendations: {e}")
                #     self.view.show_message("❌ Error generating recommendations", "error")
                #     return
        
        # Display feature selection interface
        self._display_feature_selection_interface()

    def _handle_post_processing(self):
        """Step 5: Handle post processing and output"""
        self.view.display_header("Step 5: Post Processing")
        self.view.display_markdown("---")
        
        # Create final dataset with selected features
        if self.session.get('final_df') is None:
            selected_features = [rec["feature_name"] for rec in self.session.get('recommendations')["recommendations"] 
                               if rec["selected"]]
            field_mappings = self.session.get('field_mappings')
            critical_fields = [v for v in field_mappings.values() if v is not None]
            
            final_columns = critical_fields + selected_features
            final_df = self.session.get('df')[final_columns].copy()
            self.session.set('final_df', final_df)
        
        # Display operation options
        operation_type = self.view.radio_select(
            "Choose an operation:",
            ["View Selected Features", "Download Data","Finish"]
        )
        
        self._handle_post_processing_operation(operation_type)

    def _display_mapping_and_classification_results(self):
        """Helper method to display mapping and classification results"""
        # Display field mappings
        self.view.display_subheader("Critical Field Mappings")
        field_mappings = self.session.get('field_mappings')
        for field, mapped_col in field_mappings.items():
            self.view.display_markdown(f"**{field}**: {mapped_col or 'Not Found'}")
        
        # Display feature classifications
        self.view.display_subheader("Feature Classifications")
        classifications = self.session.get('feature_classifications')
        for feature, info in classifications.items():
            if feature not in field_mappings.values():
                self.view.display_markdown(
                    f"**{feature}** ({info['category']}): {info['description']}"
                )

    def _display_method_selection_interface(self):
        """Helper method to display method selection interface"""
        suggested_methods = self.session.get('suggested_methods')
        selected_methods = self.session.get('selected_methods', [])
        
        self.view.display_subheader("Feature Selection Methods")
        
        # First show AI recommendations message
        recommended_methods = [method["method_name"] for method in suggested_methods["suggested_methods"]]
        message = "🎯 AI Recommended Methods:\n"
        for method in suggested_methods["suggested_methods"]:
            message += f"- **{method['method_name']}** ({method['priority']} priority)\n"
        self.view.show_message(message, "info")
        
        self.view.display_markdown("---")
        self.view.display_markdown("Select methods to use for feature selection:")
        
        # Get all available methods from method pool
        all_methods = self._get_all_available_methods()  # You'll need to implement this
        
        # Display all methods with recommendations highlighted
        for method_info in all_methods:
            method_name = method_info["name"]
            is_recommended = method_name in recommended_methods
            is_selected = method_name in selected_methods
            
            col1, col2 = self.view.create_columns([1, 11])
            with col1:
                if self.view.checkbox("", 
                                    value=is_selected or is_recommended,  # Pre-select recommended methods
                                    key=f"method_{method_name}",
                                    label_visibility="collapsed"):
                    if method_name not in selected_methods:
                        selected_methods.append(method_name)
                else:
                    if method_name in selected_methods:
                        selected_methods.remove(method_name)
            
            with col2:
                # If it's recommended, show the AI's reason
                if is_recommended:
                    rec_method = next(m for m in suggested_methods["suggested_methods"] 
                                    if m["method_name"] == method_name)
                    self.view.display_markdown(
                        f"**{method_name}** ({rec_method['priority']} priority) "
                        f"{'🤖 AI Recommended' if is_recommended else ''}\n\n"
                        f"{rec_method['reason']}"
                    )
                else:
                    # Show default description for non-recommended methods
                    self.view.display_markdown(
                        f"**{method_name}**\n\n"
                        f"{method_info['description']}"
                    )
        
        self.session.set('selected_methods', selected_methods)

    def _display_feature_selection_interface(self):
        """Helper method to display feature selection interface"""
        recommendations = self.session.get('recommendations')
        
        self.view.display_subheader("Feature Recommendations")
        
        # Display summary
        summary = recommendations["summary"]
        self.view.display_markdown(f"**Selection Summary**: {summary['selection_criteria']}")
        self.view.display_markdown(f"Selected {summary['selected_count']} out of {summary['total_features']} features")
        
        # Create DataFrame for the table
        table_data = []
        for rec in recommendations["recommendations"]:
            table_data.append({
                "Selected": "✅" if rec["selected"] else "❌",
                "Feature Name": rec["feature_name"],
                "Explanation": rec["explanation"],
                "Status": {"G": "GREEN", "Y": "YELLOW", "R": "RED"}[rec["status"]]
            })
        
        # Convert to DataFrame
        import pandas as pd
        df_table = pd.DataFrame(table_data)
        
        # Display interactive table with checkboxes
        for idx, row in df_table.iterrows():
            col1, col2, col3, col4 = self.view.create_columns([1, 2, 4, 1])
            with col1:
                is_selected = self.view.checkbox("", 
                                              value=row["Selected"] == "✅",
                                              key=f"checkbox_{idx}",
                                              label_visibility="collapsed")  # Simplified key
                recommendations["recommendations"][idx]["selected"] = is_selected
            with col2:
                self.view.display_markdown(f"**{row['Feature Name']}**")
            with col3:
                self.view.display_markdown(row["Explanation"])
            with col4:
                self.view.display_markdown(row["Status"])
        
        self.session.set('recommendations', recommendations)
        
        # Add confirmation button at the bottom
        self.view.display_markdown("---")
        if self.view.display_button("Confirm Feature Selection", key="confirm_features_btn"):
            self.session.set('feature_selection_complete', True)
            self.view.rerun_script()

    def _handle_post_processing_operation(self, operation_type: str):
        """Helper method to handle post processing operations"""
        if operation_type == "View Selected Features":
            self.view.display_dataframe(self.session.get('final_df'))
            
        elif operation_type == "Download Data":
            post_processor = SFNDataPostProcessor(self.session.get('final_df'))
            csv_data = post_processor.download_data('csv')
            self.view.create_download_button(
                label="Download CSV",
                data=csv_data,
                file_name="selected_features.csv",
                mime_type="text/csv"
            )
                        
        elif operation_type == "Finish":
            if self.view.display_button("Confirm Finish"):
                self.view.show_message("Thank you for using the Feature Selection App!", "success")
                self.session.clear()
                self.view.rerun_script()

    def _display_step_summary(self, current_step: int):
        """Display summary of completed steps"""
        if current_step <= 1:
            return
        
        if current_step == 2:
            # Step 1 Summary (Data Upload)
            if self.session.get('data_upload_complete'):
                self.view.display_header("Step 1: Data Upload and Preview")
                self.view.display_markdown("---")
                df = self.session.get('df')
                self.view.show_message(f"✅ Data loaded successfully. **Shape: {df.shape}**", "success")
                self.view.display_subheader("Data Preview")
                self.view.display_dataframe(df.head())
                self.view.display_markdown("---")
        
        if current_step == 3:
            # Step 2 Summary (Field Mapping)
            if self.session.get('field_mapping_complete'):
                self.view.display_header("Step 2: Critical Field Mappings")
                message = "✅ Critical Fields Mapped Successfully:\n"
                field_mappings = self.session.get('field_mappings')
                for field, mapped_col in field_mappings.items():
                    message += f"- {field}: **{mapped_col or 'Not Found'}**\n"
                self.view.show_message(message, "success")
                self.view.display_markdown("---")
        
        if current_step == 4:
            # Step 3 Summary (Method Selection)
            if self.session.get('method_selection_complete'):
                self.view.display_header("Step 3: Feature Selection Methods")
                selected_methods = self.session.get('selected_methods', [])
                message = "✅ Selected Feature Selection Methods:\n"
                for method in selected_methods:
                    message += f"- **{method}**\n"
                self.view.show_message(message, "success")
                self.view.display_markdown("---")
        
        if current_step == 5:
            # Step 4 Summary (Feature Selection)
            if self.session.get('feature_selection_complete'):
                self.view.display_header("Step 4: Feature Selection")
                recommendations = self.session.get('recommendations')
                selected_features = [rec["feature_name"] for rec in recommendations["recommendations"] 
                               if rec["selected"]]
                message = "✅ Selected Features:\n"
                for feature in selected_features:
                    message += f"- **{feature}**\n"
                self.view.show_message(message, "success")
                self.view.display_markdown("---")

    def _get_all_available_methods(self):
        """Get all available feature selection methods with their descriptions"""
        import json
        import os
        
        # Get target information to determine which methods are applicable
        field_mappings = self.session.get('field_mappings', {})
        has_target = field_mappings.get('target') is not None
        
        # Load method configurations from json file
        config_path = os.path.join(os.path.dirname(__file__), '..', 'feature_selection_agent', 
                                  'config', 'method_configs.json')
        with open(config_path, 'r') as f:
            method_configs = json.load(f)
        
        # Return all methods if target is present, otherwise only methods that don't require target
        if has_target:
            return method_configs['methods_without_target'] + method_configs['methods_with_target']
        return method_configs['methods_without_target']

if __name__ == "__main__":
    app = FeatureSelectionApp()
    app.run_app() 