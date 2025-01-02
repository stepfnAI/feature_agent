from typing import Dict, List, Union
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sfn_blueprint import SFNAgent
from sfn_blueprint import Task

class SFNFeatureSelectionExecutorAgent(SFNAgent):
    """Agent responsible for executing feature selection methods and computing scores"""
    
    def __init__(self):
        super().__init__(name="Feature Selection Executor", role="Method Executor")
        self.critical_fields = ["customer_id", "date_mapping", "revenue_mapping", "target"]
        
    def execute_task(self, task: Task) -> Dict:
        """
        Execute feature selection methods on the dataset
        
        :param task: Task object containing:
            - data: DataFrame
            - params: {
                'methods': List of method dictionaries from method suggester,
                'field_mappings': Dict from field mapping agent,
                'metadata': Dict of feature metadata
            }
        :return: Dictionary with test results for each feature
        """
        if not isinstance(task.data, dict) or not isinstance(task.data.get('dataframe'), pd.DataFrame):
            raise ValueError("Task data must be a dictionary containing a pandas DataFrame under 'dataframe' key")
            
        df = task.data.get('dataframe').copy()
        methods = task.data.get('methods', [])
        field_mappings = task.data.get('field_mappings', {})
        metadata = task.data.get('metadata', {})
        
        # Remove critical fields from feature selection
        features_to_analyze = [col for col in df.columns if col not in field_mappings.values()]
        
        results = {feature: {} for feature in features_to_analyze}
        
        for method in methods:
            method_name = method['method_name']
            method_scores = self._execute_method(
                method_name=method_name,
                df=df,
                features=features_to_analyze,
                target_column=field_mappings.get('target'),
                metadata=metadata
            )
            
            # Add scores to results
            for feature in features_to_analyze:
                results[feature][method_name] = method_scores.get(feature)
        
        return results

    def _execute_method(self, method_name: str, df: pd.DataFrame, 
                       features: List[str], target_column: str = None,
                       metadata: Dict = None) -> Dict[str, float]:
        """Execute a specific feature selection method"""
        
        method_functions = {
            # Variance based methods
            'variance_threshold': self._variance_threshold,
            'quasi_constant_removal': self._quasi_constant_removal,
            
            # Correlation based methods
            'pearson_correlation': self._pearson_correlation,
            'spearman_correlation': self._spearman_correlation,
            'cramer_v': self._cramers_v,
            
            # Missing value based methods
            'missing_ratio_threshold': self._missing_ratio,
            
            # Cardinality based methods
            'unique_value_ratio': self._unique_value_ratio,
            'low_variance_categories': self._category_variance,
            
            # Statistical tests
            'chi_square_test': self._chi_square_test,
            'mutual_information': self._mutual_information,
            
            # Target correlation methods
            'pearson_with_target': self._pearson_with_target,
            'spearman_with_target': self._spearman_with_target,
            
            # Model based methods
            'random_forest_importance': self._random_forest_importance
        }
        
        if method_name not in method_functions:
            raise ValueError(f"Method {method_name} not implemented")
            
        return method_functions[method_name](df, features, target_column, metadata)

    def _variance_threshold(self, df: pd.DataFrame, features: List[str], 
                          target_column: str = None, metadata: Dict = None) -> Dict[str, float]:
        """Calculate variance for numerical features"""
        results = {}
        for feature in features:
            if pd.api.types.is_numeric_dtype(df[feature]):
                results[feature] = float(df[feature].var())
            else:
                results[feature] = None
        return results

    def _missing_ratio(self, df: pd.DataFrame, features: List[str], 
                      target_column: str = None, metadata: Dict = None) -> Dict[str, float]:
        """Calculate missing value ratio"""
        return {feature: df[feature].isnull().mean() for feature in features}

    def _unique_value_ratio(self, df: pd.DataFrame, features: List[str], 
                          target_column: str = None, metadata: Dict = None) -> Dict[str, float]:
        """Calculate unique value ratio"""
        return {feature: df[feature].nunique() / len(df) for feature in features}

    def _pearson_correlation(self, df: pd.DataFrame, features: List[str], 
                           target_column: str = None, metadata: Dict = None) -> Dict[str, float]:
        """Calculate maximum absolute Pearson correlation with other features"""
        results = {}
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
        
        if len(numeric_features) < 2:
            return {feature: None for feature in features}
            
        corr_matrix = df[numeric_features].corr(method='pearson').abs()
        
        for feature in features:
            if feature in numeric_features:
                # Get maximum correlation excluding self-correlation
                feature_corrs = corr_matrix[feature].drop(feature)
                results[feature] = float(feature_corrs.max())
            else:
                results[feature] = None
        return results

    def _mutual_information(self, df: pd.DataFrame, features: List[str], 
                          target_column: str, metadata: Dict = None) -> Dict[str, float]:
        """Calculate mutual information with target"""
        if not target_column or target_column not in df.columns:
            return {feature: None for feature in features}
            
        results = {}
        target = df[target_column]
        
        # Handle categorical target
        if not pd.api.types.is_numeric_dtype(target):
            le = LabelEncoder()
            target = le.fit_transform(target)
            
        for feature in features:
            feature_data = df[feature]
            if not pd.api.types.is_numeric_dtype(feature_data):
                le = LabelEncoder()
                feature_data = le.fit_transform(feature_data)
                
            mi_score = mutual_info_regression(
                feature_data.values.reshape(-1, 1),
                target
            )[0]
            results[feature] = float(mi_score)
            
        return results

    def _random_forest_importance(self, df: pd.DataFrame, features: List[str], 
                                target_column: str, metadata: Dict = None) -> Dict[str, float]:
        """Calculate feature importance using Random Forest"""
        if not target_column or target_column not in df.columns:
            return {feature: None for feature in features}
            
        results = {}
        target = df[target_column]
        
        # Prepare feature matrix
        X = pd.get_dummies(df[features])
        feature_names = X.columns
        
        # Choose model based on target type
        if pd.api.types.is_numeric_dtype(target):
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            target = LabelEncoder().fit_transform(target)
            
        # Fit model and get feature importances
        model.fit(X, target)
        importances = model.feature_importances_
        
        # Map importance scores back to original features
        for feature in features:
            if feature in feature_names:
                results[feature] = float(importances[feature_names.get_loc(feature)])
            else:
                # For one-hot encoded features, sum their importances
                encoded_cols = [col for col in feature_names if col.startswith(f"{feature}_")]
                if encoded_cols:
                    importance_sum = sum(importances[feature_names.get_loc(col)] for col in encoded_cols)
                    results[feature] = float(importance_sum)
                else:
                    results[feature] = None
                    
        return results

    def _quasi_constant_removal(self, df: pd.DataFrame, features: List[str], 
                              target_column: str = None, metadata: Dict = None) -> Dict[str, float]:
        """Calculate ratio of most frequent value for each feature"""
        results = {}
        for feature in features:
            value_counts = df[feature].value_counts(normalize=True)
            if len(value_counts) > 0:
                results[feature] = float(value_counts.iloc[0])  # Ratio of most frequent value
            else:
                results[feature] = None
        return results

    def _spearman_correlation(self, df: pd.DataFrame, features: List[str], 
                            target_column: str = None, metadata: Dict = None) -> Dict[str, float]:
        """Calculate maximum absolute Spearman correlation with other features"""
        results = {}
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
        
        if len(numeric_features) < 2:
            return {feature: None for feature in features}
            
        corr_matrix = df[numeric_features].corr(method='spearman').abs()
        
        for feature in features:
            if feature in numeric_features:
                feature_corrs = corr_matrix[feature].drop(feature)
                results[feature] = float(feature_corrs.max())
            else:
                results[feature] = None
        return results

    def _cramers_v(self, df: pd.DataFrame, features: List[str], 
                   target_column: str = None, metadata: Dict = None) -> Dict[str, float]:
        """Calculate maximum Cramer's V correlation with other categorical features"""
        def cramers_v_stat(confusion_matrix):
            chi2 = stats.chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            min_dim = min(confusion_matrix.shape) - 1
            return np.sqrt(chi2 / (n * min_dim)) if min_dim != 0 else 0

        results = {}
        categorical_features = [f for f in features if not pd.api.types.is_numeric_dtype(df[f])]
        
        if len(categorical_features) < 2:
            return {feature: None for feature in features}
            
        for feature in features:
            if feature in categorical_features:
                max_correlation = 0
                for other_feature in categorical_features:
                    if feature != other_feature:
                        confusion_matrix = pd.crosstab(df[feature], df[other_feature])
                        correlation = cramers_v_stat(confusion_matrix)
                        max_correlation = max(max_correlation, correlation)
                results[feature] = float(max_correlation)
            else:
                results[feature] = None
        return results

    def _category_variance(self, df: pd.DataFrame, features: List[str], 
                         target_column: str = None, metadata: Dict = None) -> Dict[str, float]:
        """Calculate category distribution variance for categorical features"""
        results = {}
        for feature in features:
            if not pd.api.types.is_numeric_dtype(df[feature]):
                value_counts = df[feature].value_counts(normalize=True)
                results[feature] = float(value_counts.var()) if len(value_counts) > 1 else 0.0
            else:
                results[feature] = None
        return results

    def _chi_square_test(self, df: pd.DataFrame, features: List[str], 
                        target_column: str, metadata: Dict = None) -> Dict[str, float]:
        """Calculate chi-square test p-values for categorical features vs target"""
        if not target_column or target_column not in df.columns:
            return {feature: None for feature in features}
            
        results = {}
        for feature in features:
            if not pd.api.types.is_numeric_dtype(df[feature]):
                contingency = pd.crosstab(df[feature], df[target_column])
                chi2, p_value = stats.chi2_contingency(contingency)[:2]
                results[feature] = float(p_value)
            else:
                results[feature] = None
        return results

    def _pearson_with_target(self, df: pd.DataFrame, features: List[str], 
                            target_column: str, metadata: Dict = None) -> Dict[str, float]:
        """Calculate Pearson correlation with target for numerical features"""
        if not target_column or target_column not in df.columns or not pd.api.types.is_numeric_dtype(df[target_column]):
            return {feature: None for feature in features}
            
        results = {}
        for feature in features:
            if pd.api.types.is_numeric_dtype(df[feature]):
                correlation = df[feature].corr(df[target_column], method='pearson')
                results[feature] = float(abs(correlation))
            else:
                results[feature] = None
        return results

    def _spearman_with_target(self, df: pd.DataFrame, features: List[str], 
                             target_column: str, metadata: Dict = None) -> Dict[str, float]:
        """Calculate Spearman correlation with target"""
        if not target_column or target_column not in df.columns:
            return {feature: None for feature in features}
            
        results = {}
        for feature in features:
            if pd.api.types.is_numeric_dtype(df[feature]):
                correlation = df[feature].corr(df[target_column], method='spearman')
                results[feature] = float(abs(correlation))
            else:
                results[feature] = None
        return results
 