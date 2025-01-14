# Feature Selection Advisor

An AI-powered feature selection tool that intelligently analyzes your dataset and recommends optimal features through multiple selection methods, with interactive selection capabilities and comprehensive feature analysis.

## 🌟 Features

- **Intelligent Feature Analysis**: Automatically analyzes and classifies features
- **Smart Method Selection**: AI-powered selection of appropriate feature selection methods
- **Multi-Method Approach**: Combines multiple feature selection techniques:
  - Variance-based methods
  - Correlation analysis
  - Statistical tests
  - Model-based importance
  - Missing value analysis
  - Cardinality checks
- **Interactive Selection**: Review and modify AI-suggested feature selections
- **Flexible Data Input**: Supports multiple file formats (CSV, Excel, JSON, Parquet)
- **Visual Progress Tracking**: Clear feedback on selection process
- **Data Export**: Download processed dataset with selected features

## 🚀 Getting Started

### Prerequisites

- Python 3.8-3.11
- OpenAI API key

### Installation

1. Clone the repository:

```bash
git clone git@github.com:stepfnAI/feature_agent.git
cd feature_agent
```

2. Create and activate a virtual environment using virtualenv:

```bash
pip install virtualenv                # Install virtualenv if not already installed
virtualenv venv                       # Create virtual environment
source venv/bin/activate             # Linux/Mac
# OR
.\venv\Scripts\activate              # Windows
```

3. Install the package in editable mode:

```bash
pip install -e .
```

4. Set up your OpenAI API key:

```bash
export OPENAI_API_KEY='your_openai_api_key'
```

### Running the Application

```bash
streamlit run .\examples\app.py
```

## 🔄 Workflow

1. **Data Loading and Preview**
   - Upload your dataset (CSV, Excel, JSON, or Parquet)
   - Preview the loaded data
   - Reset functionality available at any point

2. **Field Mapping and Classification**
   - AI identifies critical fields (CUST_ID, BILLING_DATE, REVENUE, TARGET)
   - Automatic feature classification and metadata generation
   - Interactive review and modification of mappings

3. **Feature Selection Methods**
   - AI suggests appropriate selection methods based on data characteristics
   - Methods available for both with and without target variable
   - Interactive method selection with AI recommendations

4. **Feature Selection**
   - Comprehensive feature analysis using selected methods
   - AI-powered recommendations with explanations
   - Interactive feature selection interface
   - Visual indicators for feature importance

5. **Post Processing**
   - Three operation options:
     - View selected features
     - Download processed dataset (CSV format)
     - Finish and reset application

## 🛠️ Architecture

The application follows a modular architecture with these key components:

- **SFNFieldMappingAgent**: Identifies critical fields
- **FeatureClassificationAgent**: Analyzes and classifies features
- **SFNMethodSuggesterAgent**: Suggests appropriate selection methods
- **SFNFeatureSelectionExecutorAgent**: Executes selection methods
- **SFNFeatureRecommenderAgent**: Provides final recommendations
- **StreamlitView**: Manages user interface
- **SFNSessionManager**: Handles application state

## 🔒 Security

- Secure API key handling
- Input validation
- Safe data processing
- Environment variable management

## 📊 Selection Methods

### Methods Without Target
- Variance threshold
- Quasi-constant removal
- Correlation analysis (Pearson, Spearman, Kendall)
- Missing value analysis
- Cardinality checks

### Methods With Target
- Chi-square test
- ANOVA F-test
- Mutual information
- Information value
- Target correlation analysis
- Model-based importance (Random Forest, XGBoost)

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

Email: puneet@stepfunction.ai