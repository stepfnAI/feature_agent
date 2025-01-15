from sfn_blueprint import MODEL_CONFIG
DEFAULT_LLM_PROVIDER = 'openai'
DEFAULT_LLM_MODEL = 'gpt-4o-mini'

MODEL_CONFIG["field_mapper"] = {
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 300,
        "n": 1,
        "stop": None
    }
}

MODEL_CONFIG["method_suggester"] = {
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 1000,
        "n": 1,
        "stop": None
    }
}

MODEL_CONFIG["feature_recommender"] = {
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 2000,
        "n": 1,
        "stop": None
    }
} 


MODEL_CONFIG["category_identifier"] = {
    "openai": {
        "model": "gpt-4o-mini", #"gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 100,
        "n": 1,
        "stop": None
    }
}

MODEL_CONFIG["feature_suggester"] = {
    "openai": {
        "model": "gpt-4o-mini", #"gpt-3.5-turbo",
        "temperature": 0.3,
        "max_tokens": 2000,
        "n": 1,
        "stop": None
    }
}
