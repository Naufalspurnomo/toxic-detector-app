"""
Configuration settings for Toxic Comment Detection System
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Data files
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")
SLANG_DICT_PATH = os.path.join(DATA_DIR, "slang_dict.json")

# Model files
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

# Model parameters
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "svm_kernel": "linear",  # 'linear' or 'rbf'
    "svm_C": 1.0,
    "tfidf_max_features": 5000,
    "tfidf_ngram_range": (1, 2),
}

# UI Configuration
UI_CONFIG = {
    "page_title": "🛡️ Toxic Comment Detector",
    "page_icon": "🛡️",
    "layout": "wide",
    "theme_primary_color": "#6366F1",
    "safe_color": "#10B981",
    "toxic_color": "#EF4444",
}

# Labels
LABELS = {
    0: {"name": "Aman", "emoji": "✅", "color": "#10B981"},
    1: {"name": "Toxic", "emoji": "⚠️", "color": "#EF4444"},
}
