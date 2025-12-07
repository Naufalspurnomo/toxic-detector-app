"""
Model Training Module for SVM-based Toxic Comment Detection

Features:
- TF-IDF Vectorization
- SVM Training (Linear & RBF kernels)
- Cross-validation
- Model Evaluation (Accuracy, Precision, Recall, F1)
- Model Persistence
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, List
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODEL_CONFIG, 
    DATASET_PATH, 
    SVM_MODEL_PATH, 
    TFIDF_VECTORIZER_PATH,
    SLANG_DICT_PATH,
    MODELS_DIR
)
from modules.preprocessing import TextPreprocessor


class ModelTrainer:
    """
    Train and evaluate SVM model for toxic comment detection.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or MODEL_CONFIG
        self.preprocessor = TextPreprocessor(SLANG_DICT_PATH)
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.get('tfidf_max_features', 5000),
            ngram_range=self.config.get('tfidf_ngram_range', (1, 2)),
            min_df=2,
            max_df=0.95
        )
        
        # Initialize SVM model
        self.model = SVC(
            kernel=self.config.get('svm_kernel', 'linear'),
            C=self.config.get('svm_C', 1.0),
            probability=True,
            random_state=self.config.get('random_state', 42)
        )
        
        # Training results
        self.training_results = {}
        self.is_trained = False
    
    def load_dataset(self, dataset_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and validate dataset.
        
        Args:
            dataset_path: Path to CSV dataset
            
        Returns:
            DataFrame with 'text' and 'label' columns
        """
        path = dataset_path or DATASET_PATH
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        
        df = pd.read_csv(path)
        
        # Validate required columns
        required_cols = ['text', 'label']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Dataset missing required column: {col}")
        
        # Clean dataset
        df = df.dropna(subset=['text', 'label'])
        df['label'] = df['label'].astype(int)
        
        print(f"Loaded dataset: {len(df)} samples")
        print(f"  - Aman (0): {len(df[df['label'] == 0])} samples")
        print(f"  - Toxic (1): {len(df[df['label'] == 1])} samples")
        
        return df
    
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing to all texts in dataset.
        
        Args:
            df: DataFrame with 'text' column
            
        Returns:
            DataFrame with added 'processed_text' column
        """
        print("Preprocessing texts...")
        df['processed_text'] = df['text'].apply(self.preprocessor.preprocess)
        
        # Remove empty processed texts
        df = df[df['processed_text'].str.strip() != '']
        
        print(f"After preprocessing: {len(df)} valid samples")
        return df
    
    def train(
        self, 
        dataset_path: Optional[str] = None,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Full training pipeline.
        
        Args:
            dataset_path: Path to CSV dataset
            save_model: Whether to save trained model
            
        Returns:
            Dictionary containing training results and metrics
        """
        # Load and preprocess data
        df = self.load_dataset(dataset_path)
        df = self.preprocess_dataset(df)
        
        X = df['processed_text'].values
        y = df['label'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.get('test_size', 0.2),
            random_state=self.config.get('random_state', 42),
            stratify=y
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Testing set: {len(X_test)} samples")
        
        # TF-IDF Vectorization
        print("\nApplying TF-IDF vectorization...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"Feature dimensions: {X_train_tfidf.shape[1]} features")
        
        # Train SVM
        print(f"\nTraining SVM (kernel={self.config.get('svm_kernel')})...")
        self.model.fit(X_train_tfidf, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_tfidf)
        y_pred_proba = self.model.predict_proba(X_test_tfidf)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=['Aman', 'Toxic'], output_dict=True)
        
        # Cross-validation
        print("\nPerforming 5-fold cross-validation...")
        X_all_tfidf = self.vectorizer.transform(X)
        cv_scores = cross_val_score(self.model, X_all_tfidf, y, cv=5)
        
        # Store results
        self.training_results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'dataset': {
                'total_samples': len(df),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'class_distribution': {
                    'aman': int(sum(y == 0)),
                    'toxic': int(sum(y == 1))
                }
            },
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'cross_validation': {
                'scores': cv_scores.tolist(),
                'mean': float(cv_scores.mean()),
                'std': float(cv_scores.std())
            },
            'feature_count': X_train_tfidf.shape[1]
        }
        
        self.is_trained = True
        
        # Print results
        self._print_results()
        
        # Save model
        if save_model:
            self.save_model()
        
        return self.training_results
    
    def _print_results(self):
        """Print training results summary."""
        print("\n" + "="*50)
        print("TRAINING RESULTS")
        print("="*50)
        
        metrics = self.training_results['metrics']
        print(f"\n📊 Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        cv = self.training_results['cross_validation']
        print(f"\n🔄 Cross-Validation (5-fold):")
        print(f"  Mean: {cv['mean']:.4f} (+/- {cv['std']*2:.4f})")
        
        cm = self.training_results['confusion_matrix']
        print(f"\n📋 Confusion Matrix:")
        print(f"  {'':>10} Pred:Aman  Pred:Toxic")
        print(f"  {'True:Aman':>10}  {cm[0][0]:>8}  {cm[0][1]:>10}")
        print(f"  {'True:Toxic':>10}  {cm[1][0]:>8}  {cm[1][1]:>10}")
        
        print("\n" + "="*50)
    
    def save_model(self, model_path: Optional[str] = None, vectorizer_path: Optional[str] = None):
        """
        Save trained model and vectorizer.
        
        Args:
            model_path: Path to save SVM model
            vectorizer_path: Path to save TF-IDF vectorizer
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Ensure directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        model_path = model_path or SVM_MODEL_PATH
        vectorizer_path = vectorizer_path or TFIDF_VECTORIZER_PATH
        
        # Save model
        joblib.dump(self.model, model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Save vectorizer
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"✅ Vectorizer saved: {vectorizer_path}")
        
        # Save training results
        results_path = os.path.join(MODELS_DIR, 'training_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved: {results_path}")
    
    def load_model(self, model_path: Optional[str] = None, vectorizer_path: Optional[str] = None) -> bool:
        """
        Load trained model and vectorizer.
        
        Args:
            model_path: Path to SVM model
            vectorizer_path: Path to TF-IDF vectorizer
            
        Returns:
            True if loaded successfully
        """
        model_path = model_path or SVM_MODEL_PATH
        vectorizer_path = vectorizer_path or TFIDF_VECTORIZER_PATH
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            return False
        
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.is_trained = True
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform grid search for optimal hyperparameters.
        
        Args:
            X: Feature matrix (TF-IDF)
            y: Labels
            
        Returns:
            Best parameters found
        """
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        
        grid_search = GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }


def train_model(dataset_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to train model with default settings.
    
    Args:
        dataset_path: Path to dataset CSV
        
    Returns:
        Training results dictionary
    """
    trainer = ModelTrainer()
    return trainer.train(dataset_path)


if __name__ == "__main__":
    # Train model when run directly
    results = train_model()
    print(f"\nFinal Accuracy: {results['metrics']['accuracy']*100:.2f}%")
