"""
Prediction Module for Toxic Comment Detection

Provides interface for making predictions on new text inputs.
"""

import os
import joblib
from typing import Dict, Optional, Tuple, List, Any

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SVM_MODEL_PATH, TFIDF_VECTORIZER_PATH, SLANG_DICT_PATH, LABELS
from modules.preprocessing import TextPreprocessor


class ToxicityPredictor:
    """
    Predict toxicity of text comments using trained SVM model.
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        vectorizer_path: Optional[str] = None,
        slang_dict_path: Optional[str] = None
    ):
        """
        Initialize predictor with model and vectorizer.
        
        Args:
            model_path: Path to trained SVM model
            vectorizer_path: Path to fitted TF-IDF vectorizer
            slang_dict_path: Path to slang dictionary
        """
        self.model_path = model_path or SVM_MODEL_PATH
        self.vectorizer_path = vectorizer_path or TFIDF_VECTORIZER_PATH
        self.slang_dict_path = slang_dict_path or SLANG_DICT_PATH
        
        self.model = None
        self.vectorizer = None
        self.preprocessor = None
        self.is_loaded = False
        
        # Try to load model on initialization
        self._load_components()
    
    def _load_components(self) -> bool:
        """Load model, vectorizer, and preprocessor."""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.preprocessor = TextPreprocessor(self.slang_dict_path)
                self.is_loaded = True
                return True
            else:
                print("Warning: Model files not found. Please train the model first.")
                return False
        except Exception as e:
            print(f"Error loading components: {e}")
            return False
    
    def reload(self) -> bool:
        """Reload model components."""
        return self._load_components()
    
    def predict(self, text: str, return_details: bool = False) -> Dict[str, Any]:
        """
        Predict toxicity of a single text.
        
        Args:
            text: Input text to analyze
            return_details: If True, include preprocessing details
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded:
            if not self._load_components():
                return {
                    'error': 'Model not loaded. Please train the model first.',
                    'label': None,
                    'confidence': 0.0
                }
        
        # Preprocess text
        if return_details:
            preprocessing_steps = self.preprocessor.preprocess(text, verbose=True)
            processed_text = preprocessing_steps['final']
        else:
            processed_text = self.preprocessor.preprocess(text)
        
        # Handle empty processed text
        if not processed_text.strip():
            return {
                'original_text': text,
                'processed_text': processed_text,
                'label': 0,
                'label_name': LABELS[0]['name'],
                'emoji': LABELS[0]['emoji'],
                'color': LABELS[0]['color'],
                'confidence': 1.0,
                'probabilities': {'aman': 1.0, 'toxic': 0.0},
                'message': 'Tidak ada konten yang dapat dianalisis',
                'preprocessing_steps': preprocessing_steps if return_details else None
            }
        
        # Vectorize
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(text_tfidf)[0]
        probabilities = self.model.predict_proba(text_tfidf)[0]
        
        # Get label info
        label_info = LABELS[prediction]
        confidence = float(probabilities[prediction])
        
        result = {
            'original_text': text,
            'processed_text': processed_text,
            'label': int(prediction),
            'label_name': label_info['name'],
            'emoji': label_info['emoji'],
            'color': label_info['color'],
            'confidence': confidence,
            'probabilities': {
                'aman': float(probabilities[0]),
                'toxic': float(probabilities[1])
            }
        }
        
        # Add preprocessing details if requested
        if return_details:
            result['preprocessing_steps'] = preprocessing_steps
        
        # Add human-readable message
        if prediction == 1:
            if confidence > 0.8:
                result['message'] = "⚠️ Teks ini sangat mungkin mengandung konten toxic"
            else:
                result['message'] = "⚠️ Teks ini kemungkinan mengandung konten toxic"
        else:
            if confidence > 0.8:
                result['message'] = "✅ Teks ini aman dan tidak mengandung konten toxic"
            else:
                result['message'] = "✅ Teks ini kemungkinan aman"
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict toxicity for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction results
        """
        return [self.predict(text) for text in texts]
    
    def get_toxicity_score(self, text: str) -> float:
        """
        Get toxicity score (0-1) for text.
        
        Args:
            text: Input text
            
        Returns:
            Toxicity probability (0 = safe, 1 = toxic)
        """
        result = self.predict(text)
        return result['probabilities'].get('toxic', 0.0)
    
    def is_toxic(self, text: str, threshold: float = 0.5) -> bool:
        """
        Check if text is toxic based on threshold.
        
        Args:
            text: Input text
            threshold: Toxicity threshold (default 0.5)
            
        Returns:
            True if toxic, False otherwise
        """
        return self.get_toxicity_score(text) >= threshold


def create_predictor() -> ToxicityPredictor:
    """Create and return a ToxicityPredictor instance."""
    return ToxicityPredictor()


# Quick prediction function
def predict_toxicity(text: str) -> Dict[str, Any]:
    """
    Quick function to predict toxicity of text.
    
    Args:
        text: Input text
        
    Returns:
        Prediction result dictionary
    """
    predictor = ToxicityPredictor()
    return predictor.predict(text)


if __name__ == "__main__":
    # Test predictor
    predictor = ToxicityPredictor()
    
    test_texts = [
        "Main bareng yuk!",
        "dasar nubs bego banget sih lu",
        "GG WP mantap gamenya",
        "anjir tolol bgt sih",
    ]
    
    print("\n🔍 Testing Toxicity Predictor\n")
    for text in test_texts:
        result = predictor.predict(text)
        print(f"Text: {text}")
        print(f"  → {result['emoji']} {result['label_name']} ({result['confidence']*100:.1f}%)")
        print()
