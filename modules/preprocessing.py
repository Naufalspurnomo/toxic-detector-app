"""
Text Preprocessing Module for Indonesian Gaming Comments

Pipeline:
1. Cleaning (emoji, URL, username removal)
2. Case Folding (lowercase)
3. Slang Normalization (using dictionary)
4. Stopword Removal (Sastrawi)
5. Stemming (Sastrawi)
"""

import re
import json
import os
from typing import List, Optional, Dict

# Try to import NLP libraries
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    print("Warning: Sastrawi not installed. Using basic preprocessing.")

try:
    import nltk
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class TextPreprocessor:
    """
    Indonesian text preprocessing for toxic comment detection.
    """
    
    def __init__(self, slang_dict_path: Optional[str] = None):
        """
        Initialize preprocessor with optional slang dictionary.
        
        Args:
            slang_dict_path: Path to JSON file containing slang mappings
        """
        # Load slang dictionary
        self.slang_dict = {}
        if slang_dict_path and os.path.exists(slang_dict_path):
            with open(slang_dict_path, 'r', encoding='utf-8') as f:
                self.slang_dict = json.load(f)
        
        # Initialize Sastrawi components if available
        self.stemmer = None
        self.stopword_remover = None
        
        if SASTRAWI_AVAILABLE:
            try:
                stemmer_factory = StemmerFactory()
                self.stemmer = stemmer_factory.createStemmer()
                
                stopword_factory = StopWordRemoverFactory()
                self.stopword_remover = stopword_factory.createStopWordRemover()
            except Exception as e:
                print(f"Warning: Could not initialize Sastrawi: {e}")
        
        # Download NLTK data if available
        if NLTK_AVAILABLE:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
            except:
                pass
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance."""
        # URL pattern
        self.url_pattern = re.compile(
            r'https?://\S+|www\.\S+|bit\.ly/\S+',
            re.IGNORECASE
        )
        
        # Mention pattern (@username)
        self.mention_pattern = re.compile(r'@\w+')
        
        # Hashtag pattern
        self.hashtag_pattern = re.compile(r'#\w+')
        
        # Emoji pattern (Unicode emoji ranges)
        self.emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"  # dingbats
            u"\U000024C2-\U0001F251"  # enclosed characters
            u"\U0001F900-\U0001F9FF"  # supplemental symbols
            u"\U0001FA00-\U0001FA6F"  # chess symbols
            u"\U0001FA70-\U0001FAFF"  # symbols extended
            u"\U00002600-\U000026FF"  # misc symbols
            "]+",
            re.UNICODE
        )
        
        # Special characters and punctuation (keep alphanumeric and spaces)
        self.special_char_pattern = re.compile(r'[^a-zA-Z0-9\s]')
        
        # Multiple whitespace
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Repeated characters (more than 2)
        self.repeated_char_pattern = re.compile(r'(.)\1{2,}')
        
        # Numbers
        self.number_pattern = re.compile(r'\d+')
    
    def clean_text(self, text: str) -> str:
        """
        Step 1: Clean text by removing URLs, mentions, emojis, etc.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove mentions (@username)
        text = self.mention_pattern.sub(' ', text)
        
        # Remove hashtags
        text = self.hashtag_pattern.sub(' ', text)
        
        # Remove emojis
        text = self.emoji_pattern.sub(' ', text)
        
        # Remove special characters and punctuation
        text = self.special_char_pattern.sub(' ', text)
        
        # Reduce repeated characters (e.g., "woooow" -> "wow")
        text = self.repeated_char_pattern.sub(r'\1\1', text)
        
        # Remove extra whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        return text.strip()
    
    def case_folding(self, text: str) -> str:
        """
        Step 2: Convert text to lowercase.
        
        Args:
            text: Input text
            
        Returns:
            Lowercase text
        """
        return text.lower() if text else ""
    
    def normalize_slang(self, text: str) -> str:
        """
        Step 3: Normalize slang/informal words to standard form.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized slang words
        """
        if not text or not self.slang_dict:
            return text
        
        words = text.split()
        normalized_words = []
        
        for word in words:
            # Check if word exists in slang dictionary
            normalized = self.slang_dict.get(word.lower(), word)
            normalized_words.append(normalized)
        
        return ' '.join(normalized_words)
    
    def remove_stopwords(self, text: str) -> str:
        """
        Step 4: Remove Indonesian stopwords.
        
        Args:
            text: Input text
            
        Returns:
            Text without stopwords
        """
        if not text:
            return ""
        
        if self.stopword_remover:
            return self.stopword_remover.remove(text)
        
        # Fallback: Basic Indonesian stopwords
        basic_stopwords = {
            'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 
            'untuk', 'pada', 'adalah', 'sebagai', 'dalam', 'tidak',
            'akan', 'juga', 'atau', 'ada', 'mereka', 'sudah', 'saya',
            'kamu', 'kami', 'kita', 'ia', 'dia', 'apa', 'siapa', 'mana',
            'kapan', 'kenapa', 'mengapa', 'bagaimana', 'bila', 'kalau',
            'jika', 'ketika', 'saat', 'waktu', 'sambil', 'seperti',
            'agar', 'supaya', 'bahwa', 'karena', 'oleh', 'tentang',
            'antara', 'setelah', 'sebelum', 'sementara', 'sedang',
            'hanya', 'saja', 'pun', 'lah', 'kah', 'tah', 'kan'
        }
        
        words = text.split()
        filtered_words = [w for w in words if w.lower() not in basic_stopwords]
        return ' '.join(filtered_words)
    
    def stem_text(self, text: str) -> str:
        """
        Step 5: Stem words to their root form.
        
        Args:
            text: Input text
            
        Returns:
            Stemmed text
        """
        if not text:
            return ""
        
        if self.stemmer:
            return self.stemmer.stem(text)
        
        # Fallback: No stemming if Sastrawi not available
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text)
            except:
                pass
        
        # Fallback: Simple split
        return text.split()
    
    def preprocess(self, text: str, verbose: bool = False) -> str:
        """
        Full preprocessing pipeline.
        
        Args:
            text: Raw input text
            verbose: If True, return dict with each step's output
            
        Returns:
            Fully preprocessed text (or dict if verbose)
        """
        if verbose:
            steps = {}
            steps['original'] = text
            
            # Step 1: Clean
            text = self.clean_text(text)
            steps['after_cleaning'] = text
            
            # Step 2: Case folding
            text = self.case_folding(text)
            steps['after_case_folding'] = text
            
            # Step 3: Slang normalization
            text = self.normalize_slang(text)
            steps['after_slang_normalization'] = text
            
            # Step 4: Stopword removal
            text = self.remove_stopwords(text)
            steps['after_stopword_removal'] = text
            
            # Step 5: Stemming
            text = self.stem_text(text)
            steps['after_stemming'] = text
            
            steps['final'] = text
            return steps
        
        # Regular pipeline
        text = self.clean_text(text)
        text = self.case_folding(text)
        text = self.normalize_slang(text)
        text = self.remove_stopwords(text)
        text = self.stem_text(text)
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess multiple texts.
        
        Args:
            texts: List of raw texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


# Convenience function
def create_preprocessor(slang_dict_path: Optional[str] = None) -> TextPreprocessor:
    """Create and return a TextPreprocessor instance."""
    return TextPreprocessor(slang_dict_path)
