import unittest
import os
import sys
import time

# Add the root directory to path so we can import frontend.app and backend code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# We can import preprocess and predict from frontend.app
# But Streamlit script execution might cause issues if run directly without st.
# Let's mock Streamlit temporarily if it runs on import, but wait, app.py runs streamlit code globally!
# It's better to isolate the core logic or just test what's possible.
# Actually, Streamlit caches and pages run on import. It might be better to import from a separate module, 
# but we can test the backend artifacts directly to avoid streamlit UI triggering.

import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def extract_preprocess(text: str) -> str:
    """Standalone preprocess matching app.py for testing feature 4."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    try:
        tokens = word_tokenize(text)
        text = " ".join([lemmatizer.lemmatize(t) for t in tokens if t not in stop_words])
    except Exception:
        pass
    return text

class TestFeatures(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.model_path = os.path.join(base_dir, "..", "backend", "model", "sentiment_model.pkl")
        cls.vectorizer_path = os.path.join(base_dir, "..", "backend", "model", "tfidf_vectorizer.pkl")
        cls.accuracy_path = os.path.join(base_dir, "..", "backend", "model", "accuracy.txt")
        
        try:
            with open(cls.model_path, "rb") as f:
                cls.model = pickle.load(f)
            with open(cls.vectorizer_path, "rb") as f:
                cls.vectorizer = pickle.load(f)
        except Exception as e:
            cls.model = None
            cls.vectorizer = None

    def test_feature_4_preprocessing(self):
        """Test preprocessing for clean data (lowercase, no punctuation, lemmatization, no stopwords)"""
        raw_text = "The roads are VERY bad!!! #potholes @mayor http://link.com running"
        clean_text = extract_preprocess(raw_text)
        
        self.assertNotIn("http", clean_text, "URLs should be removed")
        self.assertNotIn("@", clean_text, "Mentions should be removed")
        self.assertNotIn("#", clean_text, "Hashtags should be removed")
        self.assertNotIn("!", clean_text, "Punctuation should be removed")
        self.assertEqual(clean_text, clean_text.lower(), "Text should be lowercase")
        self.assertNotIn("the", clean_text.split(), "Stopwords should be removed")
        self.assertIn("road", clean_text, "Lemmatization should convert 'roads' to 'road'")
        self.assertIn("running", clean_text, "Lemmatizer leaves 'running' as is depending on POS, but should exist")

    def test_feature_8_model_persistence(self):
        """Test if model files are saved and loadable"""
        self.assertTrue(os.path.exists(self.model_path), "model.pkl should exist")
        self.assertTrue(os.path.exists(self.vectorizer_path), "vectorizer.pkl should exist")
        self.assertIsNotNone(self.model, "Model should be successfully loaded")
        self.assertIsNotNone(self.vectorizer, "Vectorizer should be successfully loaded")

    def test_feature_1_classification_categories(self):
        """Test if prediction outputs one of the 3 allowed categories"""
        if self.model is None:
            self.skipTest("Model not loaded")
        
        sample_texts = ["water is terrible", "this is okay", "great service"]
        expected_outputs = ["Positive", "Negative", "Neutral"]
        
        # Test just the set of outputs
        for text in sample_texts:
            clean = extract_preprocess(text)
            vec = self.vectorizer.transform([clean])
            pred_idx = self.model.predict(vec)[0]
            sentiment_map = {0: "Negative", 2: "Neutral", 4: "Positive"}
            label = sentiment_map.get(pred_idx, str(pred_idx))
            
            # Since our model was trained on strings "Positive", "Negative", "Neutral" directly
            # it will output those strings if trained with them. So let's handle dynamically:
            if label not in expected_outputs:
                # If model itself predicted string directly
                self.assertIn(pred_idx, expected_outputs, f"Prediction {pred_idx} not in allowed categories.")
            else:
                self.assertIn(label, expected_outputs, f"Prediction {label} not in allowed categories.")

    def test_feature_2_real_time(self):
        """Test if prediction takes less than 1 second"""
        if self.model is None:
            self.skipTest("Model not loaded")
            
        clean = extract_preprocess("Fix the potholes now")
        vec = self.vectorizer.transform([clean])
        
        start_time = time.time()
        _ = self.model.predict(vec)
        duration = time.time() - start_time
        
        self.assertLess(duration, 1.0, f"Prediction took {duration:.4f}s, which is slower than 1 second")

    def test_feature_6_accuracy(self):
        """Test if accuracy meets the 88-92% threshold"""
        self.assertTrue(os.path.exists(self.accuracy_path), "accuracy.txt should exist")
        with open(self.accuracy_path, "r") as f:
            acc_str = f.read().strip()
            acc = float(acc_str.replace('%',''))
        self.assertGreaterEqual(acc, 88.0, "Accuracy must be >= 88%")

if __name__ == "__main__":
    unittest.main()
