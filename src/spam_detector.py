from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np
from typing import List, Dict, Tuple
import logging
from .config import SPAM_MODEL_NAME, SPAM_THRESHOLD, DEVICE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreTrainedSpamDetector:
    def __init__(self, model_name: str = SPAM_MODEL_NAME):
        logger.info(f"Loading spam detection model: {model_name}")
        
        try:
            # Try to load the toxic comment classifier
            self.classifier = pipeline(
                "text-classification",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            self.model_name = model_name
            logger.info("✅ Spam detection model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load {model_name}, trying backup model: {e}")
            
            # Backup model
            try:
                self.classifier = pipeline(
                    "text-classification",
                    model="martin-ha/toxic-comment-model",
                    device=0 if torch.cuda.is_available() else -1,
                    return_all_scores=True
                )
                self.model_name = "martin-ha/toxic-comment-model"
                logger.info("✅ Backup spam detection model loaded")
                
            except Exception as backup_error:
                logger.error(f"Failed to load backup model: {backup_error}")
                raise Exception("Could not load any spam detection model")
    
    def predict(self, text: str) -> Tuple[bool, float]:
        """
        Predict if text is spam/toxic
        Returns: (is_spam: bool, spam_probability: float)
        """
        if not text or len(text.strip()) < 3:
            return False, 0.0
        
        try:
            # Clean text
            cleaned_text = text.strip()[:512]  # Limit length for model
            
            # Get prediction
            results = self.classifier(cleaned_text)
            
            # Handle different model outputs
            spam_probability = 0.0
            
            if isinstance(results[0], list):  # Multiple scores returned
                for result in results[0]:
                    label = result['label'].upper()
                    score = result['score']
                    
                    # Look for toxic/spam indicators
                    if any(keyword in label for keyword in ['TOXIC', 'SPAM', 'OBSCENE', 'THREAT', 'INSULT']):
                        spam_probability = max(spam_probability, score)
            
            else:  # Single result
                result = results[0]
                label = result['label'].upper()
                score = result['score']
                
                if any(keyword in label for keyword in ['TOXIC', 'SPAM', 'OBSCENE', 'THREAT', 'INSULT']):
                    spam_probability = score
            
            # Additional heuristic checks for obvious spam patterns
            spam_indicators = [
                'subscribe to my channel',
                'check out my',
                'visit my website',
                'click the link',
                'dm me',
                'follow for follow',
                'sub4sub',
                '🔥🔥🔥',
                'amazing offer',
                'limited time'
            ]
            
            text_lower = text.lower()
            heuristic_spam_score = sum(0.2 for indicator in spam_indicators if indicator in text_lower)
            
            # Combine model prediction with heuristics
            final_spam_probability = min(1.0, spam_probability + heuristic_spam_score)
            is_spam = final_spam_probability > SPAM_THRESHOLD
            
            return is_spam, final_spam_probability
            
        except Exception as e:
            logger.error(f"Error in spam prediction for text '{text[:50]}...': {e}")
            return False, 0.0
    
    def filter_comments(self, comments: List[Dict], threshold: float = SPAM_THRESHOLD) -> List[Dict]:
        """Filter spam comments from a list"""
        if not comments:
            return []
        
        clean_comments = []
        spam_count = 0
        
        logger.info(f"Filtering spam from {len(comments)} comments...")
        
        for comment in comments:
            try:
                is_spam, spam_prob = self.predict(comment['text'])
                
                if not is_spam or spam_prob < threshold:
                    # Add spam probability to comment for reference
                    comment_copy = comment.copy()
                    comment_copy['spam_probability'] = spam_prob
                    comment_copy['is_spam'] = is_spam
                    clean_comments.append(comment_copy)
                else:
                    spam_count += 1
                    
            except Exception as e:
                logger.warning(f"Error processing comment: {e}")
                # Include comment if error occurs (err on side of inclusion)
                clean_comments.append(comment)
        
        logger.info(f"Filtered out {spam_count} spam comments, {len(clean_comments)} remain")
        return clean_comments
    
    def get_spam_statistics(self, comments: List[Dict]) -> Dict:
        """Get detailed spam statistics"""
        if not comments:
            return {'total': 0, 'spam': 0, 'clean': 0, 'spam_ratio': 0.0}
        
        spam_count = 0
        spam_probabilities = []
        
        for comment in comments:
            is_spam, spam_prob = self.predict(comment['text'])
            spam_probabilities.append(spam_prob)
            if is_spam:
                spam_count += 1
        
        return {
            'total': len(comments),
            'spam': spam_count,
            'clean': len(comments) - spam_count,
            'spam_ratio': spam_count / len(comments),
            'avg_spam_probability': np.mean(spam_probabilities),
            'max_spam_probability': np.max(spam_probabilities),
            'min_spam_probability': np.min(spam_probabilities)
        }

# Test function
def test_spam_detector():
    """Test the spam detector"""
    detector = PreTrainedSpamDetector()
    
    test_comments = [
        {"text": "This is a great video! Thanks for sharing."},
        {"text": "SUBSCRIBE TO MY CHANNEL FOR AMAZING CONTENT!!!"},
        {"text": "Check out my website for free money!"},
        {"text": "I really enjoyed this tutorial."},
        {"text": "First comment! Please like and subscribe!"},
    ]
    
    print("Testing spam detection:")
    for comment in test_comments:
        is_spam, prob = detector.predict(comment['text'])
        status = "SPAM" if is_spam else "CLEAN"
        print(f"{status} ({prob:.3f}): {comment['text'][:50]}...")

if __name__ == "__main__":
    test_spam_detector()
