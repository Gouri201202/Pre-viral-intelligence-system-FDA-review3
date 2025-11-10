from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np
from typing import List, Dict
import logging
from .config import SENTIMENT_MODEL_NAME, DEVICE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreTrainedSentimentAnalyzer:
    def __init__(self, model_name: str = SENTIMENT_MODEL_NAME):
        logger.info(f"Loading sentiment analysis model: {model_name}")
        
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            self.model_name = model_name
            
            # Test the model to understand its labels
            test_result = self.sentiment_pipeline("This is a test.")
            self.labels = [item['label'] for item in test_result[0]]
            
            logger.info(f"✅ Sentiment model loaded. Labels: {self.labels}")
            
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise Exception("Could not load sentiment analysis model")
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text
        Returns: float between -1 (very negative) and 1 (very positive)
        """
        if not text or len(text.strip()) < 2:
            return 0.0
        
        try:
            # Clean and truncate text
            cleaned_text = text.strip()[:512]
            
            # Get sentiment predictions
            results = self.sentiment_pipeline(cleaned_text)[0]  # First result
            
            # Convert to sentiment score based on model labels
            sentiment_score = 0.0
            
            # Handle different label formats
            for result in results:
                label = result['label'].upper()
                score = result['score']
                
                if 'POSITIVE' in label:
                    sentiment_score += score
                elif 'NEGATIVE' in label:
                    sentiment_score -= score
                elif 'NEUTRAL' in label:
                    # Neutral contributes nothing to sentiment score
                    pass
                elif 'LABEL_2' in label:  # RoBERTa format: LABEL_2 = Positive
                    sentiment_score += score
                elif 'LABEL_0' in label:  # RoBERTa format: LABEL_0 = Negative
                    sentiment_score -= score
                # LABEL_1 is usually neutral
            
            # Ensure score is in [-1, 1] range
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            return sentiment_score
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for '{text[:50]}...': {e}")
            return 0.0
    
    def analyze_comment_batch(self, comments: List[Dict]) -> List[Dict]:
        """Analyze sentiment for a batch of comments"""
        if not comments:
            return []
        
        logger.info(f"Analyzing sentiment for {len(comments)} comments...")
        
        results = []
        for comment in comments:
            try:
                sentiment_score = self.analyze_sentiment(comment['text'])
                comment_with_sentiment = comment.copy()
                comment_with_sentiment['sentiment'] = sentiment_score
                comment_with_sentiment['sentiment_label'] = self._score_to_label(sentiment_score)
                results.append(comment_with_sentiment)
                
            except Exception as e:
                logger.warning(f"Error processing comment sentiment: {e}")
                # Add comment with neutral sentiment if error occurs
                comment_copy = comment.copy()
                comment_copy['sentiment'] = 0.0
                comment_copy['sentiment_label'] = 'neutral'
                results.append(comment_copy)
        
        logger.info("Sentiment analysis completed")
        return results
    
    def calculate_sentiment_trajectory(self, comments: List[Dict]) -> Dict:
        """Calculate sentiment trends over time"""
        if not comments:
            return {
                'trend': 'no_data',
                'average': 0.0,
                'volatility': 0.0,
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0}
            }
        
        # Ensure comments have sentiment scores
        if 'sentiment' not in comments[0]:
            comments = self.analyze_comment_batch(comments)
        
        sentiments = [comment['sentiment'] for comment in comments]
        
        # Calculate basic statistics
        avg_sentiment = np.mean(sentiments)
        sentiment_std = np.std(sentiments) if len(sentiments) > 1 else 0.0
        
        # Determine trend (comparing first half vs second half)
        trend = 'stable'
        if len(sentiments) >= 6:
            mid_point = len(sentiments) // 2
            first_half_avg = np.mean(sentiments[:mid_point])
            second_half_avg = np.mean(sentiments[mid_point:])
            
            difference = second_half_avg - first_half_avg
            
            if difference > 0.2:
                trend = 'improving'
            elif difference < -0.2:
                trend = 'declining'
            else:
                trend = 'stable'
        elif len(sentiments) >= 2:
            # For smaller samples, compare beginning vs end
            if sentiments[-1] > sentiments[0] + 0.3:
                trend = 'improving'
            elif sentiments[-1] < sentiments[0] - 0.3:
                trend = 'declining'
        
        # Calculate distribution
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        total = len(sentiments)
        distribution = {
            'positive': positive_count / total,
            'neutral': neutral_count / total,
            'negative': negative_count / total
        }
        
        return {
            'trend': trend,
            'average': float(avg_sentiment),
            'volatility': float(sentiment_std),
            'sentiment_distribution': distribution,
            'total_comments': total,
            'sentiment_evolution': sentiments[-10:] if len(sentiments) > 10 else sentiments
        }
    
    def _score_to_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.3:
            return 'positive'
        elif score < -0.3:
            return 'negative'
        else:
            return 'neutral'
    
    def get_sentiment_insights(self, comments: List[Dict]) -> Dict:
        """Get detailed sentiment insights"""
        if not comments:
            return {}
        
        trajectory = self.calculate_sentiment_trajectory(comments)
        
        # Find most positive and negative comments
        comments_with_sentiment = self.analyze_comment_batch(comments)
        
        most_positive = max(comments_with_sentiment, key=lambda x: x['sentiment'])
        most_negative = min(comments_with_sentiment, key=lambda x: x['sentiment'])
        
        return {
            **trajectory,
            'most_positive_comment': {
                'text': most_positive['text'][:200] + '...',
                'sentiment': most_positive['sentiment'],
                'author': most_positive.get('author', 'Anonymous')
            },
            'most_negative_comment': {
                'text': most_negative['text'][:200] + '...',
                'sentiment': most_negative['sentiment'],
                'author': most_negative.get('author', 'Anonymous')
            }
        }

# Test function
def test_sentiment_analyzer():
    """Test the sentiment analyzer"""
    analyzer = PreTrainedSentimentAnalyzer()
    
    test_comments = [
        {"text": "This video is absolutely amazing! I love it!", "author": "User1"},
        {"text": "This is terrible and boring content.", "author": "User2"},
        {"text": "Not bad, could be better though.", "author": "User3"},
        {"text": "Okay video, nothing special.", "author": "User4"},
        {"text": "Best tutorial ever! Thank you so much!", "author": "User5"},
    ]
    
    print("Testing sentiment analysis:")
    for comment in test_comments:
        sentiment = analyzer.analyze_sentiment(comment['text'])
        label = analyzer._score_to_label(sentiment)
        print(f"{label.upper()} ({sentiment:.3f}): {comment['text']}")
    
    print("\nTesting sentiment trajectory:")
    trajectory = analyzer.calculate_sentiment_trajectory(test_comments)
    print(f"Average sentiment: {trajectory['average']:.3f}")
    print(f"Trend: {trajectory['trend']}")
    print(f"Distribution: {trajectory['sentiment_distribution']}")

if __name__ == "__main__":
    test_sentiment_analyzer()
