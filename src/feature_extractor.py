import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict
from collections import Counter
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        # Regex patterns for text analysis
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "]+", flags=re.UNICODE
        )
        
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        
        logger.info("✅ Feature Extractor initialized")
    
    def calculate_comment_velocity(self, comments: List[Dict], window_minutes: int = 60) -> float:
        """Calculate comments per minute within a time window"""
        if len(comments) < 2:
            return 0.0
        
        try:
            timestamps = []
            for comment in comments:
                if 'published_at' in comment and comment['published_at']:
                    # Parse ISO timestamp
                    timestamp_str = comment['published_at']
                    if 'Z' in timestamp_str:
                        timestamp_str = timestamp_str.replace('Z', '+00:00')
                    elif '+' not in timestamp_str:
                        timestamp_str += '+00:00'
                    
                    timestamp = datetime.fromisoformat(timestamp_str)
                    timestamps.append(timestamp)
            
            if len(timestamps) < 2:
                return 0.0
            
            # Sort timestamps
            timestamps.sort()
            
            # Calculate time span in minutes
            time_span_minutes = (timestamps[-1] - timestamps[0]).total_seconds() / 60
            
            # Avoid division by zero
            if time_span_minutes <= 0:
                return len(comments)  # All comments at same time
            
            # Calculate velocity
            velocity = len(comments) / time_span_minutes
            
            # Cap at reasonable maximum (e.g., 100 comments per minute)
            return min(velocity, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating velocity: {e}")
            return 0.0
    
    def calculate_engagement_metrics(self, comments: List[Dict]) -> Dict:
        """Calculate engagement-related features"""
        if not comments:
            return {
                'total_likes': 0,
                'total_replies': 0,
                'avg_likes_per_comment': 0.0,
                'avg_replies_per_comment': 0.0,
                'engagement_ratio': 0.0,
                'high_engagement_comments': 0
            }
        
        total_likes = sum(comment.get('like_count', 0) for comment in comments)
        total_replies = sum(comment.get('reply_count', 0) for comment in comments)
        
        # Count high-engagement comments (likes > 5 or replies > 2)
        high_engagement = sum(1 for c in comments 
                            if c.get('like_count', 0) > 5 or c.get('reply_count', 0) > 2)
        
        return {
            'total_likes': total_likes,
            'total_replies': total_replies,
            'avg_likes_per_comment': total_likes / len(comments),
            'avg_replies_per_comment': total_replies / len(comments),
            'engagement_ratio': (total_likes + total_replies * 2) / len(comments),
            'high_engagement_comments': high_engagement,
            'high_engagement_ratio': high_engagement / len(comments)
        }
    
    def calculate_text_features(self, comments: List[Dict]) -> Dict:
        """Calculate text-based features"""
        if not comments:
            return {
                'avg_comment_length': 0.0,
                'avg_word_count': 0.0,
                'emoji_ratio': 0.0,
                'caps_ratio': 0.0,
                'url_ratio': 0.0,
                'mention_ratio': 0.0,
                'question_ratio': 0.0,
                'exclamation_ratio': 0.0
            }
        
        lengths = []
        word_counts = []
        emoji_counts = 0
        caps_counts = 0
        total_chars = 0
        url_count = 0
        mention_count = 0
        question_count = 0
        exclamation_count = 0
        
        for comment in comments:
            text = comment.get('text', '')
            if not text:
                continue
                
            lengths.append(len(text))
            word_counts.append(len(text.split()))
            
            # Count various text features
            emoji_matches = self.emoji_pattern.findall(text)
            emoji_counts += len(emoji_matches)
            
            caps_counts += sum(1 for c in text if c.isupper())
            total_chars += len(text)
            
            url_count += len(self.url_pattern.findall(text))
            mention_count += len(self.mention_pattern.findall(text))
            
            if '?' in text:
                question_count += 1
            if '!' in text:
                exclamation_count += 1
        
        total_comments = len(comments)
        
        return {
            'avg_comment_length': np.mean(lengths) if lengths else 0,
            'avg_word_count': np.mean(word_counts) if word_counts else 0,
            'emoji_ratio': emoji_counts / total_comments,
            'caps_ratio': caps_counts / max(total_chars, 1),
            'url_ratio': url_count / total_comments,
            'mention_ratio': mention_count / total_comments,
            'question_ratio': question_count / total_comments,
            'exclamation_ratio': exclamation_count / total_comments
        }
    
    def calculate_author_diversity(self, comments: List[Dict]) -> Dict:
        """Calculate author diversity metrics"""
        if not comments:
            return {
                'unique_authors': 0,
                'author_diversity_ratio': 0.0,
                'repeat_commenters': 0,
                'top_commenter_dominance': 0.0
            }
        
        authors = [comment.get('author', 'Anonymous') for comment in comments]
        author_counts = Counter(authors)
        
        unique_authors = len(author_counts)
        repeat_commenters = sum(1 for count in author_counts.values() if count > 1)
        
        # Calculate how much the top commenter dominates
        if author_counts:
            top_commenter_count = max(author_counts.values())
            top_commenter_dominance = top_commenter_count / len(comments)
        else:
            top_commenter_dominance = 0.0
        
        return {
            'unique_authors': unique_authors,
            'author_diversity_ratio': unique_authors / len(comments),
            'repeat_commenters': repeat_commenters,
            'repeat_commenter_ratio': repeat_commenters / unique_authors if unique_authors > 0 else 0,
            'top_commenter_dominance': top_commenter_dominance
        }
    
    def calculate_temporal_features(self, comments: List[Dict]) -> Dict:
        """Calculate time-based features"""
        if not comments:
            return {
                'comment_spread_hours': 0.0,
                'peak_activity_hour': 0,
                'activity_consistency': 0.0
            }
        
        try:
            timestamps = []
            for comment in comments:
                if 'published_at' in comment and comment['published_at']:
                    timestamp_str = comment['published_at']
                    if 'Z' in timestamp_str:
                        timestamp_str = timestamp_str.replace('Z', '+00:00')
                    elif '+' not in timestamp_str:
                        timestamp_str += '+00:00'
                    
                    timestamp = datetime.fromisoformat(timestamp_str)
                    timestamps.append(timestamp)
            
            if len(timestamps) < 2:
                return {
                    'comment_spread_hours': 0.0,
                    'peak_activity_hour': 0,
                    'activity_consistency': 0.0
                }
            
            timestamps.sort()
            
            # Time spread
            time_spread = (timestamps[-1] - timestamps[0]).total_seconds() / 3600  # hours
            
            # Peak activity hour
            hours = [ts.hour for ts in timestamps]
            hour_counts = Counter(hours)
            peak_hour = hour_counts.most_common(1)[0][0] if hour_counts else 0
            
            # Activity consistency (inverse of standard deviation of intervals)
            if len(timestamps) > 2:
                intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                           for i in range(1, len(timestamps))]
                consistency = 1 / (1 + np.std(intervals) / 60)  # normalized
            else:
                consistency = 1.0
            
            return {
                'comment_spread_hours': time_spread,
                'peak_activity_hour': peak_hour,
                'activity_consistency': min(1.0, consistency)
            }
            
        except Exception as e:
            logger.error(f"Error calculating temporal features: {e}")
            return {
                'comment_spread_hours': 0.0,
                'peak_activity_hour': 0,
                'activity_consistency': 0.0
            }
    
    def extract_all_features(self, comments: List[Dict], video_metadata: Dict = None) -> Dict:
        """Extract all features for viral prediction"""
        if not comments:
            return self._get_empty_features()
        
        logger.info(f"Extracting features from {len(comments)} comments")
        
        # Basic metrics
        features = {
            'comment_count': len(comments),
        }
        
        # Velocity features (different time windows)
        features['comment_velocity_1min'] = self.calculate_comment_velocity(comments, 1)
        features['comment_velocity_5min'] = self.calculate_comment_velocity(comments, 5)
        features['comment_velocity_10min'] = self.calculate_comment_velocity(comments, 10)
        features['comment_velocity_60min'] = self.calculate_comment_velocity(comments, 60)
        
        # Engagement features
        engagement_features = self.calculate_engagement_metrics(comments)
        features.update(engagement_features)
        
        # Text features
        text_features = self.calculate_text_features(comments)
        features.update(text_features)
        
        # Author diversity features
        author_features = self.calculate_author_diversity(comments)
        features.update(author_features)
        
        # Temporal features
        temporal_features = self.calculate_temporal_features(comments)
        features.update(temporal_features)
        
        # Sentiment features (if sentiment analysis has been done)
        if comments and 'sentiment' in comments[0]:
            sentiments = [c['sentiment'] for c in comments if 'sentiment' in c]
            if sentiments:
                features.update({
                    'avg_sentiment': np.mean(sentiments),
                    'sentiment_std': np.std(sentiments) if len(sentiments) > 1 else 0,
                    'positive_ratio': sum(1 for s in sentiments if s > 0.1) / len(sentiments),
                    'negative_ratio': sum(1 for s in sentiments if s < -0.1) / len(sentiments),
                    'neutral_ratio': sum(1 for s in sentiments if -0.1 <= s <= 0.1) / len(sentiments),
                    'sentiment_range': max(sentiments) - min(sentiments) if len(sentiments) > 1 else 0
                })
            else:
                features.update(self._get_empty_sentiment_features())
        else:
            features.update(self._get_empty_sentiment_features())
        
        # Video metadata features
        if video_metadata:
            features.update({
                'minutes_since_upload': video_metadata.get('minutes_since_upload', 0),
                'video_view_count': video_metadata.get('view_count', 0),
                'video_like_count': video_metadata.get('like_count', 0),
                'video_comment_count': video_metadata.get('comment_count', 0),
                'video_has_description': 1 if video_metadata.get('description') else 0,
                'video_has_tags': len(video_metadata.get('tags', [])),
            })
        else:
            features.update({
                'minutes_since_upload': 0,
                'video_view_count': 0,
                'video_like_count': 0,
                'video_comment_count': 0,
                'video_has_description': 0,
                'video_has_tags': 0,
            })
        
        # Derived features
        features['comments_per_view'] = (features['comment_count'] / 
                                       max(features['video_view_count'], 1)) * 1000
        features['likes_per_comment'] = (features['video_like_count'] / 
                                       max(features['comment_count'], 1))
        
        logger.info(f"Extracted {len(features)} features")
        return features
    
    def _get_empty_features(self) -> Dict:
        """Return empty feature set"""
        features = {}
        
        # Basic features
        basic_features = [
            'comment_count', 'comment_velocity_1min', 'comment_velocity_5min',
            'comment_velocity_10min', 'comment_velocity_60min'
        ]
        
        # Engagement features
        engagement_features = [
            'total_likes', 'total_replies', 'avg_likes_per_comment',
            'avg_replies_per_comment', 'engagement_ratio', 'high_engagement_comments',
            'high_engagement_ratio'
        ]
        
        # Text features
        text_features = [
            'avg_comment_length', 'avg_word_count', 'emoji_ratio', 'caps_ratio',
            'url_ratio', 'mention_ratio', 'question_ratio', 'exclamation_ratio'
        ]
        
        # Author features
        author_features = [
            'unique_authors', 'author_diversity_ratio', 'repeat_commenters',
            'repeat_commenter_ratio', 'top_commenter_dominance'
        ]
        
        # Temporal features
        temporal_features = [
            'comment_spread_hours', 'peak_activity_hour', 'activity_consistency'
        ]
        
        # Video metadata features
        video_features = [
            'minutes_since_upload', 'video_view_count', 'video_like_count',
            'video_comment_count', 'video_has_description', 'video_has_tags'
        ]
        
        # Derived features
        derived_features = ['comments_per_view', 'likes_per_comment']
        
        all_feature_names = (basic_features + engagement_features + text_features + 
                           author_features + temporal_features + video_features + 
                           derived_features)
        
        for feature_name in all_feature_names:
            features[feature_name] = 0.0
        
        # Add sentiment features
        features.update(self._get_empty_sentiment_features())
        
        return features
    
    def _get_empty_sentiment_features(self) -> Dict:
        """Return empty sentiment features"""
        return {
            'avg_sentiment': 0.0,
            'sentiment_std': 0.0,
            'positive_ratio': 0.0,
            'negative_ratio': 0.0,
            'neutral_ratio': 0.0,
            'sentiment_range': 0.0
        }

# Test function
def test_feature_extractor():
    """Test the feature extractor"""
    extractor = FeatureExtractor()
    
    # Create test comments
    test_comments = [
        {
            'text': 'This is amazing! 😍 Thanks for sharing!',
            'author': 'User1',
            'published_at': '2025-09-23T10:00:00Z',
            'like_count': 5,
            'reply_count': 2,
            'sentiment': 0.8
        },
        {
            'text': 'Not sure about this... seems okay I guess.',
            'author': 'User2', 
            'published_at': '2025-09-23T10:05:00Z',
            'like_count': 1,
            'reply_count': 0,
            'sentiment': -0.1
        },
        {
            'text': 'Great tutorial! When will you make the next one?',
            'author': 'User3',
            'published_at': '2025-09-23T10:10:00Z',
            'like_count': 3,
            'reply_count': 1,
            'sentiment': 0.6
        }
    ]
    
    video_metadata = {
        'minutes_since_upload': 30,
        'view_count': 1000,
        'like_count': 50,
        'comment_count': 25,
        'description': 'Test video description',
        'tags': ['tutorial', 'test']
    }
    
    print("Testing feature extraction:")
    features = extractor.extract_all_features(test_comments, video_metadata)
    
    print(f"Total features extracted: {len(features)}")
    print("\nKey features:")
    for key, value in list(features.items())[:10]:
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_feature_extractor()
