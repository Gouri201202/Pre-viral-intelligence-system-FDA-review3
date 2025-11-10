from transformers import pipeline
import torch
import json
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from .config import LLM_MODEL_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMViralPredictor:
    def __init__(self, model_name: str = LLM_MODEL_NAME):
        logger.info(f"Loading LLM for viral prediction: {model_name}")
        
        try:
            # Try to load the specified model
            self.generator = pipeline(
                "text-generation",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                pad_token_id=50256  # GPT-2 pad token
            )
            
            self.model_name = model_name
            logger.info("✅ LLM viral predictor loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load {model_name}, trying backup model: {e}")
            
            # Try backup models
            backup_models = [
                "gpt2",
                "distilgpt2"
            ]
            
            for backup_model in backup_models:
                try:
                    self.generator = pipeline(
                        "text-generation",
                        model=backup_model,
                        device=0 if torch.cuda.is_available() else -1,
                        truncation=True,
                        pad_token_id=50256
                    )
                    self.model_name = backup_model
                    logger.info(f"✅ Loaded backup model: {backup_model}")
                    break
                    
                except Exception as backup_error:
                    logger.warning(f"Failed to load {backup_model}: {backup_error}")
                    continue
            else:
                raise Exception("Could not load any LLM model for viral prediction")
    
    def predict_viral_probability(self, features: Dict, comments: List[Dict], 
                                video_metadata: Dict = None) -> Tuple[float, Dict]:
        """Use LLM to predict viral probability based on features and comments"""
        
        try:
            # Prepare comprehensive context for LLM analysis
            context = self._prepare_viral_analysis_prompt(features, comments, video_metadata)
            
            # Generate LLM analysis
            response = self.generator(
                context,
                max_length=len(context.split()) + 100,  # Allow for response
                temperature=0.3,  # Lower temperature for more consistent results
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            # Extract response text
            generated_text = response[0]['generated_text']
            llm_response = generated_text[len(context):].strip()
            
            # Extract viral probability from LLM response
            viral_probability = self._extract_probability_from_response(llm_response)
            
            # Combine LLM prediction with heuristic analysis for robustness
            heuristic_probability = self._calculate_heuristic_probability(features, comments)
            
            # Weighted combination (70% LLM, 30% heuristic)
            final_probability = 0.7 * viral_probability + 0.3 * heuristic_probability
            final_probability = max(0.0, min(1.0, final_probability))
            
            # Determine prediction class and confidence
            prediction_class = 'viral' if final_probability > 0.6 else 'moderate' if final_probability > 0.3 else 'low'
            confidence = abs(final_probability - 0.5) * 2  # Distance from uncertain (0.5)
            
            result = {
                'viral_probability': final_probability,
                'prediction_class': prediction_class,
                'confidence': min(confidence, 1.0),
                'llm_probability': viral_probability,
                'heuristic_probability': heuristic_probability,
                'llm_reasoning': llm_response[:500],  # Truncate for storage
                'method': 'llm_heuristic_hybrid',
                'model_used': self.model_name
            }
            
            logger.info(f"Viral prediction: {final_probability:.3f} ({prediction_class})")
            return final_probability, result
            
        except Exception as e:
            logger.error(f"Error in LLM viral prediction: {e}")
            
            # Fallback to heuristic-only prediction
            heuristic_probability = self._calculate_heuristic_probability(features, comments)
            
            return heuristic_probability, {
                'viral_probability': heuristic_probability,
                'prediction_class': 'heuristic_fallback',
                'confidence': 0.6,
                'error': str(e),
                'method': 'heuristic_only'
            }
    
    def _prepare_viral_analysis_prompt(self, features: Dict, comments: List[Dict], 
                                     video_metadata: Dict = None) -> str:
        """Prepare comprehensive prompt for viral analysis"""
        
        # Extract key metrics
        comment_count = features.get('comment_count', 0)
        velocity = features.get('comment_velocity_5min', 0)
        engagement = features.get('engagement_ratio', 0)
        sentiment = features.get('avg_sentiment', 0)
        author_diversity = features.get('author_diversity_ratio', 0)
        minutes_since_upload = features.get('minutes_since_upload', 0)
        
        # Sample recent comments for context
        recent_comments = []
        if comments:
            sample_size = min(5, len(comments))
            for comment in comments[-sample_size:]:
                text = comment.get('text', '')[:100]  # Truncate for prompt
                if text:
                    recent_comments.append(f"- {text}")
        
        # Create video context
        video_context = ""
        if video_metadata:
            video_context = f"""
VIDEO INFO:
- Title: {video_metadata.get('title', 'Unknown')[:100]}...
- Views: {video_metadata.get('view_count', 0):,}
- Likes: {video_metadata.get('like_count', 0):,}
- Channel: {video_metadata.get('channel_name', 'Unknown')}"""
        
        prompt = f"""Analyze YouTube video viral potential:

{video_context}

ENGAGEMENT METRICS ({minutes_since_upload} minutes after upload):
- Comments: {comment_count} (velocity: {velocity:.2f}/5min)
- Engagement score: {engagement:.3f}
- Sentiment average: {sentiment:.3f} (-1 to 1)
- Author diversity: {author_diversity:.3f}

RECENT COMMENTS:
{chr(10).join(recent_comments[:5]) if recent_comments else "No comments available"}

VIRAL INDICATORS TO CONSIDER:
1. Early engagement velocity (>2 comments/5min = high viral potential)
2. Positive sentiment trajectory (>0.3 = very positive response)
3. High author diversity (>0.8 = organic reach)
4. Comment quality and enthusiasm level
5. Time since upload vs engagement achieved

Based on viral video patterns, this content shows"""
        
        return prompt
    
    def _extract_probability_from_response(self, response_text: str) -> float:
        """Extract probability from LLM response using multiple methods"""
        
        response_lower = response_text.lower()
        
        # Method 1: Look for explicit percentages
        percentage_patterns = [
            r'(\d{1,2}(?:\.\d+)?)\s*%',
            r'(\d{1,2}(?:\.\d+)?)\s*percent',
            r'probability.*?(\d{1,2}(?:\.\d+)?)\s*%'
        ]
        
        for pattern in percentage_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                try:
                    probability = float(matches[-1]) / 100  # Use last match
                    return max(0.0, min(1.0, probability))
                except ValueError:
                    continue
        
        # Method 2: Look for probability values
        prob_patterns = [
            r'probability[:\s]*(\d\.\d+)',
            r'score[:\s]*(\d\.\d+)',
            r'(\d\.\d+)\s*(?:probability|chance|likelihood)'
        ]
        
        for pattern in prob_patterns:
            matches = re.findall(pattern, response_lower)
            if matches:
                try:
                    probability = float(matches[-1])
                    if probability > 1:  # Convert from percentage
                        probability /= 100
                    return max(0.0, min(1.0, probability))
                except ValueError:
                    continue
        
        # Method 3: Keyword-based scoring
        keyword_scores = {
            # High viral potential
            'extremely high': 0.9, 'very high': 0.8, 'high viral': 0.8, 'strong viral': 0.8,
            'excellent': 0.85, 'outstanding': 0.9, 'exceptional': 0.9,
            
            # Moderate-high viral potential  
            'high': 0.7, 'good': 0.65, 'promising': 0.6, 'positive': 0.6,
            'likely': 0.7, 'probable': 0.7,
            
            # Moderate viral potential
            'moderate': 0.5, 'medium': 0.5, 'average': 0.45, 'okay': 0.4,
            'possible': 0.4, 'maybe': 0.4,
            
            # Low viral potential
            'low': 0.3, 'poor': 0.2, 'weak': 0.25, 'unlikely': 0.2,
            'minimal': 0.15, 'very low': 0.1
        }
        
        max_score = 0.5  # Default neutral
        for keyword, score in keyword_scores.items():
            if keyword in response_lower:
                max_score = max(max_score, score)
        
        return max_score
    
    def _calculate_heuristic_probability(self, features: Dict, comments: List[Dict]) -> float:
        """Calculate viral probability using heuristic rules as fallback"""
        
        score = 0.0
        
        # Comment velocity scoring (0-0.3)
        velocity = features.get('comment_velocity_5min', 0)
        if velocity > 10:
            score += 0.3
        elif velocity > 5:
            score += 0.25
        elif velocity > 2:
            score += 0.2
        elif velocity > 0.5:
            score += 0.1
        
        # Engagement scoring (0-0.25)
        engagement = features.get('engagement_ratio', 0)
        if engagement > 5:
            score += 0.25
        elif engagement > 2:
            score += 0.2
        elif engagement > 1:
            score += 0.15
        elif engagement > 0.5:
            score += 0.1
        
        # Sentiment scoring (0-0.2)
        sentiment = features.get('avg_sentiment', 0)
        if sentiment > 0.5:
            score += 0.2
        elif sentiment > 0.3:
            score += 0.15
        elif sentiment > 0.1:
            score += 0.1
        elif sentiment < -0.3:
            score -= 0.1  # Negative sentiment reduces score
        
        # Author diversity scoring (0-0.15)
        diversity = features.get('author_diversity_ratio', 0)
        if diversity > 0.9:
            score += 0.15
        elif diversity > 0.8:
            score += 0.12
        elif diversity > 0.7:
            score += 0.08
        elif diversity > 0.5:
            score += 0.05
        
        # Time factor (0-0.1)
        minutes_since_upload = features.get('minutes_since_upload', 0)
        if minutes_since_upload < 60:  # Fresh content with engagement is promising
            time_factor = max(0, 0.1 - (minutes_since_upload / 600))  # Decay over 10 hours
            score += time_factor
        
        return max(0.0, min(1.0, score))
    
    def get_viral_insights(self, features: Dict, comments: List[Dict]) -> Dict:
        """Get detailed insights about viral potential"""
        
        _, prediction_result = self.predict_viral_probability(features, comments)
        
        # Analyze key factors
        insights = {
            'overall_prediction': prediction_result,
            'key_strengths': [],
            'key_weaknesses': [],
            'recommendations': []
        }
        
        # Analyze strengths
        if features.get('comment_velocity_5min', 0) > 2:
            insights['key_strengths'].append("High comment velocity indicates strong engagement")
        
        if features.get('avg_sentiment', 0) > 0.3:
            insights['key_strengths'].append("Positive audience sentiment")
        
        if features.get('author_diversity_ratio', 0) > 0.8:
            insights['key_strengths'].append("High author diversity suggests organic reach")
        
        # Analyze weaknesses
        if features.get('comment_count', 0) < 10:
            insights['key_weaknesses'].append("Low comment count - needs more engagement")
        
        if features.get('avg_sentiment', 0) < -0.2:
            insights['key_weaknesses'].append("Negative sentiment trend")
        
        if features.get('engagement_ratio', 0) < 0.5:
            insights['key_weaknesses'].append("Low engagement ratio")
        
        # Generate recommendations
        if prediction_result['viral_probability'] > 0.6:
            insights['recommendations'].extend([
                "High viral potential detected - consider cross-promotion",
                "Monitor closely for next 2-4 hours",
                "Engage with commenters to boost momentum"
            ])
        elif prediction_result['viral_probability'] > 0.3:
            insights['recommendations'].extend([
                "Moderate potential - boost engagement through responses",
                "Consider sharing in relevant communities",
                "Address any negative feedback quickly"
            ])
        else:
            insights['recommendations'].extend([
                "Low viral potential - focus on content optimization",
                "Analyze successful content for patterns",
                "Consider different posting times or promotion strategies"
            ])
        
        return insights

# Test function
def test_viral_predictor():
    """Test the viral predictor"""
    predictor = LLMViralPredictor()
    
    # Create test features
    test_features = {
        'comment_count': 25,
        'comment_velocity_5min': 3.2,
        'engagement_ratio': 2.5,
        'avg_sentiment': 0.6,
        'author_diversity_ratio': 0.85,
        'minutes_since_upload': 45
    }
    
    test_comments = [
        {'text': 'This is amazing! Love it!', 'author': 'User1'},
        {'text': 'Great tutorial, very helpful', 'author': 'User2'},
        {'text': 'Thanks for sharing this!', 'author': 'User3'}
    ]
    
    print("Testing viral prediction:")
    probability, result = predictor.predict_viral_probability(test_features, test_comments)
    
    print(f"Viral Probability: {probability:.3f}")
    print(f"Prediction Class: {result['prediction_class']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Method: {result['method']}")

if __name__ == "__main__":
    test_viral_predictor()
