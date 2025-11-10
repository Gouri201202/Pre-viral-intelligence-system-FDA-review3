import json
import random
from typing import List, Dict, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIAdvisorySystem:
    def __init__(self):
        # Pre-defined advice templates for different scenarios
        self.advice_templates = {
            'high_viral': {
                'urgent': [
                    "🔥 High viral potential detected! Enable monetization immediately if not already done.",
                    "🚀 Capitalize on momentum - share across all your social platforms now!",
                    "📊 Monitor closely - prepare for potential traffic surge in next 2 hours.",
                    "💬 Engage with top comments immediately to boost algorithmic visibility."
                ],
                'strategic': [
                    "Consider creating follow-up content while momentum is high",
                    "Pin your best engaging comment to encourage more interaction",
                    "Prepare community posts to maintain engagement",
                    "Cross-post to relevant subreddits and communities"
                ]
            },
            'moderate_viral': {
                'boost': [
                    "💡 Good potential detected - boost engagement by responding to commenters",
                    "📈 Share in 2-3 relevant Facebook groups or communities",
                    "🎯 Target high-engagement comments for replies",
                    "⏰ Post community update about the video's topic"
                ],
                'optimize': [
                    "Consider updating thumbnail if CTR is low",
                    "Add cards/end screens to related content",
                    "Reply to questions to encourage more comments",
                    "Share behind-the-scenes content in community tab"
                ]
            },
            'low_viral': {
                'improve': [
                    "🔍 Analyze high-performing videos in your niche for patterns",
                    "📝 Consider updating title with trending keywords",
                    "🕒 Try reposting at peak audience hours",
                    "🎨 Test different thumbnail styles"
                ],
                'learn': [
                    "Focus on content quality improvements",
                    "Study competitor successful videos",
                    "Experiment with different content formats",
                    "Build audience through consistent posting"
                ]
            },
            'sentiment_negative': {
                'damage_control': [
                    "⚠️ Address negative feedback immediately with professional response",
                    "🔧 Consider making corrections or clarifications in comments",
                    "📢 Post community update if major concerns raised",
                    "💬 Engage constructively with criticism"
                ]
            },
            'sentiment_positive': {
                'amplify': [
                    "✨ Leverage positive sentiment - pin popular positive comments",
                    "💖 Thank engaged commenters personally",
                    "🎉 Share positive feedback in community posts",
                    "🔄 Encourage sharing with call-to-action"
                ]
            }
        }
        
        logger.info("✅ AI Advisory System initialized")
    
    def generate_advice(self, video_data: Dict, comments: List[Dict], 
                       viral_prediction: Dict, sentiment_analysis: Dict) -> Dict:
        """Generate comprehensive advice for content creators"""
        
        viral_probability = viral_prediction.get('viral_probability', 0)
        avg_sentiment = sentiment_analysis.get('average', 0)
        sentiment_trend = sentiment_analysis.get('trend', 'neutral')
        comment_count = len(comments)
        
        logger.info(f"Generating advice for video with {viral_probability:.1%} viral probability")
        
        # Generate structured recommendations
        recommendations = self._generate_recommendations(
            viral_probability, avg_sentiment, sentiment_trend, comment_count, comments
        )
        
        # Generate priority actions
        priority_actions = self._get_priority_actions(
            viral_probability, sentiment_trend, avg_sentiment
        )
        
        # Generate engagement strategy
        engagement_strategy = self._get_engagement_strategy(
            comment_count, avg_sentiment, viral_probability
        )
        
        # Generate content insights
        content_insights = self._analyze_content_patterns(comments, sentiment_analysis)
        
        # Generate timing recommendations
        timing_advice = self._get_timing_recommendations(
            video_data.get('minutes_since_upload', 0), viral_probability
        )
        
        return {
            'structured_recommendations': recommendations,
            'priority_actions': priority_actions,
            'engagement_strategy': engagement_strategy,
            'content_insights': content_insights,
            'timing_advice': timing_advice,
            'overall_assessment': self._get_overall_assessment(viral_probability, sentiment_trend),
            'next_steps': self._get_next_steps(viral_probability, comment_count),
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, viral_prob: float, avg_sentiment: float, 
                                sentiment_trend: str, comment_count: int, 
                                comments: List[Dict]) -> List[Dict]:
        """Generate structured recommendations based on current state"""
        
        recommendations = []
        
        # Viral probability based recommendations
        if viral_prob > 0.7:
            category = 'high_viral'
            urgency = 'high'
            advice_pool = self.advice_templates['high_viral']['urgent'] + \
                         self.advice_templates['high_viral']['strategic']
        elif viral_prob > 0.4:
            category = 'moderate_viral'
            urgency = 'medium'
            advice_pool = self.advice_templates['moderate_viral']['boost'] + \
                         self.advice_templates['moderate_viral']['optimize']
        else:
            category = 'low_viral'
            urgency = 'low'
            advice_pool = self.advice_templates['low_viral']['improve'] + \
                         self.advice_templates['low_viral']['learn']
        
        # Add 2-3 viral-based recommendations
        selected_advice = random.sample(advice_pool, min(3, len(advice_pool)))
        
        for advice in selected_advice:
            recommendations.append({
                'type': category,
                'action': advice.split('!')[0].replace('🔥', '').replace('🚀', '').strip(),
                'description': advice,
                'urgency': urgency,
                'confidence': 0.8 + viral_prob * 0.15  # Higher confidence for higher viral prob
            })
        
        # Sentiment-based recommendations
        if avg_sentiment < -0.3:
            sentiment_advice = random.choice(self.advice_templates['sentiment_negative']['damage_control'])
            recommendations.append({
                'type': 'sentiment_management',
                'action': 'Address negative feedback',
                'description': sentiment_advice,
                'urgency': 'high',
                'confidence': 0.9
            })
        elif avg_sentiment > 0.5:
            sentiment_advice = random.choice(self.advice_templates['sentiment_positive']['amplify'])
            recommendations.append({
                'type': 'sentiment_amplification',
                'action': 'Leverage positive sentiment',
                'description': sentiment_advice,
                'urgency': 'medium',
                'confidence': 0.85
            })
        
        # Comment volume based recommendations
        if comment_count > 50:
            recommendations.append({
                'type': 'high_engagement',
                'action': 'Manage high engagement',
                'description': "🔥 High comment volume! Consider going live or posting community update to capitalize on engagement.",
                'urgency': 'medium',
                'confidence': 0.8
            })
        elif comment_count < 5:
            recommendations.append({
                'type': 'low_engagement',
                'action': 'Boost initial engagement',
                'description': "📢 Low comment count - share with friends/family to get initial engagement momentum.",
                'urgency': 'high',
                'confidence': 0.7
            })
        
        return recommendations
    
    def _get_priority_actions(self, viral_prob: float, sentiment_trend: str, 
                            avg_sentiment: float) -> List[Dict]:
        """Get immediate priority actions"""
        
        actions = []
        
        # High priority actions based on viral probability
        if viral_prob > 0.6:
            actions.extend([
                {
                    'action': 'Enable monetization',
                    'reason': 'High viral potential detected',
                    'timeframe': 'Immediately',
                    'priority': 'urgent'
                },
                {
                    'action': 'Cross-platform sharing',
                    'reason': 'Capitalize on momentum',
                    'timeframe': 'Next 30 minutes',
                    'priority': 'high'
                },
                {
                    'action': 'Monitor analytics closely',
                    'reason': 'Track viral progression',
                    'timeframe': 'Next 2 hours',
                    'priority': 'high'
                }
            ])
        
        # Sentiment-based priority actions
        if sentiment_trend == 'declining' or avg_sentiment < -0.3:
            actions.append({
                'action': 'Address negative feedback',
                'reason': 'Prevent reputation damage',
                'timeframe': 'Next 15 minutes',
                'priority': 'urgent'
            })
        
        if sentiment_trend == 'improving' and viral_prob > 0.3:
            actions.append({
                'action': 'Amplify positive momentum',
                'reason': 'Sentiment improving with viral potential',
                'timeframe': 'Next hour',
                'priority': 'high'
            })
        
        return actions
    
    def _get_engagement_strategy(self, comment_count: int, avg_sentiment: float, 
                               viral_prob: float) -> Dict:
        """Generate engagement strategy recommendations"""
        
        if comment_count < 10:
            strategy_type = "bootstrap_engagement"
            description = "Focus on getting initial engagement through personal network and targeted sharing"
            tactics = [
                "Share with friends and family for initial comments",
                "Post in relevant niche communities",
                "Engage with early commenters immediately"
            ]
        elif comment_count < 50:
            strategy_type = "moderate_engagement"
            description = "Selectively engage with high-value comments and encourage discussions"
            tactics = [
                "Reply to questions and thoughtful comments",
                "Pin engaging comments to encourage more",
                "Ask follow-up questions in replies"
            ]
        else:
            strategy_type = "high_engagement"
            description = "Manage high volume engagement strategically"
            tactics = [
                "Focus on trending discussion threads",
                "Heart/like comments strategically",
                "Consider live engagement or community posts"
            ]
        
        # Adjust strategy based on viral probability
        if viral_prob > 0.6:
            tactics.append("Prepare for potential viral management")
            tactics.append("Consider creating follow-up content")
        
        return {
            'strategy_type': strategy_type,
            'description': description,
            'recommended_tactics': tactics,
            'suggested_response_rate': min(0.4, 15 / max(comment_count, 1)),
            'focus_areas': self._get_engagement_focus_areas(avg_sentiment)
        }
    
    def _get_engagement_focus_areas(self, avg_sentiment: float) -> List[str]:
        """Determine where to focus engagement efforts"""
        
        focus_areas = ['questions', 'constructive_feedback']
        
        if avg_sentiment > 0.3:
            focus_areas.extend(['positive_reactions', 'sharing_requests'])
        elif avg_sentiment < -0.2:
            focus_areas.extend(['concerns_and_criticism', 'clarifications'])
        else:
            focus_areas.extend(['general_discussion', 'topic_expansion'])
        
        return focus_areas
    
    def _analyze_content_patterns(self, comments: List[Dict], sentiment_analysis: Dict) -> Dict:
        """Analyze patterns in comments to provide content insights"""
        
        if not comments:
            return {'pattern': 'insufficient_data', 'insights': []}
        
        # Analyze comment themes
        common_words = self._extract_common_themes(comments)
        
        # Sentiment pattern
        sentiment_dist = sentiment_analysis.get('sentiment_distribution', {})
        
        insights = []
        
        # Theme-based insights
        if any(word in common_words for word in ['helpful', 'useful', 'learn']):
            insights.append("Educational content resonating well with audience")
        
        if any(word in common_words for word in ['funny', 'hilarious', 'lol']):
            insights.append("Humor is connecting with viewers")
        
        if any(word in common_words for word in ['more', 'next', 'part']):
            insights.append("Audience wants follow-up content - consider series format")
        
        # Sentiment-based insights
        if sentiment_dist.get('positive', 0) > 0.6:
            insights.append("Strong positive reception - replicate content style")
        elif sentiment_dist.get('negative', 0) > 0.4:
            insights.append("Mixed reception - analyze negative feedback for improvements")
        
        return {
            'common_themes': common_words[:10],
            'sentiment_pattern': sentiment_analysis.get('trend', 'neutral'),
            'insights': insights,
            'engagement_type': self._classify_engagement_type(comments)
        }
    
    def _extract_common_themes(self, comments: List[Dict]) -> List[str]:
        """Extract common words/themes from comments"""
        
        all_text = ' '.join([comment.get('text', '').lower() for comment in comments])
        
        # Simple word frequency (excluding common words)
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'this', 'it', 'of', 'you', 'that', 'for', 'with', 'have', 'be', 'not', 'or', 'as', 'from', 'they', 'but', 'by', 'so', 'can', 'if', 'would', 'there', 'what', 'about', 'get', 'all', 'were', 'when', 'we', 'your', 'an', 'are', 'my', 'one', 'time', 'has', 'had', 'up', 'his', 'her', 'who', 'oil', 'its', 'now', 'he', 'than', 'she', 'may', 'these', 'some', 'very', 'them', 'well', 'much'}
        
        words = [word.strip('.,!?;:"()[]{}') for word in all_text.split() if len(word) > 3]
        words = [word for word in words if word not in stop_words]
        
        # Count frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return most common words
        return sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)
    
    def _classify_engagement_type(self, comments: List[Dict]) -> str:
        """Classify the type of engagement happening"""
        
        if not comments:
            return 'none'
        
        # Analyze comment characteristics
        avg_length = sum(len(c.get('text', '')) for c in comments) / len(comments)
        total_questions = sum(1 for c in comments if '?' in c.get('text', ''))
        
        if avg_length > 100:
            return 'deep_discussion'
        elif total_questions / len(comments) > 0.3:
            return 'inquisitive_audience'
        elif avg_length < 30:
            return 'quick_reactions'
        else:
            return 'balanced_engagement'
    
    def _get_timing_recommendations(self, minutes_since_upload: int, viral_prob: float) -> Dict:
        """Generate timing-based recommendations"""
        
        if minutes_since_upload < 30:
            timing_status = "golden_hour"
            advice = "Critical first 30 minutes - maximize promotion efforts now"
        elif minutes_since_upload < 120:
            timing_status = "early_phase"
            advice = "Still in early engagement window - good time for community sharing"
        elif minutes_since_upload < 360:
            timing_status = "growth_phase"
            advice = "Monitor momentum - consider targeted promotion if engagement is strong"
        else:
            timing_status = "mature_phase"
            advice = "Focus on long-term optimization and audience building"
        
        return {
            'timing_status': timing_status,
            'minutes_since_upload': minutes_since_upload,
            'advice': advice,
            'next_milestone': self._get_next_timing_milestone(minutes_since_upload),
            'optimal_actions': self._get_time_specific_actions(minutes_since_upload, viral_prob)
        }
    
    def _get_next_timing_milestone(self, current_minutes: int) -> Dict:
        """Get the next important timing milestone"""
        
        milestones = [30, 60, 120, 360, 720, 1440]  # 30min, 1hr, 2hr, 6hr, 12hr, 24hr
        
        for milestone in milestones:
            if current_minutes < milestone:
                return {
                    'milestone_minutes': milestone,
                    'time_remaining': milestone - current_minutes,
                    'significance': self._get_milestone_significance(milestone)
                }
        
        return {
            'milestone_minutes': 1440,
            'time_remaining': 0,
            'significance': '24-hour performance established'
        }
    
    def _get_milestone_significance(self, milestone: int) -> str:
        """Get significance of timing milestones"""
        
        significance_map = {
            30: "Critical engagement window",
            60: "First hour algorithm boost",
            120: "Early viral determination point",
            360: "Extended engagement window",
            720: "Half-day performance indicator",
            1440: "24-hour success benchmark"
        }
        
        return significance_map.get(milestone, "Performance milestone")
    
    def _get_time_specific_actions(self, minutes: int, viral_prob: float) -> List[str]:
        """Get actions specific to current timing"""
        
        actions = []
        
        if minutes < 30:
            actions.extend([
                "Share on personal social media immediately",
                "Notify subscribers through community post",
                "Engage with every comment personally"
            ])
        elif minutes < 120:
            if viral_prob > 0.5:
                actions.extend([
                    "Share in relevant Facebook groups",
                    "Post on Twitter with trending hashtags",
                    "Consider reaching out to influencer friends"
                ])
            else:
                actions.extend([
                    "Share in niche communities",
                    "Reply to comments to boost engagement",
                    "Cross-post on other platforms"
                ])
        else:
            actions.extend([
                "Focus on long-term SEO optimization",
                "Plan follow-up content based on reception",
                "Analyze performance data for insights"
            ])
        
        return actions
    
    def _get_overall_assessment(self, viral_prob: float, sentiment_trend: str) -> Dict:
        """Generate overall assessment of video performance"""
        
        if viral_prob > 0.7:
            status = "excellent"
            message = "🚀 Outstanding viral potential! This content is performing exceptionally well."
        elif viral_prob > 0.5:
            status = "very_good"
            message = "⭐ Strong performance with good viral indicators."
        elif viral_prob > 0.3:
            status = "moderate"
            message = "📈 Decent performance with room for optimization."
        else:
            status = "needs_improvement"
            message = "🔧 Performance below average - focus on content and promotion improvements."
        
        # Adjust based on sentiment
        if sentiment_trend == 'declining':
            message += " Monitor negative feedback closely."
        elif sentiment_trend == 'improving':
            message += " Positive momentum building!"
        
        return {
            'status': status,
            'message': message,
            'viral_probability': viral_prob,
            'sentiment_trend': sentiment_trend,
            'recommendation': self._get_status_recommendation(status)
        }
    
    def _get_status_recommendation(self, status: str) -> str:
        """Get recommendation based on overall status"""
        
        recommendations = {
            'excellent': "Maximize promotion and prepare for scaling",
            'very_good': "Strategic promotion and engagement optimization",
            'moderate': "Focus on engagement and targeted sharing",
            'needs_improvement': "Analyze and improve content strategy"
        }
        
        return recommendations.get(status, "Continue monitoring and optimizing")
    
    def _get_next_steps(self, viral_prob: float, comment_count: int) -> List[str]:
        """Get specific next steps for the creator"""
        
        next_steps = []
        
        # Immediate next steps based on performance
        if viral_prob > 0.6:
            next_steps.extend([
                "Monitor analytics every 30 minutes for next 4 hours",
                "Prepare follow-up content while momentum is high",
                "Engage with top comments immediately",
                "Share performance update in community tab"
            ])
        elif viral_prob > 0.3:
            next_steps.extend([
                "Share in 3-5 relevant communities within next hour",
                "Reply to all comments within next 2 hours",
                "Consider boosting with small ad spend",
                "Plan content series if topic is resonating"
            ])
        else:
            next_steps.extend([
                "Analyze top-performing competitor videos",
                "Optimize title and thumbnail if possible",
                "Focus on SEO improvements",
                "Plan content improvements for next video"
            ])
        
        # Comment management steps
        if comment_count > 20:
            next_steps.append("Prioritize replies to high-engagement comments")
        elif comment_count < 5:
            next_steps.append("Reach out to personal network for initial engagement")
        
        return next_steps

# Test function
def test_ai_advisory():
    """Test the AI advisory system"""
    advisory = AIAdvisorySystem()
    
    # Test data
    test_video_data = {'minutes_since_upload': 45}
    test_comments = [
        {'text': 'This is amazing! Thanks for sharing!', 'author': 'User1'},
        {'text': 'Great tutorial, very helpful', 'author': 'User2'},
        {'text': 'Could you make a part 2?', 'author': 'User3'}
    ]
    test_viral_prediction = {'viral_probability': 0.75}
    test_sentiment_analysis = {'average': 0.6, 'trend': 'improving'}
    
    print("Testing AI Advisory System:")
    advice = advisory.generate_advice(
        test_video_data, test_comments, test_viral_prediction, test_sentiment_analysis
    )
    
    print(f"Overall Assessment: {advice['overall_assessment']['message']}")
    print(f"Number of recommendations: {len(advice['structured_recommendations'])}")
    print(f"Priority actions: {len(advice['priority_actions'])}")

if __name__ == "__main__":
    test_ai_advisory()
