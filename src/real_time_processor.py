import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import json

from .collector import YouTubeDataCollector
from .spam_detector import PreTrainedSpamDetector
from .sentiment_analyzer import PreTrainedSentimentAnalyzer
from .feature_extractor import FeatureExtractor
from .viral_predictor import LLMViralPredictor
from .ai_advisory import AIAdvisorySystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeProcessor:
    def __init__(self):
        logger.info("Initializing Real-Time Processor...")
        
        # Initialize all components
        self.collector = YouTubeDataCollector()
        self.spam_detector = PreTrainedSpamDetector()
        self.sentiment_analyzer = PreTrainedSentimentAnalyzer()
        self.feature_extractor = FeatureExtractor()
        self.viral_predictor = LLMViralPredictor()
        self.ai_advisory = AIAdvisorySystem()
        
        # Storage for results and monitoring
        self.results_cache = {}
        self.monitoring_threads = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info("✅ Real-Time Processor initialized successfully!")
    
    def analyze_video(self, video_id: str, detailed: bool = True) -> Dict:
        """Perform complete analysis of a video"""
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Starting analysis for video: {video_id}")
            
            # Step 1: Get video metadata
            video_details = self.collector.get_video_details(video_id)
            if not video_details:
                return self._get_error_analysis(video_id, "Could not fetch video details")
            
            # Step 2: Get comments
            raw_comments = self.collector.get_video_comments(video_id, max_results=100)
            if not raw_comments:
                return self._get_empty_analysis(video_id, video_details)
            
            logger.info(f"📊 Processing {len(raw_comments)} comments...")
            
            # Step 3: Filter spam
            clean_comments = self.spam_detector.filter_comments(raw_comments)
            
            # Step 4: Analyze sentiment
            comments_with_sentiment = self.sentiment_analyzer.analyze_comment_batch(clean_comments)
            sentiment_analysis = self.sentiment_analyzer.calculate_sentiment_trajectory(comments_with_sentiment)
            
            # Step 5: Extract features
            minutes_since_upload = self.collector.calculate_minutes_since_upload(video_id)
            video_metadata = {**video_details, 'minutes_since_upload': minutes_since_upload}
            features = self.feature_extractor.extract_all_features(comments_with_sentiment, video_metadata)
            
            # Step 6: Predict viral probability
            viral_probability, prediction_details = self.viral_predictor.predict_viral_probability(
                features, comments_with_sentiment, video_metadata
            )
            
            # Step 7: Generate AI advice (if detailed analysis requested)
            ai_advice = {}
            if detailed:
                ai_advice = self.ai_advisory.generate_advice(
                    video_metadata, comments_with_sentiment, prediction_details, sentiment_analysis
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Compile comprehensive results
            analysis = {
                'video_id': video_id,
                'timestamp': datetime.now().isoformat(),
                'video_details': video_details,
                'comment_stats': {
                    'total_comments': len(raw_comments),
                    'clean_comments': len(clean_comments),
                    'spam_filtered': len(raw_comments) - len(clean_comments),
                    'spam_percentage': (len(raw_comments) - len(clean_comments)) / len(raw_comments) * 100 if raw_comments else 0
                },
                'sentiment_analysis': sentiment_analysis,
                'features': features,
                'viral_prediction': prediction_details,
                'ai_advice': ai_advice,
                'performance_metrics': {
                    'processing_time_seconds': round(processing_time, 2),
                    'minutes_since_upload': minutes_since_upload,
                    'analysis_quality': self._assess_analysis_quality(len(clean_comments), processing_time)
                },
                'model_info': {
                    'spam_model': self.spam_detector.model_name,
                    'sentiment_model': self.sentiment_analyzer.model_name,
                    'viral_model': self.viral_predictor.model_name,
                    'analysis_version': '1.0'
                }
            }
            
            # Cache results
            self.results_cache[video_id] = analysis
            
            logger.info(f"✅ Analysis complete! Viral probability: {viral_probability:.1%} (processed in {processing_time:.2f}s)")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing video {video_id}: {e}")
            return self._get_error_analysis(video_id, str(e))
    
    def start_monitoring(self, video_id: str, duration_minutes: int = 60, 
                        check_interval_seconds: int = 30) -> bool:
        """Start continuous monitoring of a video"""
        
        if video_id in self.monitoring_threads:
            logger.warning(f"Already monitoring video {video_id}")
            return False
        
        logger.info(f"🔄 Starting monitoring for video {video_id} (duration: {duration_minutes}min, interval: {check_interval_seconds}s)")
        
        def monitoring_loop():
            end_time = datetime.now() + timedelta(minutes=duration_minutes)
            iteration_count = 0
            
            while datetime.now() < end_time:
                try:
                    iteration_count += 1
                    logger.info(f"📊 Monitoring iteration {iteration_count} for video {video_id}")
                    
                    # Perform analysis
                    analysis = self.analyze_video(video_id, detailed=(iteration_count % 3 == 1))  # Detailed analysis every 3rd iteration
                    
                    # Log key metrics
                    if 'viral_prediction' in analysis:
                        viral_prob = analysis['viral_prediction'].get('viral_probability', 0)
                        comment_count = analysis['comment_stats']['clean_comments']
                        
                        logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] {video_id}: "
                                  f"Viral: {viral_prob:.1%}, Comments: {comment_count}")
                        
                        # Store monitoring data point
                        self._store_monitoring_point(video_id, analysis, iteration_count)
                    
                    # Sleep until next check
                    time.sleep(check_interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop for {video_id}: {e}")
                    time.sleep(min(check_interval_seconds * 2, 300))  # Back off on error
            
            # Monitoring completed
            if video_id in self.monitoring_threads:
                del self.monitoring_threads[video_id]
            
            logger.info(f"✅ Monitoring completed for video {video_id} after {iteration_count} iterations")
        
        # Start monitoring thread
        thread = threading.Thread(target=monitoring_loop, daemon=True)
        thread.start()
        self.monitoring_threads[video_id] = {
            'thread': thread,
            'start_time': datetime.now(),
            'duration_minutes': duration_minutes,
            'check_interval': check_interval_seconds
        }
        
        return True
    
    def stop_monitoring(self, video_id: str) -> bool:
        """Stop monitoring a video"""
        if video_id in self.monitoring_threads:
            # Note: Can't actually stop thread, but remove from tracking
            del self.monitoring_threads[video_id]
            logger.info(f"🛑 Stopped monitoring video {video_id}")
            return True
        return False
    
    def get_latest_analysis(self, video_id: str) -> Optional[Dict]:
        """Get the latest analysis for a video"""
        return self.results_cache.get(video_id)
    
    def get_monitoring_status(self) -> Dict:
        """Get status of all monitoring operations"""
        status = {
            'active_monitoring': {},
            'cached_results': list(self.results_cache.keys()),
            'total_videos_analyzed': len(self.results_cache),
            'system_status': 'operational'
        }
        
        # Add details for active monitoring
        for video_id, info in self.monitoring_threads.items():
            status['active_monitoring'][video_id] = {
                'started_at': info['start_time'].isoformat(),
                'duration_minutes': info['duration_minutes'],
                'check_interval_seconds': info['check_interval'],
                'running_for_minutes': int((datetime.now() - info['start_time']).total_seconds() / 60)
            }
        
        return status
    
    def get_video_timeline(self, video_id: str) -> List[Dict]:
        """Get timeline of analysis results for a video"""
        # This would typically come from a database, but for now use cache
        timeline_key = f"{video_id}_timeline"
        return getattr(self, timeline_key, [])
    
    def batch_analyze(self, video_ids: List[str]) -> Dict[str, Dict]:
        """Analyze multiple videos in parallel"""
        logger.info(f"🔄 Starting batch analysis for {len(video_ids)} videos")
        
        results = {}
        
        # Use ThreadPoolExecutor for parallel processing
        future_to_video = {
            self.thread_pool.submit(self.analyze_video, video_id): video_id 
            for video_id in video_ids
        }
        
        for future in future_to_video:
            video_id = future_to_video[future]
            try:
                result = future.result(timeout=60)  # 60 second timeout per video
                results[video_id] = result
                logger.info(f"✅ Completed analysis for {video_id}")
            except Exception as e:
                logger.error(f"❌ Failed to analyze {video_id}: {e}")
                results[video_id] = self._get_error_analysis(video_id, str(e))
        
        logger.info(f"📊 Batch analysis completed: {len(results)} results")
        return results
    
    def _store_monitoring_point(self, video_id: str, analysis: Dict, iteration: int):
        """Store a monitoring data point for timeline tracking"""
        timeline_key = f"{video_id}_timeline"
        
        if not hasattr(self, timeline_key):
            setattr(self, timeline_key, [])
        
        timeline = getattr(self, timeline_key)
        
        data_point = {
            'timestamp': analysis['timestamp'],
            'iteration': iteration,
            'viral_probability': analysis['viral_prediction'].get('viral_probability', 0),
            'comment_count': analysis['comment_stats']['clean_comments'],
            'sentiment_average': analysis['sentiment_analysis'].get('average', 0),
            'minutes_since_upload': analysis['performance_metrics']['minutes_since_upload']
        }
        
        timeline.append(data_point)
        
        # Keep only last 100 data points
        if len(timeline) > 100:
            timeline.pop(0)
    
    def _assess_analysis_quality(self, comment_count: int, processing_time: float) -> str:
        """Assess the quality of analysis based on data availability"""
        
        if comment_count >= 50 and processing_time < 30:
            return 'excellent'
        elif comment_count >= 20 and processing_time < 60:
            return 'good'
        elif comment_count >= 5 and processing_time < 120:
            return 'fair'
        else:
            return 'limited'
    
    def _get_empty_analysis(self, video_id: str, video_details: Dict) -> Dict:
        """Return analysis structure for videos with no comments"""
        return {
            'video_id': video_id,
            'timestamp': datetime.now().isoformat(),
            'video_details': video_details,
            'comment_stats': {
                'total_comments': 0,
                'clean_comments': 0,
                'spam_filtered': 0,
                'spam_percentage': 0
            },
            'sentiment_analysis': {
                'trend': 'no_data',
                'average': 0.0,
                'volatility': 0.0,
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0}
            },
            'viral_prediction': {
                'viral_probability': 0.05,  # Very low but not zero
                'prediction_class': 'no_data',
                'confidence': 0.1,
                'method': 'no_comments_fallback'
            },
            'ai_advice': {
                'structured_recommendations': [{
                    'type': 'no_comments',
                    'action': 'Boost initial engagement',
                    'description': 'No comments yet - share with personal network to get initial engagement',
                    'urgency': 'high',
                    'confidence': 0.9
                }]
            },
            'performance_metrics': {
                'processing_time_seconds': 0.1,
                'minutes_since_upload': self.collector.calculate_minutes_since_upload(video_id),
                'analysis_quality': 'no_data'
            }
        }
    
    def _get_error_analysis(self, video_id: str, error_message: str) -> Dict:
        """Return error analysis structure"""
        return {
            'video_id': video_id,
            'timestamp': datetime.now().isoformat(),
            'error': error_message,
            'video_details': {},
            'comment_stats': {'total_comments': 0, 'clean_comments': 0, 'spam_filtered': 0},
            'sentiment_analysis': {'trend': 'error', 'average': 0.0, 'volatility': 0.0},
            'viral_prediction': {'viral_probability': 0.0, 'prediction_class': 'error', 'confidence': 0.0},
            'ai_advice': {},
            'performance_metrics': {'processing_time_seconds': 0, 'analysis_quality': 'error'}
        }
    
    def __del__(self):
        """Cleanup when processor is destroyed"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)

# Test function
def test_real_time_processor():
    """Test the real-time processor"""
    processor = RealTimeProcessor()
    
    # Test with a known video ID
    test_video_id = "dQw4w9WgXcQ"  # Rick Roll - always has comments
    
    print("Testing single video analysis...")
    analysis = processor.analyze_video(test_video_id)
    
    print(f"Video: {analysis.get('video_details', {}).get('title', 'Unknown')}")
    print(f"Viral Probability: {analysis.get('viral_prediction', {}).get('viral_probability', 0):.1%}")
    print(f"Comments: {analysis.get('comment_stats', {}).get('clean_comments', 0)}")
    print(f"Processing Time: {analysis.get('performance_metrics', {}).get('processing_time_seconds', 0):.2f}s")

if __name__ == "__main__":
    test_real_time_processor()
