import googleapiclient.discovery
import pandas as pd
import json
from datetime import datetime, timezone
from typing import List, Dict, Optional
import time
import logging
from .config import YOUTUBE_API_KEY, YOUTUBE_API_QUOTA_PER_DAY, MAX_COMMENTS_PER_REQUEST

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeDataCollector:
    def __init__(self, api_key: str = YOUTUBE_API_KEY):
        if not api_key:
            raise ValueError("YouTube API key is required. Please set YOUTUBE_API_KEY in .env file")
        
        self.api_key = api_key
        self.youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=api_key
        )
        self.requests_made = 0
        self.daily_quota_used = 0
        
        logger.info("✅ YouTube Data Collector initialized")
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats"""
        try:
            if "watch?v=" in url:
                return url.split("watch?v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                return url.split("youtu.be/")[1].split("?")[0]
            elif "youtube.com/embed/" in url:
                return url.split("youtube.com/embed/")[1].split("?")[0]
            else:
                # Assume it's already a video ID
                if len(url) == 11:  # YouTube video IDs are 11 characters
                    return url
                else:
                    logger.error(f"Could not extract video ID from: {url}")
                    return None
        except Exception as e:
            logger.error(f"Error extracting video ID: {e}")
            return None
    
    def get_video_details(self, video_id: str) -> Dict:
        """Get comprehensive video metadata"""
        try:
            request = self.youtube.videos().list(
                part="snippet,statistics,status",
                id=video_id
            )
            response = request.execute()
            self._update_quota_usage(1)
            
            if response['items']:
                item = response['items'][0]
                snippet = item['snippet']
                statistics = item['statistics']
                
                return {
                    'video_id': video_id,
                    'title': snippet.get('title', 'Unknown Title'),
                    'channel_name': snippet.get('channelTitle', 'Unknown Channel'),
                    'channel_id': snippet.get('channelId', ''),
                    'published_at': snippet.get('publishedAt', ''),
                    'description': snippet.get('description', '')[:500] + '...',
                    'category_id': snippet.get('categoryId', ''),
                    'tags': snippet.get('tags', []),
                    'view_count': int(statistics.get('viewCount', 0)),
                    'like_count': int(statistics.get('likeCount', 0)),
                    'comment_count': int(statistics.get('commentCount', 0)),
                    'duration': self._get_video_duration(video_id)
                }
            else:
                logger.warning(f"No video details found for ID: {video_id}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting video details for {video_id}: {e}")
            return {}
    
    def get_video_comments(self, video_id: str, max_results: int = MAX_COMMENTS_PER_REQUEST) -> List[Dict]:
        """Get comments from a video with comprehensive error handling"""
        comments = []
        
        try:
            logger.info(f"Fetching comments for video: {video_id}")
            
            request = self.youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=min(max_results, 100),
                order="time",  # Get chronological order for early prediction
                textFormat="plainText"
            )
            
            response = request.execute()
            self._update_quota_usage(1)
            
            for item in response['items']:
                try:
                    top_comment = item['snippet']['topLevelComment']['snippet']
                    
                    comment_data = {
                        'comment_id': item['id'],
                        'text': top_comment.get('textDisplay', ''),
                        'author': top_comment.get('authorDisplayName', 'Anonymous'),
                        'author_channel_id': top_comment.get('authorChannelId', {}).get('value', ''),
                        'published_at': top_comment.get('publishedAt', ''),
                        'updated_at': top_comment.get('updatedAt', ''),
                        'like_count': int(top_comment.get('likeCount', 0)),
                        'reply_count': int(item['snippet'].get('totalReplyCount', 0)),
                        'is_reply': False
                    }
                    
                    comments.append(comment_data)
                    
                    # Add replies if they exist and we want them
                    if 'replies' in item and len(comments) < max_results:
                        for reply in item['replies']['comments'][:3]:  # Limit replies
                            try:
                                reply_snippet = reply['snippet']
                                reply_data = {
                                    'comment_id': reply['id'],
                                    'text': reply_snippet.get('textDisplay', ''),
                                    'author': reply_snippet.get('authorDisplayName', 'Anonymous'),
                                    'author_channel_id': reply_snippet.get('authorChannelId', {}).get('value', ''),
                                    'published_at': reply_snippet.get('publishedAt', ''),
                                    'updated_at': reply_snippet.get('updatedAt', ''),
                                    'like_count': int(reply_snippet.get('likeCount', 0)),
                                    'reply_count': 0,
                                    'is_reply': True,
                                    'parent_id': item['id']
                                }
                                comments.append(reply_data)
                            except Exception as reply_error:
                                logger.warning(f"Error processing reply: {reply_error}")
                                continue
                    
                except Exception as comment_error:
                    logger.warning(f"Error processing comment: {comment_error}")
                    continue
            
            logger.info(f"Successfully fetched {len(comments)} comments")
            return comments[:max_results]
        
        except googleapiclient.errors.HttpError as e:
            if "commentsDisabled" in str(e):
                logger.warning(f"Comments are disabled for video: {video_id}")
            elif "quotaExceeded" in str(e):
                logger.error("YouTube API quota exceeded!")
            else:
                logger.error(f"YouTube API error: {e}")
            return []
        
        except Exception as e:
            logger.error(f"Unexpected error fetching comments: {e}")
            return []
    
    def calculate_minutes_since_upload(self, video_id: str) -> int:
        """Calculate minutes since video was uploaded"""
        try:
            video_details = self.get_video_details(video_id)
            if video_details and 'published_at' in video_details:
                upload_time = datetime.fromisoformat(
                    video_details['published_at'].replace('Z', '+00:00')
                )
                now = datetime.now(timezone.utc)
                delta_minutes = int((now - upload_time).total_seconds() / 60)
                return max(0, delta_minutes)  # Ensure non-negative
        except Exception as e:
            logger.error(f"Error calculating upload time: {e}")
        
        return 0
    
    def _get_video_duration(self, video_id: str) -> str:
        """Get video duration in ISO 8601 format"""
        try:
            request = self.youtube.videos().list(
                part="contentDetails",
                id=video_id
            )
            response = request.execute()
            
            if response['items']:
                return response['items'][0]['contentDetails']['duration']
        except Exception as e:
            logger.warning(f"Could not get video duration: {e}")
        
        return "PT0M0S"  # Default duration
    
    def _update_quota_usage(self, cost: int):
        """Track API quota usage"""
        self.requests_made += 1
        self.daily_quota_used += cost
        
        if self.daily_quota_used > YOUTUBE_API_QUOTA_PER_DAY * 0.9:
            logger.warning(f"Approaching daily quota limit: {self.daily_quota_used}/{YOUTUBE_API_QUOTA_PER_DAY}")
    
    def get_quota_status(self) -> Dict:
        """Get current quota usage status"""
        return {
            'requests_made': self.requests_made,
            'quota_used': self.daily_quota_used,
            'quota_limit': YOUTUBE_API_QUOTA_PER_DAY,
            'quota_remaining': YOUTUBE_API_QUOTA_PER_DAY - self.daily_quota_used
        }

# Test function
def test_collector():
    """Test the YouTube collector with a popular video"""
    collector = YouTubeDataCollector()
    
    # Test with Rick Roll video (always has comments)
    test_video_id = "dQw4w9WgXcQ"
    
    print("Testing video details...")
    details = collector.get_video_details(test_video_id)
    print(f"Video: {details.get('title', 'Unknown')}")
    
    print("Testing comment collection...")
    comments = collector.get_video_comments(test_video_id, max_results=10)
    print(f"Collected {len(comments)} comments")
    
    for comment in comments[:3]:
        print(f"- {comment['text'][:100]}...")

if __name__ == "__main__":
    test_collector()
