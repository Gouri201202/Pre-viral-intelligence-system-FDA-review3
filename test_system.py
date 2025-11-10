import sys
sys.path.append('.')
from src.real_time_processor import RealTimeProcessor

# Initialize the system
processor = RealTimeProcessor()

# Paste your YouTube video URL here
video_url = "https://youtu.be/SlK-0ha1bTw?si=YyA39LcNHnvppKRP"  # Example: Hugging Face Tutorial for Beginners
video_id = processor.collector.extract_video_id(video_url)

print("\U0001F50D Analyzing video...")
analysis = processor.analyze_video(video_id)

print(f"\n\U0001F4CA RESULTS:")
print(f"Video: {analysis['video_details']['title']}")
print(f"Viral Probability: {analysis['viral_prediction']['viral_probability']:.1%}")
print(f"Comments Analyzed: {analysis['comment_stats']['clean_comments']}")
print(f"Sentiment: {analysis['sentiment_analysis']['average']:.2f}")

print("\n\U0001F916 AI Recommendations:")
for rec in analysis['ai_advice']['structured_recommendations']:
    print(f"- {rec['description']}")
