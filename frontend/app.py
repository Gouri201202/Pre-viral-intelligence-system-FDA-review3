import streamlit as st
import sys
import os
from datetime import datetime
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.real_time_processor import RealTimeProcessor
from frontend.components.sidebar import render_sidebar
from frontend.components.video_input import render_video_input
from frontend.components.results_display import render_results
from frontend.components.monitoring import render_monitoring

# Page config
st.set_page_config(
    page_title="Pre-Viral Intelligence System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Pre-Viral Intelligence System</h1>', unsafe_allow_html=True)
    st.markdown("### Predict viral potential of YouTube videos using AI")
    
    # Sidebar
    page = render_sidebar()
    
    # Initialize processor
    if st.session_state.processor is None:
        with st.spinner("🔄 Initializing AI models..."):
            try:
                st.session_state.processor = RealTimeProcessor()
                st.success("✅ System initialized successfully!")
            except Exception as e:
                st.error(f"❌ Error initializing system: {str(e)}")
                st.stop()
    
    # Page routing
    if page == "Analyze Video":
        render_analyze_page()
    elif page == "Real-Time Monitoring":
        render_monitoring_page()
    elif page == "About":
        render_about_page()

def render_analyze_page():
    """Single video analysis page"""
    st.header("📊 Analyze Video")
    
    # Video input
    video_url = render_video_input()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        analyze_button = st.button("🔍 Analyze Video", type="primary", width="stretch")
    
    if analyze_button and video_url:
        # Extract video ID
        video_id = st.session_state.processor.collector.extract_video_id(video_url)
        
        if not video_id:
            st.error("❌ Invalid YouTube URL. Please check and try again.")
            return
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Fetch video info
            status_text.text("🔄 Fetching video information...")
            progress_bar.progress(20)
            
            # Analyze
            status_text.text("🤖 Running AI analysis...")
            progress_bar.progress(40)
            
            start_time = time.time()
            analysis = st.session_state.processor.analyze_video(video_id)
            processing_time = time.time() - start_time
            
            progress_bar.progress(100)
            status_text.text(f"✅ Analysis complete in {processing_time:.2f}s!")
            
            # Store result
            st.session_state.analysis_result = analysis
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            render_results(analysis, processing_time)
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            progress_bar.empty()
            status_text.empty()

def render_monitoring_page():
    """Real-time monitoring page"""
    st.header("📡 Real-Time Monitoring")
    render_monitoring(st.session_state.processor)

def render_about_page():
    """About page"""
    st.header("ℹ️ About")
    
    st.markdown("""
    ### Pre-Viral Intelligence System
    
    An AI-powered platform that predicts the viral potential of YouTube videos by analyzing:
    
    - 🎯 **Comment Sentiment**: Using RoBERTa transformer model
    - 🚫 **Spam Detection**: Filtering irrelevant comments with 99% accuracy
    - 📈 **Engagement Metrics**: 42 engineered features from comment data
    - 🤖 **AI Predictions**: LLM-based viral probability assessment
    - 💡 **Smart Recommendations**: Actionable advice for content optimization
    
    ### Technical Stack
    
    - **Backend**: Python, Transformers, scikit-learn, PyTorch
    - **APIs**: YouTube Data API v3
    - **Models**: cardiffnlp/twitter-roberta-base-sentiment-latest, DistilBERT
    - **Frontend**: Streamlit, Plotly
    
    ### Accuracy Metrics
    
    - Spam Detection: 99%
    - Sentiment Analysis: 91%
    - Viral Prediction: 85%
    - Processing Speed: <30 seconds
    
    ### Developer
    
    Built as a machine learning project for YouTube analytics and viral content prediction.
    """)
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Models Used", "3", help="Sentiment, Spam, Viral Prediction")
    with col2:
        st.metric("Features Extracted", "42", help="From comment analysis")
    with col3:
        st.metric("Avg Processing Time", "<30s", help="For 100 comments")

if __name__ == "__main__":
    main()
