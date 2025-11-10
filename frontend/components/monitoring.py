import streamlit as st
import time
from datetime import datetime

def render_monitoring(processor):
    """Render real-time monitoring UI"""
    
    st.markdown("### 📡 Monitor Video in Real-Time")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=VIDEO_ID")
    
    with col2:
        duration = st.number_input("Duration (minutes)", min_value=5, max_value=120, value=30, step=5)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_button = st.button("▶️ Start Monitoring", type="primary", width="stretch")
    with col2:
        stop_button = st.button("⏸️ Stop Monitoring",width="stretch")
    with col3:
        st.session_state.monitoring_active = st.checkbox("Active", value=st.session_state.get('monitoring_active', False))
    
    if start_button and video_url:
        video_id = processor.collector.extract_video_id(video_url)
        if video_id:
            st.session_state.monitoring_active = True
            st.session_state.monitoring_video_id = video_id
            st.success(f"✅ Started monitoring video: {video_id}")
        else:
            st.error("❌ Invalid YouTube URL")
    
    if stop_button:
        st.session_state.monitoring_active = False
        st.info("⏸️ Monitoring stopped")
    
    # Monitoring display
    if st.session_state.get('monitoring_active') and st.session_state.get('monitoring_video_id'):
        render_monitoring_display(processor, st.session_state.monitoring_video_id, duration)

def render_monitoring_display(processor, video_id, duration):
    """Display monitoring results"""
    
    st.divider()
    st.subheader("📊 Live Monitoring Dashboard")
    
    # Placeholder for live updates
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    interval = st.session_state.get('monitor_interval', 60)
    iterations = 0
    max_iterations = (duration * 60) // interval
    
    while st.session_state.monitoring_active and iterations < max_iterations:
        iterations += 1
        
        # Update status
        status_placeholder.info(f"🔄 Update {iterations}/{max_iterations} - {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Analyze
            analysis = processor.analyze_video(video_id)
            
            # Display metrics
            with metrics_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Viral Probability", f"{analysis['viral_prediction']['viral_probability']:.1%}")
                with col2:
                    st.metric("Sentiment", f"{analysis['sentiment_analysis']['average']:.2f}")
                with col3:
                    st.metric("Comments", analysis['comment_stats']['clean_comments'])
            
            # Wait for next iteration
            time.sleep(interval)
            
        except Exception as e:
            status_placeholder.error(f"❌ Error: {str(e)}")
            break
    
    if iterations >= max_iterations:
        status_placeholder.success("✅ Monitoring completed!")
        st.session_state.monitoring_active = False
