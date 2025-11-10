import streamlit as st

def render_video_input():
    """Render video URL input component"""
    
    st.markdown("### 🎥 Enter YouTube Video URL")
    
    # Check if example was selected
    default_url = ""
    if 'selected_example' in st.session_state:
        default_url = st.session_state.selected_example
        del st.session_state.selected_example
    
    video_url = st.text_input(
        "YouTube URL",
        value=default_url,
        placeholder="https://www.youtube.com/watch?v=VIDEO_ID or https://youtu.be/VIDEO_ID",
        label_visibility="collapsed",
        help="Paste any YouTube video URL here"
    )
    
    # Example videos
    with st.expander("📝 Try Example Videos"):
        st.markdown("Click to use example URLs:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎓 Tutorial Video", key="example1"):
                st.session_state.selected_example = "https://www.youtube.com/watch?v=3xLTD5wSBEs"
                st.rerun()
        
        with col2:
            if st.button("🎵 Music Video", key="example2"):
                st.session_state.selected_example = "https://www.youtube.com/watch?v=SlK-0ha1bTw"
                st.rerun()
        
        st.divider()
        
        # Show URLs for manual copy-paste
        st.markdown("**Or copy these URLs:**")
        st.code("https://www.youtube.com/watch?v=3xLTD5wSBEs", language=None)
        st.code("https://www.youtube.com/watch?v=SlK-0ha1bTw", language=None)
    
    return video_url
