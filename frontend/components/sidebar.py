import streamlit as st

def render_sidebar():
    """Render sidebar navigation"""
    with st.sidebar:
        # Logo placeholder (using text since we don't have an actual logo)
        st.markdown("### 🚀 Pre-Viral AI")
        st.markdown("---")
        
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["Analyze Video", "Real-Time Monitoring", "About"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.subheader("⚙️ Settings")
        
        # Settings
        st.slider("Comment Limit", 50, 200, 100, 10, key="comment_limit")
        st.slider("Monitoring Interval (sec)", 30, 300, 60, 30, key="monitor_interval")
        
        st.divider()
        
        # System info
        st.subheader("📊 System Status")
        if st.session_state.get('processor'):
            st.success("✅ System Ready")
            
            # Show processor info
            if st.session_state.get('analysis_result'):
                st.info(f"Last analysis: {st.session_state.analysis_result.get('timestamp', 'N/A')[:19]}")
        else:
            st.warning("⏳ Initializing...")
        
        st.divider()
        
        # Footer
        st.caption("Pre-Viral Intelligence v1.0")
        st.caption("Powered by AI & ML")
        
        return page
