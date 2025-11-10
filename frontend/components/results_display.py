import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def render_results(analysis, processing_time):
    """Render analysis results"""
    
    st.divider()
    st.header("📊 Analysis Results")
    
    # Video info
    render_video_info(analysis.get('video_details', {}))
    
    st.divider()
    
    # Key metrics
    render_key_metrics(analysis, processing_time)
    
    st.divider()
    
    # Detailed analysis
    col1, col2 = st.columns(2)
    
    with col1:
        render_sentiment_analysis(analysis.get('sentiment_analysis', {}))
        render_comment_stats(analysis.get('comment_stats', {}))
    
    with col2:
        render_viral_prediction(analysis.get('viral_prediction', {}))
        render_ai_recommendations(analysis.get('ai_advice', {}))

def render_video_info(video_details):
    """Display video information"""
    st.subheader("🎥 Video Information")
    
    if not video_details:
        st.warning("⚠️ Video details not available")
        return
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Thumbnail - construct URL from video ID if not provided
        video_id = video_details.get('video_id', '')
        thumbnail_url = video_details.get('thumbnail_url', f'https://img.youtube.com/vi/{video_id}/maxresdefault.jpg')
        
        try:
            st.image(thumbnail_url, width=200)  # Changed from use_container_width
        except:
            st.info("🎬 Thumbnail not available")
    
    with col2:
        st.markdown(f"**Title:** {video_details.get('title', 'Unknown')}")
        st.markdown(f"**Channel:** {video_details.get('channel_name', 'Unknown')}")
        st.markdown(f"**Published:** {video_details.get('published_at', 'N/A')[:10] if video_details.get('published_at') else 'N/A'}")
        
        # Additional metrics
        if 'view_count' in video_details:
            st.markdown(f"**Views:** {video_details['view_count']:,}")
        if 'like_count' in video_details:
            st.markdown(f"**Likes:** {video_details['like_count']:,}")

def render_key_metrics(analysis, processing_time):
    """Display key metrics"""
    st.subheader("🎯 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    viral_pred = analysis.get('viral_prediction', {})
    viral_prob = viral_pred.get('viral_probability', 0)
    prediction_class = viral_pred.get('prediction_class', 'unknown')
    
    with col1:
        st.metric(
            "Viral Probability",
            f"{viral_prob:.1%}",
            delta=f"{prediction_class}",
            help="Predicted viral potential"
        )
    
    with col2:
        sentiment_data = analysis.get('sentiment_analysis', {})
        sentiment_score = sentiment_data.get('average', 0)
        sentiment_label = "Positive" if sentiment_score > 0.3 else "Neutral" if sentiment_score > -0.3 else "Negative"
        st.metric(
            "Sentiment Score",
            f"{sentiment_score:.2f}",
            delta=sentiment_label,
            help="Average comment sentiment (-1 to 1)"
        )
    
    with col3:
        comment_stats = analysis.get('comment_stats', {})
        st.metric(
            "Comments Analyzed",
            comment_stats.get('clean_comments', 0),
            delta=f"-{comment_stats.get('spam_filtered', 0)} spam",
            help="Clean comments after spam filtering"
        )
    
    with col4:
        st.metric(
            "Processing Time",
            f"{processing_time:.2f}s",
            help="Total analysis time"
        )

def render_sentiment_analysis(sentiment_data):
    """Display sentiment analysis"""
    st.subheader("😊 Sentiment Analysis")
    
    if not sentiment_data:
        st.warning("⚠️ Sentiment data not available")
        return
    
    avg_sentiment = sentiment_data.get('average', 0)
    
    # Sentiment gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_sentiment,
        domain={'x': [0, 1], 'y': [0, 1]},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.3], 'color': "lightcoral"},
                {'range': [-0.3, 0.3], 'color': "lightyellow"},
                {'range': [0.3, 1], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': avg_sentiment
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)  # This is fine for plotly
    
    st.markdown(f"**Trend:** {sentiment_data.get('trend', 'N/A')}")
    
    # Distribution
    distribution = sentiment_data.get('sentiment_distribution', {})
    if distribution:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Positive", f"{distribution.get('positive', 0):.1%}")
        with col2:
            st.metric("Neutral", f"{distribution.get('neutral', 0):.1%}")
        with col3:
            st.metric("Negative", f"{distribution.get('negative', 0):.1%}")

def render_comment_stats(comment_stats):
    """Display comment statistics"""
    st.subheader("💬 Comment Statistics")
    
    if not comment_stats:
        st.warning("⚠️ Comment statistics not available")
        return
    
    stats_data = {
        'Metric': ['Total Fetched', 'Spam Filtered', 'Clean Comments'],
        'Count': [
            comment_stats.get('total_comments', 0),
            comment_stats.get('spam_filtered', 0),
            comment_stats.get('clean_comments', 0)
        ]
    }
    
    fig = px.bar(
        stats_data,
        x='Metric',
        y='Count',
        color='Metric',
        text='Count'
    )
    fig.update_layout(showlegend=False, height=250, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)  # This is fine for plotly
    
    # Spam ratio
    total = comment_stats.get('total_comments', 0)
    spam = comment_stats.get('spam_filtered', 0)
    if total > 0:
        spam_ratio = (spam / total) * 100
        st.progress(spam_ratio / 100)
        st.caption(f"Spam Ratio: {spam_ratio:.1f}%")

def render_viral_prediction(viral_data):
    """Display viral prediction"""
    st.subheader("🚀 Viral Prediction")
    
    if not viral_data:
        st.warning("⚠️ Viral prediction data not available")
        return
    
    viral_prob = viral_data.get('viral_probability', 0)
    
    # Progress bar
    st.progress(viral_prob)
    
    st.markdown(f"**Category:** {viral_data.get('prediction_class', 'N/A')}")
    st.markdown(f"**Confidence:** {viral_prob:.1%}")
    st.markdown(f"**Method:** {viral_data.get('method', 'N/A')}")
    
    # Reasoning
    with st.expander("🧠 AI Reasoning"):
        reasoning = viral_data.get('llm_reasoning', viral_data.get('reasoning', 'No detailed reasoning available'))
        st.write(reasoning)

def render_ai_recommendations(ai_advice):
    """Display AI recommendations"""
    st.subheader("💡 AI Recommendations")
    
    if not ai_advice:
        st.warning("⚠️ AI recommendations not available")
        return
    
    recommendations = ai_advice.get('structured_recommendations', [])
    
    if not recommendations:
        st.info("No recommendations generated")
        return
    
    for i, rec in enumerate(recommendations[:5], 1):  # Limit to 5 recommendations
        urgency_color = {
            'high': '🔴',
            'medium': '🟡',
            'low': '🟢'
        }.get(rec.get('urgency', 'low').lower(), '🔵')
        
        with st.container():
            st.markdown(f"{urgency_color} **{rec.get('action', 'Action')}**")
            st.caption(rec.get('description', 'No description'))
            
            if i < len(recommendations):
                st.divider()
