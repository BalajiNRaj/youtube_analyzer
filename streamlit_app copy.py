"""
Streamlit UI for YouTube Comments Analytics
Interactive web interface for analyzing YouTube videos and comments
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import List, Dict, Any
import time
import json
from collections import Counter
import os

# Set tokenizers parallelism to false to avoid fork warnings in Streamlit
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import our modules
from main import YouTubeAnalyzer
from src.utils.helpers import VideoIDExtractor, format_number

# AI/RAG imports
try:
    from langchain_anthropic import ChatAnthropic
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.chains import ConversationalRetrievalChain
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="YouTube Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean layout
st.markdown("""
<style>
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #e6f3ff;
    }
    
    /* Sidebar styling */
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e6f3ff;
    }
    
    /* Collection item styling */
    .collection-item {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        border-left: 4px solid #1f77b4;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .collection-item:hover {
        background: #e9ecef;
        transform: translateX(2px);
    }
    
    .collection-item.active {
        background: #d4edda;
        border-left-color: #28a745;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding: 0px 16px;
        background: white;
        border-radius: 6px;
        border: 1px solid #dee2e6;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: #1f77b4;
        color: white;
        border: 1px solid #1f77b4;
    }
    
    /* Content cards */
    .content-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    /* Video info display */
    .video-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Comment cards */
    .comment-card {
        background: white;
        padding: 1.2rem;
        border-radius: 6px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    
    /* Status indicators */
    .status-ready { color: #28a745; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        border: none;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'video_info' not in st.session_state:
        st.session_state.video_info = None
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'collections' not in st.session_state:
        st.session_state.collections = []
    if 'selected_collection' not in st.session_state:
        st.session_state.selected_collection = None
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "📊 Video Analysis"
    if 'show_new_video_form' not in st.session_state:
        st.session_state.show_new_video_form = False
    
    # AI Chat session state
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'ai_chain' not in st.session_state:
        st.session_state.ai_chain = None
    if 'ai_memory' not in st.session_state:
        st.session_state.ai_memory = None
    if 'anthropic_api_key' not in st.session_state:
        st.session_state.anthropic_api_key = ""
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "claude-3-7-sonnet-20250219"
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False

def initialize_analyzer():
    """Initialize the YouTube analyzer"""
    if not st.session_state.analyzer:
        with st.spinner("🔄 Initializing YouTube Analytics System..."):
            try:
                st.session_state.analyzer = YouTubeAnalyzer()
                if st.session_state.analyzer.authenticate():
                    # Load collections on startup
                    st.session_state.collections = st.session_state.analyzer.list_stored_videos()
                    return True
                else:
                    st.error("❌ Authentication failed. Please check credentials.")
                    return False
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
                return False
    return True

def load_collection(collection):
    """Load a specific collection's data"""
    if st.session_state.analyzer:
        try:
            # Load the analysis results for this video
            video_id = collection['video_id']
            results = st.session_state.analyzer.get_video_analysis(video_id)
            if results:
                st.session_state.analysis_results = results
                st.session_state.selected_collection = collection
                st.session_state.show_new_video_form = False
                return True
        except Exception as e:
            st.error(f"Error loading collection: {str(e)}")
    return False

def analyze_new_video(video_url, max_comments, export_format, reset_collection):
    """Analyze a new YouTube video"""
    try:
        # Extract video ID
        extractor = VideoIDExtractor()
        video_id = extractor.extract_video_id(video_url)
        
        if not video_id:
            st.error("❌ Invalid YouTube URL or video ID")
            return False
        
        # Show progress
        progress_container = st.container()
        with progress_container:
            st.info(f"🎯 Analyzing video: {video_id}")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Reset collection if requested
            if reset_collection:
                status_text.text("🔄 Resetting existing collection...")
                progress_bar.progress(10)
                st.session_state.analyzer.reset_collection(video_id)
            
            # Step 2: Extract comments
            status_text.text("📥 Extracting comments...")
            progress_bar.progress(25)
            
            results = st.session_state.analyzer.analyze_video(
                video_url=video_url,
                max_comments=max_comments,
                export_format=export_format,
                reset_collection=reset_collection
            )
            
            progress_bar.progress(75)
            status_text.text("🤖 Processing with AI...")
            
            if results:
                progress_bar.progress(90)
                status_text.text("💾 Saving results...")
                
                # Export results
                export_path = st.session_state.analyzer.export_results(
                    video_id, 
                    format=export_format
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Update session state
                st.session_state.analysis_results = results
                st.session_state.collections = st.session_state.analyzer.list_stored_videos()
                
                # Find and select the current collection
                for collection in st.session_state.collections:
                    if collection['video_id'] == video_id:
                        st.session_state.selected_collection = collection
                        break
                
                st.success(f"✅ Analysis complete! Results exported to: {export_path}")
                st.session_state.show_new_video_form = False
                return True
            else:
                st.error("❌ Analysis failed")
                return False
                
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        return False

def search_comments(query, video_id, max_results, filters=None):
    """Search comments using AI"""
    try:
        if st.session_state.analyzer:
            with st.spinner("🔍 Searching comments..."):
                results = st.session_state.analyzer.search_comments(
                    query=query,
                    video_id=video_id,
                    max_results=max_results,
                    filters=filters
                )
                st.session_state.search_results = results
                return results
        return None
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return None

def initialize_rag_system(anthropic_api_key, model_name):
    """Initialize RAG system with Anthropic Claude"""
    if not RAG_AVAILABLE:
        st.error("❌ Required packages not installed. Please install langchain, langchain-anthropic, and langchain-community.")
        return False
    
    try:
        # Set environment variable for Anthropic API key
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Get current analysis data for RAG
        documents = []
        if st.session_state.selected_collection and st.session_state.analysis_results:
            results = st.session_state.analysis_results
            processed_comments = results.get('processed_comments', [])
            
            # Create documents from comments
            for i, comment in enumerate(processed_comments[:200]):  # Limit to 200 comments for efficiency
                text = comment.get('text_display', comment.get('text', ''))
                if text and len(text.strip()) > 10:
                    metadata = {
                        'author': comment.get('author_display_name', 'Unknown'),
                        'likes': comment.get('like_count', 0),
                        'comment_id': i,
                        'video_id': st.session_state.selected_collection['video_id']
                    }
                    doc = Document(page_content=text, metadata=metadata)
                    documents.append(doc)
            
            # Add AI insights as documents
            if 'ai_ready_data' in results and 'insights' in results['ai_ready_data']:
                insights = results['ai_ready_data']['insights']
                for insight_type, insight_data in insights.items():
                    if isinstance(insight_data, dict) and 'summary' in insight_data:
                        doc = Document(
                            page_content=f"{insight_type}: {insight_data['summary']}",
                            metadata={'type': 'insight', 'insight_type': insight_type}
                        )
                        documents.append(doc)
        
        if not documents:
            # Create a default document if no data available
            documents = [Document(
                page_content="No specific video data loaded. I can help with general YouTube analytics questions.",
                metadata={'type': 'default'}
            )]
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Split documents
        split_documents = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            split_documents,
            embeddings,
            persist_directory=None  # In-memory for this session
        )
        
        # Initialize Claude
        llm = ChatAnthropic(
            model=model_name,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Create prompt template
        prompt_template = """You are an expert YouTube analytics AI assistant. You have access to comment data and insights from a specific YouTube video. 
        Use the following context to answer questions about the video's comments, audience engagement, sentiment, and performance.

        Context: {context}

        Chat History: {chat_history}

        Human: {question}

        Please provide a helpful, accurate, and insightful response based on the comment data and analytics. If you don't have specific information about something, be honest about it but offer general YouTube analytics insights when relevant.

        Assistant:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "chat_history"]
        )
        
        # Initialize memory
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5  # Remember last 5 exchanges
        )
        
        # Create conversational retrieval chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            verbose=True
        )
        
        # Store in session state
        st.session_state.ai_chain = chain
        st.session_state.ai_memory = memory
        st.session_state.rag_initialized = True
        
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {str(e)}")
        return False

def chat_with_ai(message):
    """Send a message to the AI and get response"""
    if not st.session_state.rag_initialized or not st.session_state.ai_chain:
        return "❌ AI system not initialized. Please set up your API key and model first."
    
    try:
        with st.spinner("🤖 AI is thinking..."):
            response = st.session_state.ai_chain({"question": message})
            return response["answer"]
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg and "model" in error_msg:
            return f"❌ Model not found. Please check that you're using a valid Claude model name. Current model: {st.session_state.selected_model}"
        elif "401" in error_msg or "authentication" in error_msg.lower():
            return "❌ Authentication failed. Please check your Anthropic API key."
        elif "rate_limit" in error_msg.lower() or "429" in error_msg:
            return "❌ Rate limit exceeded. Please wait a moment and try again."
        else:
            return f"❌ Error getting AI response: {error_msg}"

# Initialize app
init_session_state()

# Header
st.markdown('<h1 class="main-header">🎥 YouTube Analytics Dashboard</h1>', unsafe_allow_html=True)

# Initialize analyzer
if not st.session_state.analyzer:
    initialize_analyzer()

# Main layout: Sidebar + Content
col_sidebar, col_main = st.columns([1, 3])

# === LEFT SIDEBAR ===
with col_sidebar:
    st.markdown('<div class="sidebar-title">📚 Collections</div>', unsafe_allow_html=True)
    
    # "Analyze New Video" button
    if st.button("🎯 Analyze New Video", type="secondary" if not st.session_state.show_new_video_form else "primary", key="new_video_btn"):
        st.session_state.show_new_video_form = True
        st.session_state.current_tab = "📊 Video Analysis"
        st.session_state.selected_collection = None
    
    st.markdown("---")
    
    # Collections list
    if st.session_state.collections:
        st.markdown("### 📺 Analyzed Videos")
        
        for i, collection in enumerate(st.session_state.collections):
            # Create collection display
            is_selected = (st.session_state.selected_collection and 
                          st.session_state.selected_collection['video_id'] == collection['video_id'])
            
            # Collection button
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button(
                    f"{collection['video_id']}",
                    type="secondary" if not is_selected else "primary",
                    key=f"collection_{collection['video_id']}",
                    help=f"Video: {collection['video_id']}\nComments: {collection['total_comments']}"
                ):
                    load_collection(collection)
                    st.session_state.show_new_video_form = False
            
            with col2:
                st.caption(f"{collection['total_comments']}")
    
        # Refresh button
        st.markdown("---")
        if st.button("🔄 Refresh", key="refresh_collections"):
            if st.session_state.analyzer:
                st.session_state.collections = st.session_state.analyzer.list_stored_videos()
                st.rerun()
    else:
        st.info("📝 No videos analyzed yet.\n\nClick 'Analyze New Video' to get started!")
    
    
    # System status
    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    if st.session_state.analyzer:
        st.markdown('<span class="status-ready">✅ System Ready</span>', unsafe_allow_html=True)
        st.caption(f"Collections: {len(st.session_state.collections)}")
    else:
        st.markdown('<span class="status-error">❌ System Error</span>', unsafe_allow_html=True)

# === MAIN CONTENT AREA ===
with col_main:
    # Show tabs
    if st.session_state.show_new_video_form or not st.session_state.collections:
        # Show only Video Analysis tab for new video
        st.markdown("## Analyze New YouTube Video")
        
        with st.form("video_analysis_form"):
            
            # Video URL input
            video_url = st.text_input(
                "YouTube Video URL or ID",
                placeholder="https://youtu.be/videourl or video_id",
                help="Enter a YouTube video URL or video ID to analyze"
            )
            
            # Options in columns
            col1, col2 = st.columns(2)
            
            with col1:
                max_comments = st.slider(
                    "Maximum Comments",
                    min_value=50,
                    max_value=2000,
                    value=500,
                    step=50,
                    help="Higher values provide more data but take longer to process"
                )
            
            with col2:
                export_format = st.selectbox(
                    "Export Format",
                    ["json", "csv"],
                    help="Choose format for exporting results"
                )
            
            # Advanced options
            with st.expander("Advanced Options"):
                reset_collection = st.checkbox(
                    "Reset Existing Collection",
                    value=False,
                    help="Clear existing data for this video if it exists"
                )
            
            # Submit button
            submitted = st.form_submit_button("🚀 Start Analysis", type="primary")
            
            if submitted and video_url and st.session_state.analyzer:
                analyze_new_video(video_url, max_comments, export_format, reset_collection)
            elif submitted and not video_url:
                st.error("❌ Please enter a YouTube video URL or ID")
            elif submitted and not st.session_state.analyzer:
                st.error("❌ Analyzer not initialized. Please refresh the page.")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Show all tabs when a collection is available
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Video Analysis", 
            "🔍 Comment Search", 
            "📈 Analytics", 
            "💬 Comment Explorer",
            "🤖 AI Query Engine"
        ])
        
        # === TAB 1: Video Analysis ===
        with tab1:
            st.markdown("## 🎯 Comprehensive Video Analysis")
            
            if st.session_state.selected_collection and st.session_state.analysis_results:
                results = st.session_state.analysis_results
                collection = st.session_state.selected_collection
                processed_comments = results.get('processed_comments', [])
                
                if processed_comments:
                    df_comments = pd.DataFrame(processed_comments)
                    
                    # Normalize column names
                    if 'likes' in df_comments.columns:
                        df_comments['like_count'] = df_comments['likes']
                    elif 'like_count' not in df_comments.columns:
                        df_comments['like_count'] = 0
                    
                    if 'author' in df_comments.columns:
                        df_comments['author_display_name'] = df_comments['author']
                    elif 'author_display_name' not in df_comments.columns:
                        df_comments['author_display_name'] = 'Unknown'
                    
                    if 'text' in df_comments.columns:
                        df_comments['text_display'] = df_comments['text']
                    elif 'original_text' in df_comments.columns:
                        df_comments['text_display'] = df_comments['original_text']
                    elif 'text_display' not in df_comments.columns:
                        df_comments['text_display'] = 'No text available'
                        
                    # Calculate derived metrics
                    df_comments['word_count'] = df_comments['text_display'].astype(str).apply(lambda x: len(x.split()))
                    df_comments['char_count'] = df_comments['text_display'].astype(str).apply(len)
                    total_engagement = df_comments['like_count'].sum()
                    avg_engagement = df_comments['like_count'].mean()
                    unique_authors = df_comments['author_display_name'].nunique()
                    
                    # Key Performance Indicators
                    st.markdown("### 📊 Key Performance Indicators")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric(
                            "💬 Total Comments", 
                            format_number(len(df_comments)),
                            help="Total number of comments analyzed"
                        )
                    with col2:
                        st.metric(
                            "❤️ Total Engagement", 
                            format_number(int(total_engagement)),
                            help="Total likes across all comments"
                        )
                    with col3:
                        st.metric(
                            "⚡ Avg Engagement", 
                            f"{avg_engagement:.1f}",
                            help="Average likes per comment"
                        )
                    with col4:
                        st.metric(
                            "👥 Unique Commenters", 
                            format_number(unique_authors),
                            help="Number of unique users who commented"
                        )
                    with col5:
                        retention_rate = (unique_authors / len(df_comments)) * 100 if len(df_comments) > 0 else 0
                        st.metric(
                            "🔄 Community Retention", 
                            f"{retention_rate:.1f}%",
                            help="Percentage of unique commenters vs total comments"
                        )
                    
                    # Engagement Analysis
                    st.markdown("### 🚀 Engagement Deep Dive")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Top performing comments
                        st.markdown("#### 🏆 Top Performing Comments")
                        top_comments = df_comments.nlargest(5, 'like_count')
                        
                        for idx, comment in top_comments.iterrows():
                            with st.container():
                                st.markdown(f"""
                                <div style='background: linear-gradient(90deg, #ff6b6b, #4ecdc4); padding: 1rem; border-radius: 8px; margin: 0.5rem 0; color: white;'>
                                    <strong>👤 {comment['author_display_name']}</strong> • <strong>👍 {comment['like_count']} likes</strong><br>
                                    <em>"{comment['text_display'][:100]}{'...' if len(comment['text_display']) > 100 else ''}"</em>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    with col2:
                        # Engagement distribution
                        st.markdown("#### 📈 Engagement Distribution")
                        if df_comments['like_count'].sum() > 0:
                            # Create engagement bins
                            df_comments['engagement_tier'] = pd.cut(
                                df_comments['like_count'], 
                                bins=[-1, 0, 5, 20, 100, float('inf')],
                                labels=['No Likes', '1-5 Likes', '6-20 Likes', '21-100 Likes', '100+ Likes']
                            )
                            
                            engagement_dist = df_comments['engagement_tier'].value_counts()
                            
                            fig_engagement = px.pie(
                                values=engagement_dist.values,
                                names=engagement_dist.index,
                                title="Comment Engagement Tiers",
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            fig_engagement.update_layout(height=350)
                            st.plotly_chart(fig_engagement, use_container_width=True)
                        else:
                            st.info("No engagement data available for visualization")
                    
                    # Content Quality Analysis
                    st.markdown("### 📝 Content Quality Insights")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_words = df_comments['word_count'].mean()
                        st.metric("📖 Avg Words/Comment", f"{avg_words:.1f}")
                        
                        # Word count distribution
                        fig_words = px.histogram(
                            df_comments,
                            x='word_count',
                            nbins=20,
                            title="Word Count Distribution",
                            labels={'word_count': 'Words per Comment', 'count': 'Number of Comments'}
                        )
                        fig_words.update_layout(height=300)
                        st.plotly_chart(fig_words, use_container_width=True)
                    
                    with col2:
                        # Long-form vs short-form analysis
                        long_form = len(df_comments[df_comments['word_count'] >= 20])
                        short_form = len(df_comments[df_comments['word_count'] < 20])
                        
                        st.metric("� Long-form Comments", format_number(long_form))
                        st.metric("💬 Short-form Comments", format_number(short_form))
                        
                        # Create comparison chart
                        content_types = pd.DataFrame({
                            'Type': ['Long-form (20+ words)', 'Short-form (<20 words)'],
                            'Count': [long_form, short_form],
                            'Avg Engagement': [
                                df_comments[df_comments['word_count'] >= 20]['like_count'].mean() if long_form > 0 else 0,
                                df_comments[df_comments['word_count'] < 20]['like_count'].mean() if short_form > 0 else 0
                            ]
                        })
                        
                        fig_content = px.bar(
                            content_types,
                            x='Type',
                            y='Avg Engagement',
                            title="Engagement by Content Length",
                            color='Avg Engagement',
                            color_continuous_scale='Viridis'
                        )
                        fig_content.update_layout(height=300)
                        st.plotly_chart(fig_content, use_container_width=True)
                    
                    with col3:
                        # Most active commenters
                        st.markdown("#### 🌟 Most Active Community Members")
                        active_users = df_comments['author_display_name'].value_counts().head(5)
                        
                        for user, count in active_users.items():
                            user_engagement = df_comments[df_comments['author_display_name'] == user]['like_count'].sum()
                            st.markdown(f"""
                            <div style='background: #f0f2f6; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0;'>
                                <strong>{user}</strong><br>
                                📝 {count} comments • ❤️ {user_engagement} total likes
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # AI Insights Section
                    if 'ai_ready_data' in results:
                        ai_data = results['ai_ready_data']
                        if 'insights' in ai_data:
                            st.markdown("### � AI-Generated Insights")
                            insights = ai_data['insights']
                            
                            # Create insight cards
                            insight_cols = st.columns(2)
                            col_idx = 0
                            
                            for insight_type, insight_data in insights.items():
                                if isinstance(insight_data, dict) and 'summary' in insight_data:
                                    with insight_cols[col_idx % 2]:
                                        st.markdown(f"""
                                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                                    color: white; padding: 1.5rem; border-radius: 8px; margin: 0.5rem 0;'>
                                            <h4 style='color: white; margin-top: 0;'>
                                                {insight_type.replace('_', ' ').title()} 📊
                                            </h4>
                                            <p style='margin-bottom: 0;'>{insight_data['summary']}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    col_idx += 1
                    
                    # Community Health Score
                    st.markdown("### 🌡️ Community Health Score")
                    
                    # Calculate health metrics
                    engagement_score = min(100, (avg_engagement / 10) * 100) if avg_engagement > 0 else 0
                    diversity_score = min(100, (unique_authors / len(df_comments)) * 100) if len(df_comments) > 0 else 0
                    content_quality_score = min(100, (avg_words / 15) * 100) if avg_words > 0 else 0
                    
                    overall_health = (engagement_score + diversity_score + content_quality_score) / 3
                    
                    health_col1, health_col2, health_col3, health_col4 = st.columns(4)
                    
                    with health_col1:
                        st.metric(
                            "💝 Engagement Health", 
                            f"{engagement_score:.0f}/100",
                            help="Based on average likes per comment"
                        )
                    
                    with health_col2:
                        st.metric(
                            "🌈 Community Diversity", 
                            f"{diversity_score:.0f}/100",
                            help="Based on unique commenters ratio"
                        )
                    
                    with health_col3:
                        st.metric(
                            "✨ Content Quality", 
                            f"{content_quality_score:.0f}/100",
                            help="Based on average comment length"
                        )
                    
                    with health_col4:
                        health_color = "🟢" if overall_health >= 70 else "🟡" if overall_health >= 40 else "�"
                        st.metric(
                            "🏥 Overall Health", 
                            f"{health_color} {overall_health:.0f}/100",
                            help="Combined health score"
                        )
                
                else:
                    st.info("📊 No processed comments available for analysis")
            
            else:
                st.info("�📊 Select a video collection from the sidebar to view comprehensive analysis, or click 'Analyze New Video' to start!")
        
        # === TAB 2: Comment Search ===
        with tab2:
            st.markdown("## 🔍 AI-Powered Comment Intelligence")
            
            if st.session_state.selected_collection:
                current_video_id = st.session_state.selected_collection['video_id']
                total_comments = st.session_state.selected_collection['total_comments']
                st.info(f"🎯 Searching in: **{current_video_id}** ({format_number(total_comments)} comments)")
                
                # Advanced Search Interface
                search_col1, search_col2 = st.columns([2, 1])
                
                with search_col1:
                    # Search form
                    with st.form("advanced_search_form"):
                        # Main search query
                        search_query = st.text_input(
                            "🔍 AI Search Query",
                            placeholder="e.g., 'users asking for tutorials', 'negative feedback about audio', 'suggestions for improvement'",
                            help="Use natural language to find specific types of comments. AI understands context and sentiment."
                        )
                        
                        # Search options
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            max_results = st.selectbox("Results Count", [5, 10, 20, 30, 50, 100], index=2)
                            search_type = st.selectbox("Search Type", ["Semantic (AI)", "Keyword"], help="Semantic search understands meaning, Keyword searches for exact matches")
                        
                        with col2:
                            # Content filters
                            content_filter = st.selectbox("Content Type", ["All Comments", "Questions Only", "Suggestions Only", "Complaints Only", "Praise Only"])
                            min_engagement = st.slider("Min Likes", 0, 100, 0, help="Filter by minimum number of likes")
                        
                        with col3:
                            # Author filters
                            author_type = st.selectbox("Author Type", ["All Users", "New Commenters", "Repeat Commenters", "High Engagement Users"])
                            language_filter = st.selectbox("Language", ["All Languages", "English", "Spanish", "French", "German", "Other"])
                        
                        # Advanced filters in expander
                        with st.expander("🎛️ Advanced Creator Filters"):
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                sentiment_filter = st.selectbox("Sentiment Analysis", ["All Sentiments", "Very Positive", "Positive", "Neutral", "Negative", "Very Negative"])
                                length_filter = st.selectbox("Comment Length", ["All Lengths", "Short (1-10 words)", "Medium (11-30 words)", "Long (30+ words)"])
                            
                            with col_b:
                                time_filter = st.selectbox("Comment Age", ["All Time", "Last 24h", "Last Week", "Last Month", "Older"])
                                include_replies = st.checkbox("Include Replies", value=True, help="Include comment replies in search results")
                        
                        # Search button
                        search_submitted = st.form_submit_button("� Intelligent Search", type="primary")
                        
                        if search_submitted and search_query:
                            filters = {
                                'min_likes': min_engagement if min_engagement > 0 else None,
                                'sentiment': sentiment_filter if sentiment_filter != "All Sentiments" else None,
                                'content_type': content_filter if content_filter != "All Comments" else None,
                                'author_type': author_type if author_type != "All Users" else None,
                                'length': length_filter if length_filter != "All Lengths" else None,
                                'include_replies': include_replies
                            }
                            search_comments(search_query, current_video_id, max_results, filters)
                
                with search_col2:
                    # Quick search presets for YouTubers
                    st.markdown("### 🚀 Quick Creator Searches")
                    
                    preset_searches = [
                        ("💡 Feature Requests", "feature request suggestion improve add"),
                        ("🐛 Bug Reports", "bug error problem issue doesn't work"),
                        ("❓ Common Questions", "how to tutorial guide explain help"),
                        ("💝 Positive Feedback", "love great amazing awesome fantastic"),
                        ("😞 Negative Feedback", "bad terrible awful hate disappointed"),
                        ("🔥 Viral Comments", ""), # Will search by high engagement
                        ("🆕 New Subscriber Comments", "subscribed new subscriber just found"),
                        ("🎯 Target Audience Insights", "age demographic location country"),
                    ]
                    
                    for label, query in preset_searches:
                        if st.button(label, key=f"preset_{label}", help=f"Search for: {query}" if query else "High engagement comments"):
                            if label == "🔥 Viral Comments":
                                # Special handling for viral comments
                                st.session_state.search_results = {
                                    'results': [],
                                    'query': 'High engagement comments',
                                    'preset': True
                                }
                            else:
                                search_comments(query, current_video_id, 10, {'preset': label})
                
                # Display search results with enhanced analytics
                if st.session_state.search_results and "results" in st.session_state.search_results:
                    results = st.session_state.search_results['results']
                    
                    if results:
                        # Search results analytics
                        st.markdown("---")
                        st.markdown(f"### � Search Results Analytics ({len(results)} comments found)")
                        
                        # Results overview
                        results_df = pd.DataFrame(results)
                        
                        if not results_df.empty:
                            # Analytics metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                total_likes = results_df.get('like_count', pd.Series([0])).sum()
                                st.metric("💝 Total Engagement", format_number(int(total_likes)))
                            
                            with col2:
                                avg_similarity = results_df.get('similarity', pd.Series([0])).mean() * 100 if 'similarity' in results_df.columns else 0
                                st.metric("🎯 Avg Relevance", f"{avg_similarity:.1f}%")
                            
                            with col3:
                                unique_authors = results_df.get('author_display_name', pd.Series(['Unknown'])).nunique()
                                st.metric("👥 Unique Authors", format_number(unique_authors))
                            
                            with col4:
                                if 'text_display' in results_df.columns:
                                    avg_length = results_df['text_display'].astype(str).apply(lambda x: len(x.split())).mean()
                                    st.metric("📝 Avg Words", f"{avg_length:.1f}")
                                else:
                                    st.metric("📝 Avg Words", "N/A")
                        
                        # Results display with enhanced information
                        st.markdown("### 💬 Search Results")
                        
                        for i, result in enumerate(results, 1):
                            with st.container():
                                # Create enhanced comment card
                                st.markdown(f"""
                                <div style='background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%); 
                                            padding: 1.5rem; border-radius: 12px; margin: 1rem 0; 
                                            border-left: 4px solid #007bff; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                                """, unsafe_allow_html=True)
                                
                                # Comment header with enhanced info
                                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                                with col1:
                                    author = result.get('author_display_name', 'Unknown')
                                    st.markdown(f"**👤 {author}**")
                                with col2:
                                    likes = result.get('like_count', 0)
                                    st.markdown(f"👍 **{format_number(likes)}** likes")
                                with col3:
                                    if 'similarity' in result:
                                        similarity = result.get('similarity', 0)
                                        st.markdown(f"🎯 **{similarity:.1%}** match")
                                with col4:
                                    # Engagement tier
                                    if likes >= 100:
                                        st.markdown("🔥 **Viral**")
                                    elif likes >= 20:
                                        st.markdown("⭐ **Popular**")
                                    elif likes >= 5:
                                        st.markdown("👍 **Engaged**")
                                    else:
                                        st.markdown("💬 **Standard**")
                                
                                # Comment text with highlighting
                                comment_text = result.get('text_display', 'No text available')
                                st.markdown(f"### � Comment Content")
                                st.markdown(f"> {comment_text}")
                                
                                # Additional metadata
                                metadata_col1, metadata_col2, metadata_col3 = st.columns(3)
                                
                                with metadata_col1:
                                    published_at = result.get('published_at', '')
                                    if published_at:
                                        st.caption(f"📅 Published: {published_at}")
                                
                                with metadata_col2:
                                    word_count = len(comment_text.split()) if comment_text else 0
                                    st.caption(f"📝 {word_count} words")
                                
                                with metadata_col3:
                                    if 'sentiment_score' in result:
                                        sentiment = result['sentiment_score']
                                        sentiment_emoji = "😊" if sentiment > 0.1 else "😐" if sentiment > -0.1 else "😞"
                                        st.caption(f"{sentiment_emoji} Sentiment: {sentiment:.2f}")
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Export search results
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("📊 Generate Search Report", type="secondary"):
                                st.info("📊 Search report generation feature coming soon!")
                        
                        with col2:
                            if st.button("💾 Export Results", type="secondary"):
                                st.info("💾 Export functionality coming soon!")
                    
                    else:
                        st.warning("🔍 No comments found matching your search criteria. Try adjusting your search terms or filters.")
                
                # Search tips for content creators
                with st.expander("💡 Pro Search Tips for Content Creators"):
                    st.markdown("""
                    ### 🎯 Advanced Search Strategies:
                    
                    **For Content Planning:**
                    - *"what video next"* - Find requests for future content
                    - *"tutorial on"* - Discover what tutorials viewers want
                    - *"please make video about"* - Direct content suggestions
                    
                    **For Community Management:**
                    - *"confused about"* - Find areas needing clarification
                    - *"doesn't work"* - Identify common issues
                    - *"thank you"* - Find appreciation comments
                    
                    **For Audience Insights:**
                    - *"first time watching"* - New audience feedback
                    - *"been following since"* - Long-time subscriber comments
                    - *"from [country]"* - Geographic audience insights
                    
                    **For Engagement Analysis:**
                    - Use high minimum likes to find viral comments
                    - Search for "question" to find Q&A opportunities
                    - Look for "disagree" to find discussion starters
                    """)
            
            else:
                st.info("🎯 Select a video collection from the sidebar to start intelligent comment searching!")
        
        # === TAB 3: Analytics ===
        with tab3:
            st.markdown("## 📈 Creator Analytics Dashboard")
            
            if st.session_state.selected_collection and st.session_state.analysis_results:
                results = st.session_state.analysis_results
                processed_comments = results.get('processed_comments', [])
                
                if processed_comments:
                    # Enhanced data processing
                    df_comments = pd.DataFrame(processed_comments)
                    
                    # Normalize column names to handle different data sources
                    if 'likes' in df_comments.columns and 'like_count' not in df_comments.columns:
                        df_comments['like_count'] = df_comments['likes']
                    elif 'like_count' not in df_comments.columns:
                        df_comments['like_count'] = 0
                    
                    if 'author' in df_comments.columns and 'author_display_name' not in df_comments.columns:
                        df_comments['author_display_name'] = df_comments['author']
                    elif 'author_display_name' not in df_comments.columns:
                        df_comments['author_display_name'] = 'Unknown'
                    
                    if 'text' in df_comments.columns and 'text_display' not in df_comments.columns:
                        df_comments['text_display'] = df_comments['text']
                    elif 'original_text' in df_comments.columns and 'text_display' not in df_comments.columns:
                        df_comments['text_display'] = df_comments['original_text']
                    elif 'text_display' not in df_comments.columns:
                        df_comments['text_display'] = 'No text available'
                    
                    if 'timestamp' in df_comments.columns and 'published_at' not in df_comments.columns:
                        df_comments['published_at'] = df_comments['timestamp']
                    elif 'published_at' not in df_comments.columns:
                        df_comments['published_at'] = ''
                    
                    # Enhanced metrics calculation
                    df_comments['word_count'] = df_comments['text_display'].astype(str).apply(lambda x: len(x.split()))
                    df_comments['char_count'] = df_comments['text_display'].astype(str).apply(len)
                    df_comments['has_question'] = df_comments['text_display'].astype(str).str.contains(r'\?', na=False)
                    df_comments['has_emoji'] = df_comments['text_display'].astype(str).str.contains(r'[😀-🿿]|[🎀-🏿]|[🚀-🛿]', na=False)
                    
                    # Creator-focused KPIs
                    st.markdown("### 🎯 Creator Performance Metrics")
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    
                    with col1:
                        total_engagement = df_comments['like_count'].sum()
                        st.metric("💝 Total Likes", format_number(int(total_engagement)))
                    
                    with col2:
                        engagement_rate = (total_engagement / len(df_comments)) if len(df_comments) > 0 else 0
                        st.metric("📊 Engagement Rate", f"{engagement_rate:.2f}")
                    
                    with col3:
                        unique_commenters = df_comments['author_display_name'].nunique()
                        st.metric("👥 Unique Viewers", format_number(unique_commenters))
                    
                    with col4:
                        questions_count = df_comments['has_question'].sum()
                        st.metric("❓ Questions Asked", format_number(questions_count))
                    
                    with col5:
                        emoji_usage = (df_comments['has_emoji'].sum() / len(df_comments) * 100) if len(df_comments) > 0 else 0
                        st.metric("😊 Emoji Usage", f"{emoji_usage:.1f}%")
                    
                    with col6:
                        avg_sentiment = df_comments['sentiment_score'].mean() if 'sentiment_score' in df_comments.columns else 0
                        sentiment_emoji = "🥳" if avg_sentiment > 0.3 else "😊" if avg_sentiment > 0 else "😐" if avg_sentiment > -0.3 else "😞"
                        st.metric("🌡️ Mood Score", f"{sentiment_emoji} {avg_sentiment:.2f}" if 'sentiment_score' in df_comments.columns else "N/A")
                    
                    # Advanced Analytics Section
                    st.markdown("---")
                    st.markdown("### 📊 Deep Dive Analytics")
                    
                    # Create tabs for different analytics sections
                    analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab4 = st.tabs([
                        "🔥 Engagement Analysis", 
                        "👥 Audience Insights", 
                        "📝 Content Analysis", 
                        "⏰ Temporal Patterns"
                    ])
                    
                    with analytics_tab1:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Engagement distribution
                            st.markdown("#### 🚀 Engagement Tiers")
                            if df_comments['like_count'].sum() > 0:
                                # Create detailed engagement bins
                                engagement_bins = [-1, 0, 1, 5, 10, 25, 50, 100, float('inf')]
                                engagement_labels = ['No Likes', '1 Like', '2-5 Likes', '6-10 Likes', '11-25 Likes', '26-50 Likes', '51-100 Likes', '100+ Likes']
                                
                                df_comments['engagement_tier'] = pd.cut(df_comments['like_count'], bins=engagement_bins, labels=engagement_labels)
                                engagement_dist = df_comments['engagement_tier'].value_counts().sort_index()
                                
                                fig_engagement = px.bar(
                                    x=engagement_dist.index,
                                    y=engagement_dist.values,
                                    title="Comment Engagement Distribution",
                                    labels={'x': 'Engagement Tier', 'y': 'Number of Comments'},
                                    color=engagement_dist.values,
                                    color_continuous_scale='RdYlGn'
                                )
                                fig_engagement.update_layout(height=400, xaxis_tickangle=45)
                                st.plotly_chart(fig_engagement, use_container_width=True)
                                
                                # Viral comments showcase
                                viral_threshold = df_comments['like_count'].quantile(0.95) if len(df_comments) > 20 else 10
                                viral_comments = df_comments[df_comments['like_count'] >= viral_threshold].nlargest(3, 'like_count')
                                
                                if not viral_comments.empty:
                                    st.markdown("#### 🔥 Viral Comments")
                                    for idx, comment in viral_comments.iterrows():
                                        st.markdown(f"""
                                        <div style='background: linear-gradient(90deg, #ff9a8b, #fecfef); 
                                                    padding: 1rem; border-radius: 8px; margin: 0.5rem 0; color: #333;'>
                                            <strong>🔥 {comment['like_count']} likes</strong> | <strong>{comment['author_display_name']}</strong><br>
                                            <em>"{comment['text_display'][:150]}{'...' if len(comment['text_display']) > 150 else ''}"</em>
                                        </div>
                                        """, unsafe_allow_html=True)
                        
                        with col2:
                            # Top performers analysis
                            st.markdown("#### 🏆 Top Performing Content")
                            
                            if not df_comments.empty:
                                # Content performance by length
                                df_comments['length_category'] = pd.cut(
                                    df_comments['word_count'], 
                                    bins=[0, 5, 15, 30, float('inf')],
                                    labels=['Very Short (1-5)', 'Short (6-15)', 'Medium (16-30)', 'Long (30+)']
                                )
                                
                                performance_by_length = df_comments.groupby('length_category', observed=True).agg({
                                    'like_count': ['mean', 'sum', 'count']
                                }).round(2)
                                
                                performance_by_length.columns = ['Avg Likes', 'Total Likes', 'Comment Count']
                                performance_by_length = performance_by_length.reset_index()
                                
                                fig_performance = px.scatter(
                                    performance_by_length,
                                    x='Comment Count',
                                    y='Avg Likes',
                                    size='Total Likes',
                                    color='length_category',
                                    title="Performance by Comment Length",
                                    hover_data=['Total Likes']
                                )
                                fig_performance.update_layout(height=400)
                                st.plotly_chart(fig_performance, use_container_width=True)
                                
                                # Engagement insights
                                best_length = performance_by_length.loc[performance_by_length['Avg Likes'].idxmax(), 'length_category']
                                st.success(f"💡 **Insight**: '{best_length}' comments perform best on average!")
                    
                    with analytics_tab2:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Community composition
                            st.markdown("#### 👥 Community Composition")
                            
                            # Categorize users by activity
                            user_activity = df_comments['author_display_name'].value_counts()
                            
                            # Create user categories
                            power_users = user_activity[user_activity >= 3].index.tolist()  # 3+ comments
                            regular_users = user_activity[(user_activity >= 2) & (user_activity < 3)].index.tolist()  # 2 comments
                            casual_users = user_activity[user_activity == 1].index.tolist()  # 1 comment
                            
                            community_composition = pd.DataFrame({
                                'User Type': ['Power Users (3+ comments)', 'Regular Users (2 comments)', 'Casual Users (1 comment)'],
                                'Count': [len(power_users), len(regular_users), len(casual_users)],
                                'Percentage': [
                                    len(power_users) / len(user_activity) * 100,
                                    len(regular_users) / len(user_activity) * 100,
                                    len(casual_users) / len(user_activity) * 100
                                ]
                            })
                            
                            fig_community = px.pie(
                                community_composition,
                                values='Count',
                                names='User Type',
                                title="Community User Distribution",
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            fig_community.update_layout(height=350)
                            st.plotly_chart(fig_community, use_container_width=True)
                            
                            # Top contributors
                            st.markdown("#### 🌟 Top Community Contributors")
                            top_contributors = user_activity.head(5)
                            for user, count in top_contributors.items():
                                user_engagement = df_comments[df_comments['author_display_name'] == user]['like_count'].sum()
                                avg_engagement = df_comments[df_comments['author_display_name'] == user]['like_count'].mean()
                                
                                st.markdown(f"""
                                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                            color: white; padding: 1rem; border-radius: 8px; margin: 0.3rem 0;'>
                                    <strong>{user}</strong><br>
                                    📝 {count} comments • ❤️ {user_engagement} total likes • 📊 {avg_engagement:.1f} avg likes
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            # Engagement patterns by user type
                            st.markdown("#### 🎯 Engagement by User Type")
                            
                            # Calculate engagement metrics by user type
                            df_comments['user_type'] = df_comments['author_display_name'].apply(
                                lambda x: 'Power User' if x in power_users else 
                                         'Regular User' if x in regular_users else 'Casual User'
                            )
                            
                            engagement_by_type = df_comments.groupby('user_type', observed=True).agg({
                                'like_count': ['mean', 'sum', 'count'],
                                'word_count': 'mean'
                            }).round(2)
                            
                            engagement_by_type.columns = ['Avg Likes', 'Total Likes', 'Comment Count', 'Avg Words']
                            engagement_by_type = engagement_by_type.reset_index()
                            
                            fig_user_engagement = px.bar(
                                engagement_by_type,
                                x='user_type',
                                y='Avg Likes',
                                color='user_type',
                                title="Average Engagement by User Type",
                                color_discrete_sequence=['#ff6b6b', '#4ecdc4', '#45b7d1']
                            )
                            fig_user_engagement.update_layout(height=350)
                            st.plotly_chart(fig_user_engagement, use_container_width=True)
                            
                            # User insights
                            for _, row in engagement_by_type.iterrows():
                                st.metric(
                                    f"📊 {row['user_type']}",
                                    f"{row['Avg Likes']:.1f} avg likes",
                                    f"{row['Comment Count']} comments"
                                )
                    
                    with analytics_tab3:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Content themes analysis
                            st.markdown("#### 📝 Content Themes")
                            
                            # Word frequency analysis (simplified)
                            all_text = ' '.join(df_comments['text_display'].astype(str).str.lower())
                            words = all_text.split()
                            # Remove common stop words
                            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'a', 'an', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
                            filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
                            
                            word_freq = Counter(filtered_words).most_common(10)
                            
                            if word_freq:
                                words_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
                                
                                fig_words = px.bar(
                                    words_df,
                                    x='Word',
                                    y='Frequency',
                                    title="Most Common Words in Comments",
                                    color='Frequency',
                                    color_continuous_scale='Viridis'
                                )
                                fig_words.update_layout(height=400, xaxis_tickangle=45)
                                st.plotly_chart(fig_words, use_container_width=True)
                            
                            # Content insights
                            question_rate = (df_comments['has_question'].sum() / len(df_comments) * 100) if len(df_comments) > 0 else 0
                            emoji_rate = (df_comments['has_emoji'].sum() / len(df_comments) * 100) if len(df_comments) > 0 else 0
                            
                            st.markdown(f"""
                            #### 💡 Content Insights
                            - **{question_rate:.1f}%** of comments contain questions
                            - **{emoji_rate:.1f}%** of comments use emojis
                            - Average comment length: **{df_comments['word_count'].mean():.1f} words**
                            """)
                        
                        with col2:
                            # Sentiment analysis
                            st.markdown("#### 😊 Sentiment Analysis")
                            
                            if 'sentiment_score' in df_comments.columns:
                                # Create sentiment categories
                                df_comments['sentiment_category'] = pd.cut(
                                    df_comments['sentiment_score'],
                                    bins=[-1, -0.5, -0.1, 0.1, 0.5, 1],
                                    labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
                                )
                                
                                sentiment_dist = df_comments['sentiment_category'].value_counts()
                                
                                fig_sentiment = px.pie(
                                    values=sentiment_dist.values,
                                    names=sentiment_dist.index,
                                    title="Comment Sentiment Distribution",
                                    color_discrete_sequence=['#ff6b6b', '#ffa07a', '#d3d3d3', '#98fb98', '#32cd32']
                                )
                                fig_sentiment.update_layout(height=350)
                                st.plotly_chart(fig_sentiment, use_container_width=True)
                                
                                # Sentiment insights
                                positive_rate = len(df_comments[df_comments['sentiment_score'] > 0.1]) / len(df_comments) * 100
                                negative_rate = len(df_comments[df_comments['sentiment_score'] < -0.1]) / len(df_comments) * 100
                                
                                st.markdown(f"""
                                #### 🌡️ Audience Mood
                                - **{positive_rate:.1f}%** positive sentiment
                                - **{negative_rate:.1f}%** negative sentiment
                                - Overall mood: **{avg_sentiment:.2f}** {"🥳 Excellent!" if avg_sentiment > 0.3 else "😊 Good!" if avg_sentiment > 0 else "😐 Neutral" if avg_sentiment > -0.3 else "😞 Needs attention"}
                                """)
                            else:
                                st.info("Sentiment analysis data not available")
                    
                    with analytics_tab4:
                        # Temporal analysis would require proper timestamp data
                        st.markdown("#### ⏰ Temporal Patterns")
                        
                        if 'published_at' in df_comments.columns and df_comments['published_at'].notna().sum() > 0:
                            # Try to parse timestamps
                            try:
                                df_comments['parsed_time'] = pd.to_datetime(df_comments['published_at'], errors='coerce')
                                df_comments_with_time = df_comments[df_comments['parsed_time'].notna()]
                                
                                if len(df_comments_with_time) > 10:
                                    # Extract time components
                                    df_comments_with_time['hour'] = df_comments_with_time['parsed_time'].dt.hour
                                    df_comments_with_time['day_of_week'] = df_comments_with_time['parsed_time'].dt.day_name()
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # Activity by hour
                                        hourly_activity = df_comments_with_time.groupby('hour').size()
                                        
                                        fig_hourly = px.bar(
                                            x=hourly_activity.index,
                                            y=hourly_activity.values,
                                            title="Comment Activity by Hour",
                                            labels={'x': 'Hour of Day', 'y': 'Number of Comments'},
                                            color=hourly_activity.values,
                                            color_continuous_scale='Blues'
                                        )
                                        fig_hourly.update_layout(height=350)
                                        st.plotly_chart(fig_hourly, use_container_width=True)
                                    
                                    with col2:
                                        # Activity by day of week
                                        daily_activity = df_comments_with_time.groupby('day_of_week').size()
                                        # Reorder days
                                        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                                        daily_activity = daily_activity.reindex([day for day in day_order if day in daily_activity.index])
                                        
                                        fig_daily = px.bar(
                                            x=daily_activity.index,
                                            y=daily_activity.values,
                                            title="Comment Activity by Day",
                                            labels={'x': 'Day of Week', 'y': 'Number of Comments'},
                                            color=daily_activity.values,
                                            color_continuous_scale='Greens'
                                        )
                                        fig_daily.update_layout(height=350, xaxis_tickangle=45)
                                        st.plotly_chart(fig_daily, use_container_width=True)
                                    
                                    # Peak times insight
                                    peak_hour = hourly_activity.idxmax()
                                    peak_day = daily_activity.idxmax()
                                    st.success(f"🕐 Peak activity: **{peak_hour}:00** on **{peak_day}s**")
                                
                                else:
                                    st.info("Insufficient timestamp data for temporal analysis")
                            
                            except Exception as e:
                                st.warning("Unable to parse timestamp data for temporal analysis")
                        
                        else:
                            st.info("Timestamp data not available for temporal analysis")
                            
                            # Show alternative metrics
                            st.markdown("#### 📊 Alternative Time Insights")
                            
                            # Comment posting patterns (if available)
                            if len(df_comments) > 0:
                                # Show comment order insights
                                df_comments['comment_index'] = range(len(df_comments))
                                df_comments['early_comment'] = df_comments['comment_index'] < len(df_comments) * 0.1  # First 10%
                                
                                early_vs_late = df_comments.groupby('early_comment', observed=True).agg({
                                    'like_count': 'mean',
                                    'word_count': 'mean'
                                }).round(2)
                                
                                st.markdown(f"""
                                **📈 Early vs Late Comments:**
                                - Early comments (first 10%): **{early_vs_late.loc[True, 'like_count']:.1f}** avg likes
                                - Later comments: **{early_vs_late.loc[False, 'like_count']:.1f}** avg likes
                                - Early bird advantage: **{'Yes! 🐦' if early_vs_late.loc[True, 'like_count'] > early_vs_late.loc[False, 'like_count'] else 'No significant difference 📊'}**
                                """)
                
                else:
                    st.info("📊 No processed comments available for analytics")
            
            else:
                st.info("📊 Select a video collection to view comprehensive analytics dashboard!")
        
        # === TAB 4: Comment Explorer ===
        with tab4:
            st.markdown("## 💬 Advanced Comment Explorer")
            
            if st.session_state.selected_collection and st.session_state.analysis_results:
                processed_comments = st.session_state.analysis_results.get('processed_comments', [])

                if processed_comments:
                    # Enhanced data processing
                    df_comments = pd.DataFrame(processed_comments)
                    
                    # Normalize column names to handle different data sources
                    if 'likes' in df_comments.columns and 'like_count' not in df_comments.columns:
                        df_comments['like_count'] = df_comments['likes']
                    elif 'like_count' not in df_comments.columns:
                        df_comments['like_count'] = 0
                    
                    if 'author' in df_comments.columns and 'author_display_name' not in df_comments.columns:
                        df_comments['author_display_name'] = df_comments['author']
                    elif 'author_display_name' not in df_comments.columns:
                        df_comments['author_display_name'] = 'Unknown'
                    
                    if 'text' in df_comments.columns and 'text_display' not in df_comments.columns:
                        df_comments['text_display'] = df_comments['text']
                    elif 'original_text' in df_comments.columns and 'text_display' not in df_comments.columns:
                        df_comments['text_display'] = df_comments['original_text']
                    elif 'text_display' not in df_comments.columns:
                        df_comments['text_display'] = 'No text available'
                    
                    if 'timestamp' in df_comments.columns and 'published_at' not in df_comments.columns:
                        df_comments['published_at'] = df_comments['timestamp']
                    elif 'published_at' not in df_comments.columns:
                        df_comments['published_at'] = ''
                    
                    # Enhanced filtering and analysis
                    df_comments['word_count'] = df_comments['text_display'].astype(str).apply(lambda x: len(x.split()))
                    df_comments['char_count'] = df_comments['text_display'].astype(str).apply(len)
                    df_comments['has_question'] = df_comments['text_display'].astype(str).str.contains(r'\?', na=False)
                    df_comments['has_emoji'] = df_comments['text_display'].astype(str).str.contains(r'[😀-🿿]|[🎀-🏿]|[🚀-🛿]', na=False)
                    df_comments['mentions_creator'] = df_comments['text_display'].astype(str).str.contains(r'@|you|your|creator|youtuber', case=False, na=False)
                    
                    # Advanced filtering interface
                    st.markdown("### 🔍 Advanced Comment Filters")
                    
                    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
                    
                    with filter_col1:
                        # Basic filters
                        sort_by = st.selectbox("📊 Sort by", [
                            "like_count", "word_count", "published_at", 
                            "author_display_name", "char_count"
                        ], help="Choose how to sort the comments")
                        
                        sort_order = st.selectbox("🔄 Order", ["Descending", "Ascending"])
                    
                    with filter_col2:
                        # Content filters
                        content_filter = st.selectbox("📝 Content Type", [
                            "All Comments", "Questions Only", "Long Comments (20+ words)", 
                            "Short Comments (<10 words)", "With Emojis", "Mentions Creator"
                        ])
                        
                        # Fix slider max value to prevent min_value == max_value error
                        max_likes = max(int(df_comments['like_count'].max()) if len(df_comments) > 0 and df_comments['like_count'].max() > 0 else 1, 1)
                        min_likes_filter = st.slider("👍 Min Likes", 0, max_likes, 0)
                    
                    with filter_col3:
                        # Engagement filters
                        engagement_tier = st.selectbox("🔥 Engagement Tier", [
                            "All", "Viral (50+ likes)", "Popular (10-49 likes)", 
                            "Moderate (1-9 likes)", "No Engagement (0 likes)"
                        ])
                        
                        author_filter = st.selectbox("👤 Author Type", [
                            "All Authors", "First Time Commenters", "Repeat Commenters", 
                            "Top Contributors", "Recent Activity"
                        ])
                    
                    with filter_col4:
                        # Display options
                        page_size = st.selectbox("📄 Comments per page", [10, 20, 30, 50, 100])
                        
                        show_analytics = st.checkbox("📊 Show Analytics", value=True, help="Show detailed analytics for each comment")
                    
                    # Apply filters
                    filtered_df = df_comments.copy()
                    
                    # Content type filter
                    if content_filter == "Questions Only":
                        filtered_df = filtered_df[filtered_df['has_question']]
                    elif content_filter == "Long Comments (20+ words)":
                        filtered_df = filtered_df[filtered_df['word_count'] >= 20]
                    elif content_filter == "Short Comments (<10 words)":
                        filtered_df = filtered_df[filtered_df['word_count'] < 10]
                    elif content_filter == "With Emojis":
                        filtered_df = filtered_df[filtered_df['has_emoji']]
                    elif content_filter == "Mentions Creator":
                        filtered_df = filtered_df[filtered_df['mentions_creator']]
                    
                    # Engagement tier filter
                    if engagement_tier == "Viral (50+ likes)":
                        filtered_df = filtered_df[filtered_df['like_count'] >= 50]
                    elif engagement_tier == "Popular (10-49 likes)":
                        filtered_df = filtered_df[(filtered_df['like_count'] >= 10) & (filtered_df['like_count'] < 50)]
                    elif engagement_tier == "Moderate (1-9 likes)":
                        filtered_df = filtered_df[(filtered_df['like_count'] >= 1) & (filtered_df['like_count'] < 10)]
                    elif engagement_tier == "No Engagement (0 likes)":
                        filtered_df = filtered_df[filtered_df['like_count'] == 0]
                    
                    # Minimum likes filter
                    if min_likes_filter > 0:
                        filtered_df = filtered_df[filtered_df['like_count'] >= min_likes_filter]
                    
                    # Author type filter
                    if author_filter != "All Authors":
                        comment_counts = df_comments['author_display_name'].value_counts()
                        
                        if author_filter == "First Time Commenters":
                            first_timers = comment_counts[comment_counts == 1].index
                            filtered_df = filtered_df[filtered_df['author_display_name'].isin(first_timers)]
                        elif author_filter == "Repeat Commenters":
                            repeaters = comment_counts[comment_counts > 1].index
                            filtered_df = filtered_df[filtered_df['author_display_name'].isin(repeaters)]
                        elif author_filter == "Top Contributors":
                            top_contributors = comment_counts.head(20).index
                            filtered_df = filtered_df[filtered_df['author_display_name'].isin(top_contributors)]
                    
                    # Sort the filtered data
                    ascending = sort_order == "Ascending"
                    
                    if sort_by in filtered_df.columns:
                        filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)
                    
                    # Filter summary
                    st.markdown("---")
                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                    
                    with summary_col1:
                        st.metric("📊 Filtered Results", format_number(len(filtered_df)))
                    with summary_col2:
                        if len(filtered_df) > 0:
                            avg_engagement = filtered_df['like_count'].mean()
                            st.metric("⚡ Avg Engagement", f"{avg_engagement:.1f}")
                        else:
                            st.metric("⚡ Avg Engagement", "0")
                    with summary_col3:
                        if len(filtered_df) > 0:
                            unique_authors = filtered_df['author_display_name'].nunique()
                            st.metric("👥 Unique Authors", format_number(unique_authors))
                        else:
                            st.metric("👥 Unique Authors", "0")
                    with summary_col4:
                        if len(filtered_df) > 0:
                            total_likes = filtered_df['like_count'].sum()
                            st.metric("❤️ Total Likes", format_number(int(total_likes)))
                        else:
                            st.metric("❤️ Total Likes", "0")
                    
                    # Pagination
                    if len(filtered_df) > 0:
                        total_comments = len(filtered_df)
                        total_pages = (total_comments - 1) // page_size + 1
                        
                        if total_pages > 1:
                            page = st.selectbox("📄 Page", range(1, total_pages + 1), 
                                              help=f"Showing {page_size} comments per page")
                            start_idx = (page - 1) * page_size
                            end_idx = min(start_idx + page_size, total_comments)
                            comments_to_show = filtered_df.iloc[start_idx:end_idx]
                            
                            st.caption(f"Showing comments {start_idx + 1}-{end_idx} of {total_comments}")
                        else:
                            comments_to_show = filtered_df.iloc[:page_size]
                            st.caption(f"Showing all {len(filtered_df)} comments")
                        
                        # Enhanced comment display
                        st.markdown("---")
                        st.markdown(f"### 💬 Comments ({len(comments_to_show)} of {total_comments})")
                        
                        for idx, comment in comments_to_show.iterrows():
                            with st.container():
                                # Create enhanced comment card with creator insights
                                
                                # Determine comment significance
                                significance_level = "🔥 Viral" if comment['like_count'] >= 50 else \
                                                   "⭐ Popular" if comment['like_count'] >= 10 else \
                                                   "👍 Engaged" if comment['like_count'] >= 1 else \
                                                   "💬 Standard"
                                
                                # Comment type analysis
                                comment_type_tags = []
                                if comment['has_question']:
                                    comment_type_tags.append("❓ Question")
                                if comment['has_emoji']:
                                    comment_type_tags.append("😊 Emoji")
                                if comment['mentions_creator']:
                                    comment_type_tags.append("🎯 Mentions You")
                                if comment['word_count'] >= 30:
                                    comment_type_tags.append("📜 Long-form")
                                
                                # Enhanced comment card
                                st.markdown(f"""
                                <div style='background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                                            padding: 1.5rem; border-radius: 12px; margin: 1rem 0; 
                                            border-left: 5px solid {"#ff4757" if comment["like_count"] >= 50 else "#2ed573" if comment["like_count"] >= 10 else "#1e90ff" if comment["like_count"] >= 1 else "#95a5a6"}; 
                                            box-shadow: 0 3px 15px rgba(0,0,0,0.1);'>
                                """, unsafe_allow_html=True)
                                
                                # Header with enhanced metrics
                                header_col1, header_col2, header_col3, header_col4 = st.columns([3, 1, 1, 1])
                                
                                with header_col1:
                                    author = comment.get('author_display_name', 'Unknown')
                                    # Check if returning commenter
                                    author_comment_count = df_comments[df_comments['author_display_name'] == author].shape[0]
                                    returning_indicator = " 🔄" if author_comment_count > 1 else " 🆕"
                                    
                                    st.markdown(f"**👤 {author}**{returning_indicator}")
                                    
                                    # Add tags
                                    if comment_type_tags:
                                        tag_text = " • ".join(comment_type_tags)
                                        st.caption(f"🏷️ {tag_text}")
                                
                                with header_col2:
                                    likes = comment.get('like_count', 0)
                                    st.markdown(f"**👍 {format_number(likes)}**")
                                    st.caption("likes")
                                
                                with header_col3:
                                    st.markdown(f"**{significance_level}**")
                                    st.caption("tier")
                                
                                with header_col4:
                                    words = comment.get('word_count', 0)
                                    st.markdown(f"**📝 {words}**")
                                    st.caption("words")
                                
                                # Comment content with enhanced formatting
                                comment_text = comment.get('text_display', 'No text available')
                                st.markdown("### 💭 Comment")
                                
                                # Highlight questions and mentions
                                if comment['has_question']:
                                    st.markdown(f"> ❓ **Question detected:** {comment_text}")
                                elif comment['mentions_creator']:
                                    st.markdown(f"> 🎯 **Mentions you:** {comment_text}")
                                else:
                                    st.markdown(f"> {comment_text}")
                                
                                # Creator insights for this comment
                                if show_analytics:
                                    with st.expander(f"📊 Creator Insights for this comment"):
                                        insight_col1, insight_col2, insight_col3 = st.columns(3)
                                        
                                        with insight_col1:
                                            # Engagement analysis
                                            engagement_percentile = (df_comments['like_count'] < comment['like_count']).mean() * 100
                                            st.metric("📈 Engagement Rank", f"Top {100-engagement_percentile:.0f}%")
                                            
                                            # Word count analysis
                                            length_percentile = (df_comments['word_count'] < comment['word_count']).mean() * 100
                                            st.metric("📝 Length Rank", f"Top {100-length_percentile:.0f}%")
                                        
                                        with insight_col2:
                                            # Author insights
                                            st.markdown(f"**Author Activity:**")
                                            st.caption(f"• {author_comment_count} total comments")
                                            
                                            author_avg_likes = df_comments[df_comments['author_display_name'] == author]['like_count'].mean()
                                            st.caption(f"• {author_avg_likes:.1f} avg likes per comment")
                                        
                                        with insight_col3:
                                            # Response suggestions
                                            st.markdown("**💡 Action Suggestions:**")
                                            
                                            if comment['has_question']:
                                                st.caption("💬 Consider responding to this question")
                                            if comment['like_count'] >= 20:
                                                st.caption("📌 Consider pinning this comment")
                                            if comment['mentions_creator']:
                                                st.caption("❤️ Great opportunity to engage")
                                            if author_comment_count == 1 and comment['like_count'] >= 5:
                                                st.caption("🎉 Welcome this new engaged viewer")
                                
                                # Footer with timestamp and additional metadata
                                footer_col1, footer_col2, footer_col3 = st.columns(3)
                                
                                with footer_col1:
                                    published = comment.get('published_at', '')
                                    if published:
                                        try:
                                            date_obj = pd.to_datetime(published)
                                            st.caption(f"📅 {date_obj.strftime('%Y-%m-%d %H:%M')}")
                                        except:
                                            st.caption(f"📅 {published}")
                                
                                with footer_col2:
                                    chars = comment.get('char_count', len(comment_text))
                                    st.caption(f"🔤 {chars} characters")
                                
                                with footer_col3:
                                    if 'sentiment_score' in comment:
                                        sentiment = comment['sentiment_score']
                                        sentiment_emoji = "😊" if sentiment > 0.1 else "😐" if sentiment > -0.1 else "😞"
                                        st.caption(f"{sentiment_emoji} Sentiment: {sentiment:.2f}")
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    else:
                        st.warning("� No comments match your current filters. Try adjusting the filter criteria.")
                    
                    # Quick action buttons
                    if len(filtered_df) > 0:
                        st.markdown("---")
                        st.markdown("### 🚀 Quick Actions")
                        
                        action_col1, action_col2, action_col3, action_col4 = st.columns(4)
                        
                        with action_col1:
                            if st.button("📊 Export Filtered Comments", type="secondary"):
                                st.info("📊 Export functionality coming soon!")
                        
                        with action_col2:
                            if st.button("📝 Generate Response Templates", type="secondary"):
                                st.info("📝 Response template generation coming soon!")
                        
                        with action_col3:
                            if st.button("🎯 Find Similar Comments", type="secondary"):
                                st.info("🎯 Similar comment finding coming soon!")
                        
                        with action_col4:
                            if st.button("📈 Create Analytics Report", type="secondary"):
                                st.info("📈 Analytics report creation coming soon!")
                
                else:
                    st.info("💬 No comments available to explore")
            
            else:
                st.info("💬 Select a video collection to start exploring comments with advanced filters and insights!")

        # Tab 5: AI Query Engine
        with tab5:
            st.header("🤖 AI Query Engine")
            st.markdown("---")
            
            # Check if RAG system is available
            if not RAG_AVAILABLE:
                st.error("""
                ❌ **AI features not available**
                
                Required packages are missing. To enable AI chat functionality, run:
                ```
                pip install langchain langchain-anthropic langchain-community
                ```
                """)
                st.stop()
                
            # API Configuration Section
            with st.expander("🔧 AI Configuration", expanded=not st.session_state.rag_initialized):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    anthropic_key = st.text_input(
                        "Anthropic API Key",
                        type="password",
                        value=st.session_state.anthropic_api_key,
                        help="Get your API key from console.anthropic.com",
                        placeholder="sk-ant-..."
                    )
                    
                    if anthropic_key != st.session_state.anthropic_api_key:
                        st.session_state.anthropic_api_key = anthropic_key
                        st.session_state.rag_initialized = False
                
                with col2:
                    model_options = [
                        "claude-3-7-sonnet-20250219",
                        "claude-sonnet-4-20250514", 
                        "claude-opus-4-1-20250805"
                    ]
                    
                    selected_model = st.selectbox(
                        "Claude Model",
                        options=model_options,
                        index=0 if st.session_state.selected_model not in model_options else model_options.index(st.session_state.selected_model),
                        help="Choose Claude model for AI responses"
                    )
                    
                    if selected_model != st.session_state.selected_model:
                        st.session_state.selected_model = selected_model
                        st.session_state.rag_initialized = False
                
                # Initialize RAG System
                if st.button("🚀 Initialize AI System", disabled=not anthropic_key):
                    if not anthropic_key:
                        st.warning("Please enter your Anthropic API key first.")
                    else:
                        with st.spinner("Setting up AI system..."):
                            success = initialize_rag_system(anthropic_key, selected_model)
                            if success:
                                st.success("✅ AI system initialized successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to initialize AI system. Check your API key and try again.")
            
            # AI Status
            if st.session_state.rag_initialized:
                st.success(f"🤖 AI Status: **Ready** (Model: {st.session_state.selected_model})")
                st.info("💡 **Tip**: Claude 3.5 Sonnet offers the best balance of performance and cost for most YouTube analytics tasks.")
            else:
                st.warning("🤖 AI Status: **Not Initialized** - Please configure and initialize the AI system above.")
                st.info("📋 **Note**: Make sure to use the latest Claude model names. Old model names will result in 404 errors.")
                
            st.markdown("---")
            
            # Chat Interface
            if st.session_state.rag_initialized:
                st.subheader("💬 Chat with Your Data")
                
                # Quick Question Templates
                with st.expander("💡 Quick Questions", expanded=False):
                    quick_questions = [
                        "What are the main themes in the comments?",
                        "How is the audience sentiment overall?",
                        "What questions are viewers asking?",
                        "Are there any controversial topics mentioned?",
                        "What do viewers like most about this video?",
                        "What improvements do viewers suggest?",
                        "Who are the most engaged commenters?",
                        "What time do most people comment?",
                        "Are there any spam or negative comments?",
                        "How does this video's engagement compare to others?"
                    ]
                    
                    cols = st.columns(2)
                    for i, question in enumerate(quick_questions):
                        col_idx = i % 2
                        if cols[col_idx].button(f"💭 {question}", key=f"quick_q_{i}"):
                            st.session_state.current_question = question
                            # Add to chat messages
                            st.session_state.chat_messages.append({
                                "role": "user", 
                                "content": question,
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            
                            # Get AI response
                            response = chat_with_ai(question)
                            st.session_state.chat_messages.append({
                                "role": "assistant",
                                "content": response,
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            st.rerun()
                
                # Chat History
                st.subheader("💬 Conversation")
                chat_container = st.container()
                
                with chat_container:
                    # Display chat messages
                    for message in st.session_state.chat_messages:
                        if message["role"] == "user":
                            with st.chat_message("user"):
                                st.markdown(f"**{message['timestamp']}**")
                                st.markdown(message["content"])
                        else:
                            with st.chat_message("assistant"):
                                st.markdown(f"**{message['timestamp']}**")
                                st.markdown(message["content"])
                
                # Chat Input
                user_question = st.chat_input("Ask me anything about your YouTube comments and analytics...")
                
                if user_question:
                    # Add user message
                    st.session_state.chat_messages.append({
                        "role": "user",
                        "content": user_question,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Get AI response
                    response = chat_with_ai(user_question)
                    
                    # Add AI response
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": response,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    st.rerun()
                
                # Clear Chat
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("🗑️ Clear Chat"):
                        st.session_state.chat_messages = []
                        if hasattr(st.session_state, 'ai_memory'):
                            st.session_state.ai_memory.clear()
                        st.rerun()
                
                with col2:
                    if st.button("📊 Data Context"):
                        if st.session_state.selected_collection:
                            video_info = st.session_state.selected_collection
                            context_msg = f"""
**Current Video Context:**
- **Title:** {video_info.get('title', 'N/A')}
- **Video ID:** {video_info.get('video_id', 'N/A')}
- **Comments:** {video_info.get('comment_count', 'N/A')} analyzed
- **Collection:** {video_info.get('collection_name', 'N/A')}
                            """
                            st.info(context_msg)
                        else:
                            st.warning("No video data currently loaded. Please select a video from another tab first.")
            
            else:
                st.info("🔧 Please initialize the AI system above to start chatting with your YouTube data.")
                
                # Show what the AI can do
                st.subheader("🎯 What can the AI help with?")
                
                features = [
                    "**Comment Analysis**: Understand themes, topics, and patterns in your comments",
                    "**Sentiment Insights**: Get detailed sentiment analysis of your audience reactions", 
                    "**Engagement Questions**: Ask about viewer behavior and engagement patterns",
                    "**Content Feedback**: Learn what viewers love and what they want improved",
                    "**Audience Understanding**: Discover who your audience is and what they care about",
                    "**Performance Metrics**: Get insights about your video's performance",
                    "**Competitive Analysis**: Compare with other content (when data available)",
                    "**Growth Strategies**: Get AI-powered recommendations for content improvement"
                ]
                
                for feature in features:
                    st.markdown(f"• {feature}")
                    
                st.markdown("---")
                st.markdown("**💡 Tip:** Load a video's comments in the other tabs first, then come here to chat about the data!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem; font-size: 0.9rem;'>"
    "🎥 YouTube Analytics Dashboard | Built with Streamlit & AI ❤️<br>"
    "🤖 Powered by ChromaDB Vector Search | 🌍 Multilingual Support | ⚡ Real-time Processing"
    "</div>", 
    unsafe_allow_html=True
)
