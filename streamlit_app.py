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
from src.utils.helpers import VideoIDExtractor, format_number, Logger

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

# Initialize logger
logger = Logger.setup_logger('streamlit_app', level='DEBUG')
logger.info("Streamlit app starting up")

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
    logger.debug("Initializing Streamlit session state")
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
        logger.debug("Initialized analyzer state")
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
        logger.debug("Initialized analysis_results state")
    if 'video_info' not in st.session_state:
        st.session_state.video_info = None
        logger.debug("Initialized video_info state")
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
        logger.debug("Initialized search_results state")
    if 'collections' not in st.session_state:
        st.session_state.collections = []
        logger.debug("Initialized collections state")
    if 'selected_collection' not in st.session_state:
        st.session_state.selected_collection = None
        logger.debug("Initialized selected_collection state")
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "🤖 AI Query Engine"
        logger.debug("Initialized current_tab state")
    if 'show_new_video_form' not in st.session_state:
        st.session_state.show_new_video_form = False
        logger.debug("Initialized show_new_video_form state")
    
    # AI Chat session state
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
        logger.debug("Initialized chat_messages state")
    if 'ai_chain' not in st.session_state:
        st.session_state.ai_chain = None
        logger.debug("Initialized ai_chain state")
    if 'ai_memory' not in st.session_state:
        st.session_state.ai_memory = None
        logger.debug("Initialized ai_memory state")
    if 'ai_llm' not in st.session_state:
        st.session_state.ai_llm = None
        logger.debug("Initialized ai_llm state")
    if 'ai_retriever' not in st.session_state:
        st.session_state.ai_retriever = None
        logger.debug("Initialized ai_retriever state")
    if 'anthropic_api_key' not in st.session_state:
        st.session_state.anthropic_api_key = ""
        logger.debug("Initialized anthropic_api_key state")
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "claude-3-7-sonnet-20250219"
        logger.debug("Initialized selected_model state")
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False
        logger.debug("Initialized rag_initialized state")
    
    logger.debug("Session state initialization completed")

def initialize_analyzer():
    """Initialize the YouTube analyzer"""
    if not st.session_state.analyzer:
        logger.info("Initializing YouTube Analytics System...")
        with st.spinner("🔄 Initializing YouTube Analytics System..."):
            try:
                st.session_state.analyzer = YouTubeAnalyzer()
                logger.info("YouTubeAnalyzer instance created successfully")
                if st.session_state.analyzer.authenticate():
                    logger.info("YouTube API authentication successful")
                    # Load collections on startup
                    st.session_state.collections = st.session_state.analyzer.list_stored_videos()
                    logger.info(f"Loaded {len(st.session_state.collections)} existing collections")
                    return True
                else:
                    logger.error("YouTube API authentication failed")
                    st.error("❌ Authentication failed. Please check credentials.")
                    return False
            except Exception as e:
                logger.error(f"YouTubeAnalyzer initialization failed: {str(e)}", exc_info=True)
                st.error(f"❌ Initialization failed: {str(e)}")
                return False
    return True

def load_collection(collection):
    """Load a specific collection's data"""
    if st.session_state.analyzer:
        try:
            # Load the analysis results for this video
            video_id = collection['video_id']
            logger.info(f"Loading collection for video: {video_id}")
            logger.debug(f"Collection details: {collection}")
            results = st.session_state.analyzer.get_video_analysis(video_id)
            if results:
                logger.info(f"Successfully loaded analysis results for {video_id}")
                logger.debug(f"Analysis results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
                st.session_state.analysis_results = results
                st.session_state.selected_collection = collection
                st.session_state.show_new_video_form = False
                return True
            else:
                logger.warning(f"No analysis results found for video: {video_id}")
        except Exception as e:
            logger.error(f"Error loading collection for {collection.get('video_id', 'unknown')}: {str(e)}", exc_info=True)
            st.error(f"Error loading collection: {str(e)}")
    return False

def analyze_new_video(video_url, max_comments, export_format, reset_collection):
    """Analyze a new YouTube video"""
    try:
        logger.info(f"Starting new video analysis: {video_url}")
        logger.debug(f"Parameters - max_comments: {max_comments}, export_format: {export_format}, reset_collection: {reset_collection}")
        
        # Extract video ID
        extractor = VideoIDExtractor()
        video_id = extractor.extract_video_id(video_url)
        
        if not video_id:
            logger.warning(f"Invalid YouTube URL or video ID: {video_url}")
            st.error("❌ Invalid YouTube URL or video ID")
            return False
            
        logger.info(f"Extracted video ID: {video_id}")
        
        # Show progress
        progress_container = st.container()
        with progress_container:
            st.info(f"🎯 Analyzing video: {video_id}")
            progress_bar = st.progress(0)
            
            # Step 1: Reset collection if requested
            if reset_collection:
                logger.info(f"Resetting existing collection for video: {video_id}")
                progress_bar.progress(10, text="🔄 Resetting existing collection...")
                st.session_state.analyzer.reset_collection(video_id)
                logger.info("Collection reset completed")
            
            # Step 2: Extract comments
            logger.info(f"Starting comment extraction for video: {video_id}")
            progress_bar.progress(25, text="📥 Extracting comments...")
            
            results = st.session_state.analyzer.analyze_video(
                video_url=video_url,
                max_comments=max_comments,
                export_format=export_format,
                reset_collection=reset_collection,
                progress_bar=progress_bar
            )
            
            progress_bar.progress(75, text="🤖 Processing with AI...")
            logger.info("AI processing phase initiated")
            
            if results:
                logger.info(f"Analysis completed successfully for video: {video_id}")
                logger.debug(f"Results summary - Keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
                
                progress_bar.progress(100, text="✅ Analysis complete!")
                
                # Update session state
                st.session_state.analysis_results = results
                st.session_state.collections = st.session_state.analyzer.list_stored_videos()
                logger.info("Session state updated with new results")
                
                # Find and select the current collection
                for collection in st.session_state.collections:
                    if collection['video_id'] == video_id:
                        st.session_state.selected_collection = collection
                        logger.info(f"Selected collection for video: {video_id}")
                        break
                
                st.success(f"✅ Analysis complete! Video ID: {video_id}")
                st.session_state.show_new_video_form = False
                return True
            else:
                logger.error(f"Analysis failed for video: {video_id} - No results returned")
                st.error("❌ Analysis failed")
                return False
                
    except Exception as e:
        logger.error(f"Error during video analysis for {video_url}: {str(e)}", exc_info=True)
        st.error(f"❌ Error during analysis: {str(e)}")
        return False

def _create_fallback_vectorstore(embeddings, video_id):
    """Create fallback vectorstore from processed comments when ChromaDB fails"""
    try:
        logger.info("Creating fallback vector store from processed comments")
        documents = []
        results = st.session_state.analysis_results
        processed_comments = results.get('processed_comments', [])
        
        logger.info(f"Creating documents from {len(processed_comments)} processed comments")
        
        # Create documents from comments
        comment_count = 0
        for i, comment in enumerate(processed_comments[:200]):  # Limit for efficiency
            text = getattr(comment, 'text_display', getattr(comment, 'text', ''))
            if hasattr(comment, 'get'):  # Dict-like object
                text = comment.get('text_display', comment.get('text', ''))
            
            if text and len(text.strip()) > 10:
                metadata = {
                    'author': getattr(comment, 'author_display_name', getattr(comment, 'author', 'Unknown')),
                    'likes': getattr(comment, 'like_count', getattr(comment, 'likes', 0)),
                    'comment_id': i,
                    'video_id': video_id,
                    'source': 'comment'
                }
                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)
                comment_count += 1
        
        logger.info(f"Created {comment_count} comment documents for fallback RAG")
        
        # Add AI insights as documents if available
        if 'ai_ready_data' in results and 'insights' in results['ai_ready_data']:
            insights = results['ai_ready_data']['insights']
            logger.info(f"Processing {len(insights)} AI insights for fallback RAG context")
            
            insight_count = 0
            for insight_type, insight_data in insights.items():
                if isinstance(insight_data, dict) and 'summary' in insight_data:
                    doc = Document(
                        page_content=f"{insight_type}: {insight_data['summary']}",
                        metadata={'type': 'insight', 'insight_type': insight_type, 'source': 'ai_analysis'}
                    )
                    documents.append(doc)
                    insight_count += 1
            
            logger.info(f"Created {insight_count} insight documents for fallback RAG")
        
        if not documents:
            logger.warning("No documents available for fallback - creating default context")
            documents = [Document(
                page_content="No specific video data loaded. I can help with general YouTube analytics questions.",
                metadata={'type': 'default', 'source': 'system'}
            )]
        
        # Create new in-memory vector store
        logger.info(f"Creating fallback vector store with {len(documents)} documents")
        vectorstore = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=None  # In-memory for this session
        )
        return vectorstore
        
    except Exception as e:
        logger.error(f"Failed to create fallback vector store: {str(e)}")
        return None

def _setup_vectorstore(embeddings, video_id=None):
    """Set up vectorstore from ChromaDB or fallback to in-memory store"""
    if video_id and st.session_state.selected_collection and st.session_state.analysis_results:
        try:
            # Get ChromaDB connection details from analyzer
            chroma_client, collection_name, db_path = st.session_state.analyzer.get_chroma_client_for_rag(video_id)
            
            if chroma_client and collection_name:
                logger.info(f"Successfully got ChromaDB connection: {collection_name}")
                
                # Create vectorstore from existing collection
                vectorstore = Chroma(
                    client=chroma_client,
                    collection_name=collection_name,
                    embedding_function=embeddings,
                )
                
                # Verify collection has data
                existing_collection = chroma_client.get_collection(collection_name)
                collection_count = existing_collection.count()
                logger.info(f"Using existing vector store with {collection_count} embedded documents")
                
                if collection_count == 0:
                    raise Exception("Empty collection - no documents found")
                    
                return vectorstore
            else:
                raise Exception("Could not get ChromaDB connection from analyzer")
                
        except Exception as db_error:
            logger.warning(f"Could not use existing ChromaDB collection: {str(db_error)}")
            logger.info("Falling back to creating new in-memory vector store from processed comments")
            
            # Fallback: Create new documents from analysis results
            vectorstore = _create_fallback_vectorstore(embeddings, video_id)
            if not vectorstore:
                raise Exception("Failed to create fallback vector store")
            return vectorstore
    else:
        # No video selected - create default context
        logger.warning("No video selected - creating default RAG context")
        documents = [Document(
            page_content="No specific video data loaded. I can help with general YouTube analytics questions.",
            metadata={'type': 'default', 'source': 'system'}
        )]
        
        return Chroma.from_documents(documents, embeddings, persist_directory=None)

def _setup_conversational_chain(llm, retriever):
    """Set up the conversational retrieval chain with memory"""
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
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        verbose=True
    )
    
    return chain, memory

def initialize_rag_system(anthropic_api_key, model_name):
    """Initialize RAG system with Anthropic Claude using existing ChromaDB embeddings"""
    if not RAG_AVAILABLE:
        logger.error("RAG system initialization failed - required packages not installed")
        st.error("❌ Required packages not installed. Please install langchain, langchain-anthropic, and langchain-community.")
        return False
    
    try:
        logger.info(f"Starting RAG system initialization with model: {model_name}")
        logger.debug(f"API key provided: {'Yes' if anthropic_api_key else 'No'}")
        
        # Set environment variable for Anthropic API key
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
        logger.debug("Anthropic API key set in environment variables")
        
        # Initialize embeddings - IMPORTANT: Using same model as ChromaDB storage
        logger.info("Initializing HuggingFace embeddings with sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        logger.info("⚠️  CRITICAL: Using same embedding model as ChromaDB storage to ensure compatibility")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        logger.info("Embeddings model loaded successfully")
        
        # Set up vectorstore
        video_id = st.session_state.selected_collection.get('video_id') if st.session_state.selected_collection else None
        vectorstore = _setup_vectorstore(embeddings, video_id)
        logger.info(f"Vector store setup completed for video: {video_id or 'default'}")
        
        # Test the retriever to make sure it works
        logger.info("Testing vector store retriever...")
        test_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        test_results = test_retriever.get_relevant_documents("test query")
        logger.info(f"✅ Retriever test successful: {len(test_results)} documents retrieved")
        
        # Log some sample document content for verification
        for i, doc in enumerate(test_results[:2]):
            sample_content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            logger.info(f"Sample doc {i+1}: {sample_content}")
            logger.debug(f"Sample doc {i+1} metadata: {doc.metadata}")
        
        # Initialize Claude
        logger.info(f"Initializing Claude model: {model_name}")
        llm = ChatAnthropic(
            model=model_name,
            temperature=0.7,
            max_tokens=1000
        )
        logger.info("Claude model initialized successfully")
        
        # Create conversational chain
        logger.info("Creating conversational retrieval chain")
        chain, memory = _setup_conversational_chain(llm, vectorstore.as_retriever(search_kwargs={"k": 5}))
        logger.info("Conversational retrieval chain created successfully")
        
        # Store in session state
        st.session_state.ai_chain = chain
        st.session_state.ai_memory = memory
        st.session_state.ai_llm = llm
        st.session_state.ai_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        st.session_state.rag_initialized = True
        
        logger.info("RAG system initialization completed successfully")
        logger.debug(f"Session state updated - rag_initialized: {st.session_state.rag_initialized}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}", exc_info=True)
        st.error(f"❌ Failed to initialize RAG system: {str(e)}")
        return False

def _validate_context_quality(relevant_docs):
    """Validate if retrieved context has meaningful content"""
    has_real_context = False
    context_preview = []
    
    for i, doc in enumerate(relevant_docs):
        content = doc.page_content
        metadata = doc.metadata
        
        logger.debug(f"Document {i+1}: {content[:200]}...")
        logger.debug(f"Document {i+1} metadata: {metadata}")
        
        # Check if this is actual video data, not default context
        if metadata.get('source') in ['comment', 'ai_analysis']:
            has_real_context = True
        
        context_preview.append({
            'content': content[:200] + '...' if len(content) > 200 else content,
            'source': metadata.get('source', 'unknown'),
            'author': metadata.get('author', 'N/A'),
            'likes': metadata.get('likes', 0)
        })
    
    return has_real_context, context_preview

def chat_with_ai(message):
    """Send a message to the AI and get response using the conversational chain"""
    if not st.session_state.rag_initialized or not st.session_state.ai_chain:
        logger.warning("AI chat attempted but RAG system not initialized")
        return "❌ AI system not initialized. Please set up your API key and model first."
    
    try:
        logger.info(f"Received user query: {message}")
        logger.debug(f"Current video context: {st.session_state.selected_collection.get('video_id', 'none') if st.session_state.selected_collection else 'none'}")
        
        with st.spinner("🤖 AI is thinking..."):
            # Option 1: Use the conversational chain directly (recommended)
            try:
                logger.info("Using conversational retrieval chain for response")
                response = st.session_state.ai_chain.invoke({"question": message})
                ai_answer = response.get("answer", "No response generated")
                
                logger.info(f"Chain response received (length: {len(ai_answer)} chars)")
                logger.debug(f"Full chain response: {ai_answer}")
                
                return ai_answer
                
            except Exception as chain_error:
                logger.warning(f"Chain execution failed: {str(chain_error)}")
                logger.info("Falling back to manual retrieval and LLM call")
                
                # Option 2: Fallback to manual approach with validation
                return _manual_rag_fallback(message)
            
    except Exception as e:
        return _handle_chat_error(e)

def _manual_rag_fallback(message):
    """Manual RAG approach as fallback when chain fails"""
    try:
        # Step 1: Get relevant context using retriever
        retriever = st.session_state.ai_retriever or st.session_state.ai_chain.retriever
        
        logger.info("Step 1: Retrieving relevant context from vector store (fallback)")
        relevant_docs = retriever.get_relevant_documents(message)
        logger.info(f"Retrieved {len(relevant_docs)} relevant documents for {message}")
        
        # Step 2: Validate context quality
        has_real_context, context_preview = _validate_context_quality(relevant_docs)
        
        if not has_real_context:
            logger.warning("No meaningful context retrieved - only default/system messages found")
            return """❌ **No relevant context found for your query.**
            
This might happen because:
1. The video data isn't properly loaded in the RAG system
2. Your query doesn't match the available comment content
3. The vector database connection isn't working properly

Please try:
- Selecting a different video from the sidebar
- Re-initializing the RAG system with your API key
- Asking more specific questions about the video content"""
        
        logger.info(f"✅ Found meaningful context: {sum(1 for cp in context_preview if cp['source'] in ['comment', 'ai_analysis'])} relevant documents")
        
        # Step 3: Build context string for prompt
        context_str = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'unknown')}\n"
            f"Content: {doc.page_content}\n"
            f"Author: {doc.metadata.get('author', 'N/A')}\n"
            f"Likes: {doc.metadata.get('likes', 0)}"
            for doc in relevant_docs if doc.metadata.get('source') in ['comment', 'ai_analysis']
        ])
        
        if not context_str.strip():
            logger.warning("Context string is empty after filtering")
            return "❌ No relevant context could be extracted for your query."
        
        logger.debug(f"Context string length: {len(context_str)} characters")
        
        # Step 4: Use stored LLM for response
        logger.info("Step 2: Generating response with stored LLM")
        llm = st.session_state.ai_llm
        
        # Step 5: Create focused prompt with context
        logger.info("Step 3: Building focused prompt with retrieved context")
        prompt_template = """You are an expert YouTube analytics AI assistant analyzing comment data from a specific YouTube video.

**RELEVANT CONTEXT FROM VIDEO COMMENTS:**
{context}

**USER QUESTION:** {question}

**INSTRUCTIONS:**
- Base your response ONLY on the provided context from the video comments
- If the context doesn't contain information to answer the question, say so clearly
- Provide specific insights from the comment data when available
- Include relevant statistics, sentiment patterns, or notable comments when applicable

**RESPONSE:**"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Step 6: Generate response
        logger.info("Step 4: Generating AI response with context")
        formatted_prompt = prompt.format(context=context_str, question=message)
        
        logger.debug(f"Sending prompt to Claude (length: {len(formatted_prompt)} chars)")
        
        response = llm.invoke(formatted_prompt)
        ai_answer = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"AI response received (length: {len(ai_answer)} chars)")
        logger.debug(f"Full AI response: {ai_answer}")
        
        # Update memory manually for conversation continuity
        if st.session_state.ai_memory:
            try:
                from langchain.schema.messages import HumanMessage, AIMessage
                st.session_state.ai_memory.chat_memory.add_user_message(message)
                st.session_state.ai_memory.chat_memory.add_ai_message(ai_answer)
                logger.debug("Updated conversation memory")
            except Exception as mem_error:
                logger.warning(f"Could not update memory: {str(mem_error)}")
        
        return ai_answer
        
    except Exception as fallback_error:
        logger.error(f"Manual fallback failed: {str(fallback_error)}")
        return f"❌ Error in fallback response generation: {str(fallback_error)}"

def _handle_chat_error(e):
    """Handle errors in AI chat with enhanced error messages"""
    error_msg = str(e)
    logger.error(f"Error in AI chat: {error_msg}", exc_info=True)
    
    # Enhanced error handling
    if "404" in error_msg and "model" in error_msg:
        logger.error(f"Model not found error - Current model: {st.session_state.selected_model}")
        return f"❌ Model not found. Please check that you're using a valid Claude model name. Current model: {st.session_state.selected_model}"
    elif "401" in error_msg or "authentication" in error_msg.lower():
        logger.error("Authentication failed - Invalid API key")
        return "❌ Authentication failed. Please check your Anthropic API key."
    elif "rate_limit" in error_msg.lower() or "429" in error_msg:
        logger.warning("Rate limit exceeded")
        return "❌ Rate limit exceeded. Please wait a moment and try again."
    else:
        logger.error(f"Unexpected error in AI chat: {error_msg}")
        return f"❌ Error getting AI response: {error_msg}"

# Initialize app
init_session_state()
logger.info("Session state initialized")

# Header
st.markdown('<h1 class="main-header">🎥 YouTube Analytics Dashboard</h1>', unsafe_allow_html=True)

# Initialize analyzer
if not st.session_state.analyzer:
    logger.info("Attempting to initialize analyzer")
    initialize_analyzer()
else:
    logger.debug("Analyzer already initialized")

# Main layout: Sidebar + Content
col_sidebar, col_main = st.columns([1, 3])

# === LEFT SIDEBAR ===
with col_sidebar:
    st.markdown('<div class="sidebar-title">📚 Collections</div>', unsafe_allow_html=True)
    
    # "Analyze New Video" button
    if st.button("🎯 Analyze New Video", type="secondary" if not st.session_state.show_new_video_form else "primary", key="new_video_btn"):
        logger.info("User clicked 'Analyze New Video' button")
        st.session_state.show_new_video_form = True
        st.session_state.current_tab = "📊 Video Analysis"
        st.session_state.selected_collection = None
        logger.debug("Switched to new video form mode")
    
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
                    logger.info(f"User selected collection: {collection['video_id']} ({collection['total_comments']} comments)")
                    load_collection(collection)
                    st.session_state.show_new_video_form = False
            
            with col2:
                st.caption(f"{collection['total_comments']}")
    
        # Refresh button
        st.markdown("---")
        if st.button("🔄 Refresh", key="refresh_collections"):
            logger.info("User clicked refresh collections button")
            if st.session_state.analyzer:
                old_count = len(st.session_state.collections)
                st.session_state.collections = st.session_state.analyzer.list_stored_videos()
                new_count = len(st.session_state.collections)
                logger.info(f"Collections refreshed: {old_count} -> {new_count} collections")
                st.rerun()
            else:
                logger.warning("Refresh attempted but analyzer not available")
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
                logger.info(f"User submitted video analysis form - URL: {video_url}, Max Comments: {max_comments}, Format: {export_format}, Reset: {reset_collection}")
                analyze_new_video(video_url, max_comments, export_format, reset_collection)
            elif submitted and not video_url:
                logger.warning("Form submitted without video URL")
                st.error("❌ Please enter a YouTube video URL or ID")
            elif submitted and not st.session_state.analyzer:
                logger.error("Form submitted but analyzer not initialized")
                st.error("❌ Analyzer not initialized. Please refresh the page.")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
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
            
            # Video selection reminder
            if not st.session_state.selected_collection:
                st.warning("💡 **Tip**: Select a video from the sidebar first to chat about specific video comments. Without a video selected, you'll get general analytics help only.")
            
            # Initialize RAG System
            if st.button("🚀 Initialize AI System", disabled=not anthropic_key):
                if not anthropic_key:
                    logger.warning("AI initialization attempted without API key")
                    st.warning("Please enter your Anthropic API key first.")
                else:
                    logger.info(f"User initiated AI system initialization with model: {selected_model}")
                    with st.spinner("Setting up AI system..."):
                        success = initialize_rag_system(anthropic_key, selected_model)
                        if success:
                            logger.info("AI system initialization successful - reloading interface")
                            st.success("✅ AI system initialized successfully!")
                            st.rerun()
                        else:
                            logger.error("AI system initialization failed")
                            st.error("❌ Failed to initialize AI system. Check your API key and try again.")
        
        # AI Status
        if st.session_state.rag_initialized:
            st.success(f"🤖 AI Status: **Ready** (Model: {st.session_state.selected_model})")
        else:
            st.warning("🤖 AI Status: **Not Initialized** - Please configure and initialize the AI system above.")
            
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
                        logger.info(f"User selected quick question: {question}")
                        st.session_state.current_question = question
                        # Add to chat messages
                        st.session_state.chat_messages.append({
                            "role": "user", 
                            "content": question,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                        logger.debug(f"Added user message to chat history (total messages: {len(st.session_state.chat_messages)})")
                        
                        # Get AI response
                        response = chat_with_ai(question)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                        logger.info(f"AI response added to chat (response length: {len(response)} chars)")
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
                logger.info(f"User asked manual question: {user_question}")
                # Add user message
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": user_question,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                logger.debug(f"Added user message to chat history (total messages: {len(st.session_state.chat_messages)})")
                
                # Get AI response
                response = chat_with_ai(user_question)
                
                # Add AI response
                st.session_state.chat_messages.append({
                    "role": "assistant", 
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                logger.info(f"AI response added to chat (response length: {len(response)} chars)")
                
                st.rerun()
            
            # Clear Chat
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("🗑️ Clear Chat"):
                    chat_count = len(st.session_state.chat_messages)
                    logger.info(f"User cleared chat history ({chat_count} messages)")
                    st.session_state.chat_messages = []
                    if hasattr(st.session_state, 'ai_memory'):
                        st.session_state.ai_memory.clear()
                        logger.debug("AI memory cleared")
                    st.rerun()
            
            with col2:
                if st.button("📊 Data Context"):
                    logger.info("User requested data context information")
                    if st.session_state.selected_collection:
                        video_info = st.session_state.selected_collection
                        logger.debug(f"Showing context for video: {video_info.get('video_id', 'unknown')}")
                        context_msg = f"""
                          **Current Video Context:**
                          - **Title:** {video_info.get('title', 'N/A')}
                          - **Video ID:** {video_info.get('video_id', 'N/A')}
                          - **Comments:** {video_info.get('comment_count', 'N/A')} analyzed
                          - **Collection:** {video_info.get('collection_name', 'N/A')}
                        """
                        st.info(context_msg)
                    else:
                        logger.warning("Data context requested but no video data loaded")
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

# Log app completion
logger.debug("Streamlit app interface fully loaded")
