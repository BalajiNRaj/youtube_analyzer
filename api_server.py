"""
FastAPI server for YouTube Comments Vector Database API.
Provides REST endpoints to explore ChromaDB stored comments and perform semantic search.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from datetime import datetime
import json

# Import our modules
from src.vector_db.chroma_client import ChromaVectorDB
from main import YouTubeAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="YouTube Comments Vector DB API",
    description="API for semantic search and exploration of YouTube comments stored in ChromaDB",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances - will be initialized on startup
vector_db = None
analyzer = None

@app.on_event("startup")
async def startup_event():
    """Initialize ChromaDB and YouTubeAnalyzer once when server starts."""
    global vector_db, analyzer
    
    print("🔄 Initializing ChromaDB Vector Database...")
    vector_db = ChromaVectorDB()
    print("✅ ChromaDB initialized successfully!")
    
    print("🔄 Initializing YouTube Analyzer...")
    analyzer = YouTubeAnalyzer()
    print("✅ YouTube Analyzer initialized successfully!")
    
    print("🚀 Server startup complete - ready to handle requests!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown."""
    print("🛑 Server shutting down...")
    # Add any cleanup code here if needed

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query in Tamil, English, or Tanglish")
    video_id: str = Field(..., description="YouTube video ID")
    n_results: int = Field(default=10, ge=1, le=50, description="Number of results (1-50)")

class AdvancedSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    video_id: str = Field(..., description="YouTube video ID")
    n_results: int = Field(default=10, ge=1, le=50, description="Number of results (1-50)")
    min_likes: Optional[int] = Field(default=None, ge=0, description="Minimum likes filter")
    has_replies: Optional[bool] = Field(default=None, description="Filter by replies existence")
    is_question: Optional[bool] = Field(default=None, description="Filter questions only")
    year: Optional[int] = Field(default=None, ge=2020, le=2030, description="Year filter")
    month: Optional[int] = Field(default=None, ge=1, le=12, description="Month filter")

class AnalyzeVideoRequest(BaseModel):
    video_url: str = Field(..., description="YouTube video URL or ID")
    max_comments: int = Field(default=500, ge=1, le=5000, description="Maximum comments to analyze")
    reset_collection: bool = Field(default=True, description="Reset existing collection")

class SearchResponse(BaseModel):
    query: str
    video_id: str
    total_results: int
    search_timestamp: str
    results: List[Dict[str, Any]]

class CollectionStats(BaseModel):
    collection_name: str
    video_id: str
    total_comments: int
    embedding_model: str
    collection_metadata: Dict[str, Any]

# Root endpoint
@app.get("/", summary="API Information")
async def root():
    """Get API information and available endpoints."""
    return {
        "message": "YouTube Comments Vector Database API",
        "version": "1.0.0",
        "description": "Semantic search and exploration of YouTube comments",
        "embedding_model": "intfloat/multilingual-e5-large",
        "supported_languages": ["Tamil", "English", "Tanglish (Tamil-English mix)"],
        "endpoints": {
            "search": "/search - Semantic search comments",
            "advanced_search": "/search/advanced - Advanced search with filters",
            "analyze": "/analyze - Analyze and store new video",
            "stats": "/collections/{video_id}/stats - Get collection statistics",
            "collections": "/collections - List all collections",
            "collection_details": "/collections/{video_id} - Get specific collection details",
            "delete_collection": "DELETE /collections/{video_id} - Delete specific collection",
            "delete_all": "DELETE /collections?confirm=true - Delete all collections (DANGEROUS)",
            "health": "/health - API health check"
        }
    }

# Health check
@app.get("/health", summary="Health Check")
async def health_check():
    """Check API and ChromaDB health."""
    try:
        collections = vector_db.list_collections()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "chromadb_status": "connected",
            "total_collections": len(collections),
            "embedding_model": vector_db.model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Semantic search endpoint
@app.post("/search", response_model=SearchResponse, summary="Semantic Search")
async def semantic_search(request: SearchRequest):
    """
    Perform semantic search on stored YouTube comments.
    Supports Tamil, English, and Tanglish (mixed) queries.
    
    Examples:
    - Tamil: "நல்லா இருக்கு"
    - English: "great explanation"
    - Tanglish: "semma video bro"
    """
    try:
        results = vector_db.semantic_search(
            query=request.query,
            video_id=request.video_id,
            n_results=request.n_results
        )
        
        if "error" in results:
            raise HTTPException(status_code=404, detail=results["error"])
        
        return SearchResponse(**results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Advanced search endpoint
@app.post("/search/advanced", response_model=SearchResponse, summary="Advanced Search with Filters")
async def advanced_search(request: AdvancedSearchRequest):
    """
    Advanced semantic search with metadata filters.
    
    Filters available:
    - min_likes: Minimum number of likes
    - has_replies: Comments with/without replies
    - is_question: Question comments only
    - year/month: Time-based filtering
    """
    try:
        # Build date range filter
        date_range = {}
        if request.year:
            date_range["year"] = request.year
        if request.month:
            date_range["month"] = request.month
        
        results = vector_db.advanced_search(
            query=request.query,
            video_id=request.video_id,
            min_likes=request.min_likes,
            has_replies=request.has_replies,
            is_question=request.is_question,
            date_range=date_range if date_range else None,
            n_results=request.n_results
        )
        
        if "error" in results:
            raise HTTPException(status_code=404, detail=results["error"])
        
        return SearchResponse(**results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced search failed: {str(e)}")

# Analyze and store video
@app.post("/analyze", summary="Analyze and Store Video")
async def analyze_video(request: AnalyzeVideoRequest):
    """
    Analyze a YouTube video and store comments in vector database.
    This will extract comments, process them with AI, generate embeddings, and store in ChromaDB.
    """
    try:
        # Authenticate if needed
        if not analyzer.comments_extractor:
            try:
                analyzer.authenticate()
            except:
                raise HTTPException(
                    status_code=401, 
                    detail="YouTube API authentication required. Please set up credentials."
                )
        
        # Extract video ID
        from src.utils.helpers import VideoIDExtractor
        video_id = VideoIDExtractor.extract_video_id(request.video_url)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL or video ID")
        
        # Analyze video
        results = analyzer.analyze_video(
            video_url=request.video_url,
            max_comments=request.max_comments,
            export_format=None  # Don't export files via API
        )
        
        # Return analysis summary
        return {
            "video_id": video_id,
            "status": "success",
            "message": f"Successfully analyzed and stored {results['total_comments_analyzed']} comments",
            "analysis_summary": {
                "total_comments": results['total_comments_analyzed'],
                "vector_storage": results.get('vector_storage', {}),
                "analysis_timestamp": results['analysis_timestamp']
            },
            "ai_summary": results['ai_ready_data']['dataset_summary'] if 'ai_ready_data' in results else None
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Get collection statistics
@app.get("/collections/{video_id}/stats", response_model=CollectionStats, summary="Get Collection Statistics")
async def get_collection_stats(video_id: str):
    """Get statistics for a specific video's comment collection."""
    try:
        stats = vector_db.get_collection_stats(video_id)
        
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        
        return CollectionStats(**stats)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# List all collections
@app.get("/collections", summary="List All Collections")
async def list_collections():
    """List all stored video collections in the database."""
    try:
        collections = vector_db.list_collections()
        return {
            "total_collections": len(collections),
            "collections": collections,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

# Get specific collection details
@app.get("/collections/{video_id}", summary="Get Collection Details")
async def get_collection_details(video_id: str):
    """Get detailed information about a specific collection."""
    try:
        stats = vector_db.get_collection_stats(video_id)
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        
        return {
            "collection_info": stats,
            "available_operations": [
                f"GET /search - Search comments in this collection",
                f"GET /search/advanced - Advanced search with filters",
                f"GET /collections/{video_id}/sample - Get sample comments"
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection details: {str(e)}")

# Get sample comments from collection
@app.get("/collections/{video_id}/sample", summary="Get Sample Comments")
async def get_sample_comments(
    video_id: str,
    limit: int = Query(default=10, ge=1, le=50, description="Number of sample comments")
):
    """Get sample comments from a collection to understand the data structure."""
    try:
        # Use a generic search to get sample comments
        results = vector_db.semantic_search(
            query="comment",  # Generic query to get any comments
            video_id=video_id,
            n_results=limit
        )
        
        if "error" in results:
            raise HTTPException(status_code=404, detail=results["error"])
        
        return {
            "video_id": video_id,
            "sample_size": len(results["results"]),
            "sample_comments": results["results"],
            "note": "These are sample comments from the collection. Use /search for semantic queries."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get samples: {str(e)}")

# Search by metadata only
@app.get("/collections/{video_id}/filter", summary="Filter by Metadata Only")
async def filter_by_metadata(
    video_id: str,
    min_likes: Optional[int] = Query(default=None, ge=0, description="Minimum likes"),
    max_likes: Optional[int] = Query(default=None, ge=0, description="Maximum likes"),
    has_replies: Optional[bool] = Query(default=None, description="Has replies"),
    is_question: Optional[bool] = Query(default=None, description="Is question"),
    has_emojis: Optional[bool] = Query(default=None, description="Has emojis"),
    is_long_comment: Optional[bool] = Query(default=None, description="Is long comment"),
    limit: int = Query(default=20, ge=1, le=100, description="Result limit")
):
    """Filter comments by metadata criteria without semantic search."""
    try:
        collection_name = f"comments_{video_id}"
        collection = vector_db.client.get_collection(collection_name)
        
        # Build where clause
        where_clause = {}
        
        if min_likes is not None:
            where_clause["likes"] = {"$gte": min_likes}
        if max_likes is not None:
            if "likes" in where_clause:
                where_clause["likes"]["$lte"] = max_likes
            else:
                where_clause["likes"] = {"$lte": max_likes}
        
        if has_replies is not None:
            if has_replies:
                where_clause["replies_count"] = {"$gt": 0}
            else:
                where_clause["replies_count"] = {"$eq": 0}
        
        if is_question is not None:
            where_clause["is_question"] = {"$eq": is_question}
        
        if has_emojis is not None:
            where_clause["has_emojis"] = {"$eq": has_emojis}
        
        if is_long_comment is not None:
            where_clause["is_long_comment"] = {"$eq": is_long_comment}
        
        # Query collection
        results = collection.get(
            where=where_clause if where_clause else None,
            limit=limit
        )
        
        # Format results
        formatted_results = []
        for i, doc_id in enumerate(results["ids"]):
            result = {
                "id": doc_id,
                "original_text": results["metadatas"][i].get("original_text", ""),
                "author": results["metadatas"][i].get("author", ""),
                "likes": results["metadatas"][i].get("likes", 0),
                "replies_count": results["metadatas"][i].get("replies_count", 0),
                "timestamp": results["metadatas"][i].get("timestamp", ""),
                "metadata": results["metadatas"][i]
            }
            formatted_results.append(result)
        
        return {
            "video_id": video_id,
            "filters_applied": where_clause,
            "total_results": len(formatted_results),
            "results": formatted_results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metadata filtering failed: {str(e)}")

# Delete collection endpoint
@app.delete("/collections/{video_id}", summary="Delete Collection")
async def delete_collection(video_id: str):
    """
    Delete a specific video's comment collection from the database.
    WARNING: This action cannot be undone.
    """
    try:
        result = vector_db.delete_collection(video_id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return {
            "status": "success",
            "message": f"Collection for video {video_id} deleted successfully",
            "details": result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")

# Delete all collections endpoint - DANGEROUS
@app.delete("/collections", summary="Delete All Collections")
async def delete_all_collections(confirm: bool = Query(default=False, description="Confirmation required to delete all data")):
    """
    Delete ALL collections from the database.
    WARNING: This will permanently remove all stored comment data!
    Requires explicit confirmation with confirm=true query parameter.
    """
    if not confirm:
        raise HTTPException(
            status_code=400, 
            detail="This action requires confirmation. Add '?confirm=true' to proceed. WARNING: This will delete ALL data!"
        )
    
    try:
        result = vector_db.delete_all_collections()
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "status": "success",
            "message": "All collections deleted successfully",
            "warning": "All YouTube comment data has been permanently removed",
            "details": result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete all collections: {str(e)}")

# Server startup
if __name__ == "__main__":
    print("🚀 Starting YouTube Comments Vector DB API...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Alternative Docs: http://localhost:8000/redoc")
    
    uvicorn.run(
        "api_server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
