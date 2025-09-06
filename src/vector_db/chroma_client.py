"""
ChromaDB client for storing and querying YouTube comment embeddings.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import os
import uuid
from datetime import datetime
from ..ai_processing import ProcessedComment


class ChromaVectorDB:
    """ChromaDB client for YouTube comment embeddings."""
    """
    Implementation Priority:
        Start with ```intfloat/multilingual-e5-large```
        Test with your actual Tamil-English YouTube comments
        Switch to ```sentence-transformers/paraphrase-multilingual-mpnet-base-v2``` if resource constraints
        Consider ```AI4Bharat/indic-bert``` only if content is primarily Tamil
    """
    def __init__(self, db_path: str = "data/chroma_db", model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        """
        Initialize ChromaDB client with multilingual embedding model.
        
        Args:
            db_path: Path to store ChromaDB data
            model_name: Sentence transformer model name
        """
        self.db_path = db_path
        self.model_name = model_name
        
        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        print(f"✅ Model loaded successfully. Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
    
    def create_collection(self, video_id: str, reset_if_exists: bool = False) -> chromadb.Collection:
        """
        Create or get collection for a video.
        
        Args:
            video_id: YouTube video ID
            reset_if_exists: Whether to reset collection if it exists
            
        Returns:
            ChromaDB collection
        """
        collection_name = f"comments_{video_id}"
        
        try:
            if reset_if_exists:
                try:
                    self.client.delete_collection(collection_name)
                    print(f"🗑️ Deleted existing collection: {collection_name}")
                except:
                    pass
            
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"video_id": video_id, "created_at": datetime.now().isoformat()}
            )
            print(f"✅ Created new collection: {collection_name}")
            
        except chromadb.errors.UniqueConstraintError:
            # Collection already exists, get it
            collection = self.client.get_collection(collection_name)
            print(f"📂 Using existing collection: {collection_name}")
        
        return collection
    
    def embed_and_store_comments(
        self, 
        processed_comments: List[ProcessedComment], 
        video_id: str,
        reset_collection: bool = False
    ) -> Dict[str, Any]:
        """
        Generate embeddings and store processed comments in ChromaDB.
        
        Args:
            processed_comments: List of ProcessedComment objects
            video_id: YouTube video ID
            reset_collection: Whether to reset the collection
            
        Returns:
            Storage summary
        """
        if not processed_comments:
            return {"error": "No processed comments provided"}
        
        print(f"🔄 Starting embedding generation for {len(processed_comments)} comments...")
        
        # Create collection
        collection = self.create_collection(video_id, reset_if_exists=reset_collection)
        
        # Prepare data for embedding
        embedding_texts = []
        metadata_list = []
        comment_ids = []
        
        for comment in processed_comments:
            # Use embedding_ready text for vectorization
            embedding_texts.append(comment.embedding_ready)
            
            # Prepare metadata for ChromaDB (must be simple types)
            metadata = {
                "original_text": comment.text[:1000],  # Truncate to avoid size limits
                "cleaned_text": comment.cleaned_text[:1000],
                "author": comment.author[:100],
                "likes": comment.likes,
                "replies_count": comment.replies_count,
                "timestamp": comment.timestamp,
                "video_id": comment.video_id,
                "comment_id": comment.id or str(uuid.uuid4()),  # Store original comment ID
                "source": "comment",  # CRITICAL: Mark as comment source for RAG
                # AI metadata (flattened)
                "word_count": comment.metadata.get("word_count", 0),
                "character_count": comment.metadata.get("character_count", 0),
                "engagement_score": comment.metadata.get("engagement_score", 0.0),
                "is_question": comment.metadata.get("is_question", False),
                "has_mentions": comment.metadata.get("has_mentions", False),
                "has_hashtags": comment.metadata.get("has_hashtags", False),
                "has_emojis": comment.metadata.get("has_emojis", False),
                "is_long_comment": comment.metadata.get("is_long_comment", False),
                "is_caps": comment.metadata.get("is_caps", False),
            }
            
            # Add timestamp parsing if available
            timestamp_parsed = comment.metadata.get("timestamp_parsed", {})
            if isinstance(timestamp_parsed, dict) and "year" in timestamp_parsed:
                metadata.update({
                    "year": timestamp_parsed.get("year", 0),
                    "month": timestamp_parsed.get("month", 0),
                    "day": timestamp_parsed.get("day", 0),
                    "hour": timestamp_parsed.get("hour", 0),
                    "weekday": timestamp_parsed.get("weekday", 0)
                })
            
            metadata_list.append(metadata)
            comment_ids.append(comment.id or str(uuid.uuid4()))
        
        # Generate embeddings in batches
        print(f"🧠 Generating embeddings using {self.model_name}...")
        embeddings = self.embedding_model.encode(
            embedding_texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Store in ChromaDB
        print(f"💾 Storing {len(embeddings)} embeddings in ChromaDB...")
        collection.add(
            embeddings=embeddings.tolist(),
            metadatas=metadata_list,
            documents=embedding_texts,
            ids=comment_ids
        )
        
        storage_summary = {
            "collection_name": collection.name,
            "video_id": video_id,
            "total_comments_stored": len(processed_comments),
            "embedding_model": self.model_name,
            "embedding_dimension": len(embeddings[0]),
            "storage_timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ Successfully stored {len(processed_comments)} comment embeddings!")
        return storage_summary
    
    def semantic_search(
        self, 
        query: str, 
        video_id: str, 
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform semantic search on stored comments.
        
        Args:
            query: Search query in Tamil, English, or mixed
            video_id: YouTube video ID
            n_results: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            Search results with comments and similarity scores
        """
        collection_name = f"comments_{video_id}"
        
        try:
            collection = self.client.get_collection(collection_name)
        except chromadb.errors.InvalidCollectionException:
            return {"error": f"Collection not found for video: {video_id}"}
        
        # Generate query embedding
        print(f"🔍 Searching for: '{query}'")
        query_embedding = self.embedding_model.encode([query])
        
        # Perform search
        search_params = {
            "query_embeddings": query_embedding.tolist(),
            "n_results": n_results
        }
        
        if filters:
            search_params["where"] = filters
        
        results = collection.query(**search_params)
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            result = {
                "id": results["ids"][0][i],
                "similarity_score": 1 - results["distances"][0][i],  # Convert distance to similarity
                "original_text": results["metadatas"][0][i].get("original_text", ""),
                "cleaned_text": results["metadatas"][0][i].get("cleaned_text", ""),
                "embedding_text": results["documents"][0][i],
                "author": results["metadatas"][0][i].get("author", ""),
                "likes": results["metadatas"][0][i].get("likes", 0),
                "replies_count": results["metadatas"][0][i].get("replies_count", 0),
                "timestamp": results["metadatas"][0][i].get("timestamp", ""),
                "engagement_score": results["metadatas"][0][i].get("engagement_score", 0),
                "metadata": results["metadatas"][0][i]
            }
            formatted_results.append(result)
        
        return {
            "query": query,
            "video_id": video_id,
            "total_results": len(formatted_results),
            "search_timestamp": datetime.now().isoformat(),
            "results": formatted_results
        }
    
    def advanced_search(
        self, 
        query: str, 
        video_id: str,
        min_likes: Optional[int] = None,
        has_replies: Optional[bool] = None,
        is_question: Optional[bool] = None,
        date_range: Optional[Dict[str, int]] = None,
        n_results: int = 10
    ) -> Dict[str, Any]:
        """
        Advanced search with metadata filtering.
        
        Args:
            query: Search query
            video_id: YouTube video ID
            min_likes: Minimum likes filter
            has_replies: Filter by replies existence
            is_question: Filter for questions
            date_range: {"year": 2025, "month": 8} format
            n_results: Number of results
            
        Returns:
            Filtered search results
        """
        # Build filters
        filters = {}
        
        if min_likes is not None:
            filters["likes"] = {"$gte": min_likes}
        
        if has_replies is not None:
            if has_replies:
                filters["replies_count"] = {"$gt": 0}
            else:
                filters["replies_count"] = {"$eq": 0}
        
        if is_question is not None:
            filters["is_question"] = {"$eq": is_question}
        
        if date_range:
            for key, value in date_range.items():
                if key in ["year", "month", "day"]:
                    filters[key] = {"$eq": value}
        
        # Perform filtered search
        return self.semantic_search(query, video_id, n_results, filters)
    
    def get_collection_stats(self, video_id: str) -> Dict[str, Any]:
        """Get statistics about a collection."""
        collection_name = f"comments_{video_id}"
        
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()
            
            return {
                "collection_name": collection_name,
                "video_id": video_id,
                "total_comments": count,
                "embedding_model": self.model_name,
                "collection_metadata": collection.metadata
            }
        except chromadb.errors.InvalidCollectionException:
            return {"error": f"Collection not found for video: {video_id}"}
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections."""
        collections = self.client.list_collections()
        
        return [
            {
                "name": col.name,
                "video_id": col.metadata.get("video_id", "unknown"),
                "created_at": col.metadata.get("created_at", "unknown"),
                "total_comments": col.count()
            }
            for col in collections
        ]
    
    def delete_collection(self, video_id: str) -> Dict[str, Any]:
        """
        Delete a collection for a specific video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dict with operation result
        """
        collection_name = f"comments_{video_id}"
        
        try:
            # Check if collection exists
            collection = self.client.get_collection(collection_name)
            comment_count = collection.count()
            
            # Delete the collection
            self.client.delete_collection(collection_name)
            
            return {
                "status": "success",
                "message": f"Successfully deleted collection for video {video_id}",
                "collection_name": collection_name,
                "deleted_comments_count": comment_count,
                "deleted_at": datetime.now().isoformat()
            }
            
        except chromadb.errors.InvalidCollectionException:
            return {"error": f"Collection not found for video: {video_id}"}
        except Exception as e:
            return {"error": f"Failed to delete collection: {str(e)}"}
    
    def reset_collection(self, video_id: str) -> Dict[str, Any]:
        """
        Reset/clear all data in a collection for a specific video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dict with operation result
        """
        return self.delete_collection(video_id)
    
    def get_all_comments(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all comments from a collection with full metadata.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            List of comment dictionaries with complete data
        """
        collection_name = f"comments_{video_id}"
        
        try:
            collection = self.client.get_collection(collection_name)
            
            # Get all data from the collection including embeddings for potential reuse
            results = collection.get(
                include=['metadatas', 'documents', 'embeddings']
            )
            
            # Convert to list of dictionaries with full metadata
            comments = []
            if results['metadatas']:
                for i, metadata in enumerate(results['metadatas']):
                    comment = {
                        # Basic comment data
                        'id': results['ids'][i] if i < len(results['ids']) else '',
                        'text': metadata.get('original_text', ''),
                        'text_display': metadata.get('original_text', ''),  # For compatibility
                        'cleaned_text': metadata.get('cleaned_text', ''),
                        'embedding_text': results['documents'][i] if i < len(results['documents']) else '',
                        'author': metadata.get('author', ''),
                        'author_display_name': metadata.get('author', ''),  # For compatibility
                        'likes': int(metadata.get('likes', 0)),
                        'like_count': int(metadata.get('likes', 0)),  # For compatibility
                        'replies_count': int(metadata.get('replies_count', 0)),
                        'reply_count': int(metadata.get('replies_count', 0)),  # For compatibility
                        'timestamp': metadata.get('timestamp', ''),
                        'published_at': metadata.get('timestamp', ''),  # For compatibility
                        'video_id': video_id,
                        
                        # AI metadata - directly accessible
                        'word_count': metadata.get('word_count', 0),
                        'character_count': metadata.get('character_count', 0),
                        'engagement_score': metadata.get('engagement_score', 0.0),
                        'is_question': metadata.get('is_question', False),
                        'has_mentions': metadata.get('has_mentions', False),
                        'has_hashtags': metadata.get('has_hashtags', False),
                        'has_emojis': metadata.get('has_emojis', False),
                        'is_long_comment': metadata.get('is_long_comment', False),
                        'is_caps': metadata.get('is_caps', False),
                        
                        # Timestamp parsing
                        'year': metadata.get('year', 0),
                        'month': metadata.get('month', 0),
                        'day': metadata.get('day', 0),
                        'hour': metadata.get('hour', 0),
                        'weekday': metadata.get('weekday', 0),
                        
                        # Full metadata for reference
                        'metadata': metadata,
                        
                        # Embedding for potential reuse (optional)
                        'embedding': results['embeddings'][i] if 'embeddings' in results and i < len(results['embeddings']) else None
                    }
                    comments.append(comment)
            
            return comments
            
        except chromadb.errors.InvalidCollectionException:
            return []
        except Exception as e:
            print(f"Error retrieving comments for {video_id}: {str(e)}")
            return []
    
    def get_rag_ready_documents(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Get documents formatted for RAG system without needing additional processing.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            List of documents ready for RAG with proper content and metadata
        """
        collection_name = f"comments_{video_id}"
        
        try:
            collection = self.client.get_collection(collection_name)
            
            # Get all data from the collection
            results = collection.get(
                include=['metadatas', 'documents']
            )
            
            # Convert to RAG-ready documents
            documents = []
            if results['metadatas']:
                for i, metadata in enumerate(results['metadatas']):
                    doc = {
                        'page_content': results['documents'][i] if i < len(results['documents']) else '',
                        'metadata': {
                            'comment_id': results['ids'][i] if i < len(results['ids']) else '',
                            'author': metadata.get('author', 'Unknown'),
                            'likes': metadata.get('likes', 0),
                            'replies_count': metadata.get('replies_count', 0),
                            'video_id': video_id,
                            'source': 'comment',
                            'engagement_score': metadata.get('engagement_score', 0.0),
                            'is_question': metadata.get('is_question', False),
                            'has_mentions': metadata.get('has_mentions', False),
                            'has_hashtags': metadata.get('has_hashtags', False),
                            'timestamp': metadata.get('timestamp', ''),
                            'year': metadata.get('year', 0),
                            'month': metadata.get('month', 0),
                            'word_count': metadata.get('word_count', 0)
                        }
                    }
                    documents.append(doc)
            
            return documents
            
        except chromadb.errors.InvalidCollectionException:
            return []
        except Exception as e:
            print(f"Error retrieving RAG documents for {video_id}: {str(e)}")
            return []
    
    def get_existing_collection(self, video_id: str):
        """
        Get the existing ChromaDB collection for direct use with LangChain.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            ChromaDB collection object or None if not found
        """
        collection_name = f"comments_{video_id}"
        
        try:
            collection = self.client.get_collection(collection_name)
            return collection
        except chromadb.errors.InvalidCollectionException:
            return None
        except Exception as e:
            print(f"Error getting collection for {video_id}: {str(e)}")
            return None
    
    def get_chroma_client(self):
        """
        Get the underlying ChromaDB client for direct use.
        
        Returns:
            ChromaDB PersistentClient
        """
        return self.client