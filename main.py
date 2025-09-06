"""
Main application entry point for YouTube Analytics.
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.youtube_api.client import YouTubeClient
from src.youtube_api.comments import CommentsExtractor
from src.ai_processing import CommentProcessor, ProcessedComment
from src.vector_db.chroma_client import ChromaVectorDB
from src.utils.helpers import (
    VideoIDExtractor, DataExporter, Logger, RateLimiter, 
    ConfigManager, format_number
)


class YouTubeAnalyzer:
    """Main YouTube Analytics application."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize YouTube Analyzer.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        try:
            self.config = ConfigManager.load_config(config_path)
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}, using defaults")
            self.config = self._get_default_config()
        
        logger_file = 'youtube_analyzer'
        # Set up logging
        self.logger = Logger.setup_logger(
            logger_file, 
            self.config.get('logging', {}).get('file', logger_file + '.log'),
            self.config.get('logging', {}).get('level', 'INFO')
        )
        
        # Initialize components
        self.client = YouTubeClient(config_path)
        self.comments_extractor = None
        
        # AI comment processor
        self.comment_processor = CommentProcessor()
        
        # Vector database
        self.vector_db = ChromaVectorDB()
        
        # Rate limiter
        rate_limit_config = self.config.get('api', {}).get('rate_limit', {})
        self.rate_limiter = RateLimiter(
            max_requests=rate_limit_config.get('max_requests', 100),
            time_window=rate_limit_config.get('time_window', 3600)
        )
        
        self.logger.info("YouTube Analyzer initialized")
    
    def authenticate(self, credentials_path: str = 'config/credentials.json') -> bool:
        """
        Authenticate with YouTube API.
        
        Args:
            credentials_path: Path to credentials file
            
        Returns:
            True if authentication successful
        """
        self.logger.info("Attempting to authenticate with YouTube API")
        
        if self.client.authenticate(credentials_path):
            self.comments_extractor = CommentsExtractor(self.client)
            self.logger.info("Successfully authenticated with YouTube API")
            return True
        else:
            self.logger.error("Failed to authenticate with YouTube API")
            return False
    
    def analyze_video(
        self, 
        video_url: str, 
        max_comments: int = 500,
        export_format: str = 'json',
        reset_collection: bool = True,
        progress_bar: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze a YouTube video's comments and generate insights.
        
        Args:
            video_url: YouTube video URL or ID
            max_comments: Maximum number of comments to analyze
            export_format: Export format for results
            reset_collection: Whether to reset existing collection
            
        Returns:
            Dictionary containing analysis results
        """
        # Extract video ID
        video_id = VideoIDExtractor.extract_video_id(video_url)
        if not video_id:
            raise ValueError(f"Could not extract video ID from: {video_url}")
        
        self.logger.info(f"Starting analysis for video: {video_id}")
        
        if progress_bar:
            progress_bar.progress(value=10, text="🔐 Authenticating...")

        # Check authentication
        if not self.comments_extractor:
            if not self.authenticate():
                raise Exception("Authentication failed")
        
        if progress_bar:
            progress_bar.progress(value=25, text="📋 Extracting comments...")
            
        try:
            # Extract comments with rate limiting and AI preprocessing
            self.logger.info(f"Extracting up to {max_comments} comments...")
            processed_comments = self._extract_comments_with_rate_limit(video_id, max_comments)
            
            if not processed_comments:
                self.logger.warning("No comments found for analysis")
                if progress_bar:
                    progress_bar.progress(value=99, text="⚠️ No comments found")

                return {"error": "No comments found"}
            
            self.logger.info(f"Extracted and processed {len(processed_comments)} AI-ready comments")
            if progress_bar:
                progress_bar.progress(value=50, text=f"Extracted {len(processed_comments)} comments, and Processing for Storing in Vector DB...")

            # Create AI-ready export format
            ai_ready_data = self.comment_processor.create_ai_ready_export(processed_comments)

            # Prepare results with AI-ready structure and vector DB info
            results = {
                'video_id': video_id,
                'analysis_timestamp': datetime.now().isoformat(),
                'total_comments_analyzed': len(processed_comments),
                'ai_ready_data': ai_ready_data,
                'processed_comments': processed_comments  # Keep for backward compatibility
            }

            # Export results
            if export_format:
                if progress_bar:
                    progress_bar.progress(value=75, text=f"💾 Exporting results to {export_format.upper()}...")
                
                filename = f"youtube_analysis_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if export_format.lower() == 'json':
                    # Export AI-ready format
                    export_path = DataExporter.to_json(ai_ready_data, f"{filename}_ai_ready")
                    self.logger.info(f"AI-ready data exported to: {export_path}")
                elif export_format.lower() == 'csv':
                    # Convert ProcessedComments to dict format for CSV
                    csv_data = [
                        {
                            'id': c.id,
                            'text': c.text,
                            'cleaned_text': c.cleaned_text,
                            'embedding_text': c.embedding_ready,
                            'author': c.author,
                            'likes': c.likes,
                            'replies_count': c.replies_count,
                            'timestamp': c.timestamp,
                            'video_id': c.video_id,
                            'word_count': c.metadata.get('word_count', 0),
                            'engagement_score': c.metadata.get('engagement_score', 0),
                            'is_question': c.metadata.get('is_question', False),
                            'has_mentions': c.metadata.get('has_mentions', False),
                            'has_hashtags': c.metadata.get('has_hashtags', False)
                        } for c in processed_comments
                    ]
                    export_path = DataExporter.to_csv(csv_data, filename)
                else:
                    export_path = DataExporter.to_json(ai_ready_data, f"{filename}_ai_ready")
                
                self.logger.info(f"Results exported to: {export_path}")
                results['export_path'] = export_path
            
            if progress_bar:
                progress_bar.progress(value=90, text="💾 Storing comments in vector database...")

            # Store in vector database for semantic search
            self.logger.info("Storing comments in vector database...")
            vector_storage_summary = self.vector_db.embed_and_store_comments(
                processed_comments=processed_comments,
                video_id=video_id,
                reset_collection=reset_collection
            )

            results['vector_storage'] = vector_storage_summary
 
            self.logger.info("Analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            raise
    
    def analyze_multiple_videos(
        self, 
        video_urls: List[str], 
        max_comments_per_video: int = 200
    ) -> Dict[str, Any]:
        """
        Analyze multiple YouTube videos.
        
        Args:
            video_urls: List of YouTube video URLs or IDs
            max_comments_per_video: Max comments per video
            
        Returns:
            Dictionary containing combined analysis results
        """
        self.logger.info(f"Starting batch analysis for {len(video_urls)} videos")
        
        all_results = {}
        all_comments = []
        
        for i, video_url in enumerate(video_urls, 1):
            try:
                self.logger.info(f"Analyzing video {i}/{len(video_urls)}")
                result = self.analyze_video(
                    video_url, 
                    max_comments_per_video, 
                    export_format=None  # Don't export individual results
                )
                
                video_id = result.get('video_id')
                all_results[video_id] = result
                # Extract ProcessedComment objects from the result
                processed_comments = result.get('processed_comments', [])
                all_comments.extend(processed_comments)
                
            except Exception as e:
                self.logger.error(f"Failed to analyze video {video_url}: {str(e)}")
                continue
        
        # Generate combined insights
        if all_comments:
            self.logger.info("Generating combined AI-ready dataset...")
            combined_ai_data = self.comment_processor.create_ai_ready_export(all_comments)
            
            # Export combined results
            combined_results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_videos_analyzed': len(all_results),
                'total_comments_analyzed': len(all_comments),
                'individual_results': all_results,
                'combined_ai_data': combined_ai_data
            }
            
            filename = f"youtube_batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            export_path = DataExporter.to_json(combined_results, filename)
            combined_results['export_path'] = export_path
            
            self.logger.info(f"Batch analysis completed. Results exported to: {export_path}")
            return combined_results
        
        else:
            self.logger.warning("No comments found in any videos")
            return {"error": "No comments found in any videos"}
    
    def _extract_comments_with_rate_limit(
        self, 
        video_id: str, 
        max_comments: int
    ) -> List[ProcessedComment]:
        """Extract comments with rate limiting and AI preprocessing."""
        processed_comments = []
        seen_comment_ids = set()  # Additional deduplication at processor level
        batch_size = min(100, max_comments)  # API maximum
        
        try:
            for batch in self.comments_extractor.stream_comments(video_id, batch_size):
                # Check rate limit
                if not self.rate_limiter.can_make_request():
                    wait_time = self.rate_limiter.get_wait_time()
                    self.logger.warning(f"Rate limit reached. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                
                self.rate_limiter.make_request()
                
                self.logger.info(f"Retrieved {len(batch)} comments in this batch...")
                
                # Process each comment in the batch for AI
                for raw_comment in batch:
                    # Additional deduplication check
                    if raw_comment['id'] not in seen_comment_ids:
                        seen_comment_ids.add(raw_comment['id'])
                        processed_comment = self.comment_processor.process_comment_data(raw_comment, video_id)
                        if processed_comment:  # Only add valid processed comments
                            processed_comments.append(processed_comment)
                
                self.logger.info(f"Processed {len(processed_comments)} unique AI-ready comments so far...")
                
                if len(processed_comments) >= max_comments:
                    break
            
            # Trim to exact requested amount
            final_comments = processed_comments[:max_comments]
            self.logger.info(f"✅ Successfully processed {len(final_comments)} unique comments for AI analysis")
            
            return final_comments
            
        except Exception as e:
            self.logger.error(f"Error extracting comments: {str(e)}")
            return processed_comments  # Return what we have so far
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file not found."""
        return {
            'api': {
                'quota_limit': 10000,
                'rate_limit': {
                    'max_requests': 100,
                    'time_window': 3600
                }
            },
            'logging': {
                'level': 'INFO',
                'file': 'youtube_analyzer.log'
            },
            'processing': {
                'export': {
                    'default_format': 'json'
                }
            }
        }
    
    def get_video_info(self, video_url: str) -> Dict[str, Any]:
        """
        Get basic information about a YouTube video.
        
        Args:
            video_url: YouTube video URL or ID
            
        Returns:
            Dictionary containing video information
        """
        video_id = VideoIDExtractor.extract_video_id(video_url)
        if not video_id:
            raise ValueError(f"Could not extract video ID from: {video_url}")
        
        if not self.client.service:
            if not self.authenticate():
                raise Exception("Authentication failed")
        
        try:
            request = self.client.service.videos().list(
                part='snippet,statistics',
                id=video_id
            )
            response = request.execute()
            
            if not response.get('items'):
                raise ValueError(f"Video not found: {video_id}")
            
            video_data = response['items'][0]
            snippet = video_data['snippet']
            stats = video_data.get('statistics', {})
            
            return {
                'video_id': video_id,
                'title': snippet.get('title'),
                'description': snippet.get('description'),
                'published_at': snippet.get('publishedAt'),
                'channel_title': snippet.get('channelTitle'),
                'view_count': int(stats.get('viewCount', 0)),
                'like_count': int(stats.get('likeCount', 0)),
                'comment_count': int(stats.get('commentCount', 0)),
                'duration': video_data.get('contentDetails', {}).get('duration')
            }
            
        except Exception as e:
            self.logger.error(f"Error getting video info: {str(e)}")
            raise
    
    def print_summary(self, results: Dict[str, Any]):
        """
        Print a formatted summary of analysis results.
        
        Args:
            results: Analysis results dictionary
        """
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return
        
        ai_data = results.get('ai_ready_data', {})
        summary = ai_data.get('dataset_summary', {})
        
        print("\n" + "="*60)
        print("📊 YOUTUBE COMMENTS AI-READY ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"🎥 Video ID: {results.get('video_id', 'N/A')}")
        print(f"📅 Analysis Date: {results.get('analysis_timestamp', 'N/A')}")
        print(f"💬 Comments Processed: {format_number(results.get('total_comments_analyzed', 0))}")
        
        if summary:
            print(f"\n📈 AI PROCESSING METRICS:")
            print(f"   • Total Words: {format_number(summary.get('total_words', 0))}")
            print(f"   • Avg Words/Comment: {summary.get('average_words_per_comment', 0):.1f}")
            print(f"   • Avg Engagement Score: {summary.get('average_engagement_score', 0):.1f}")
            
            content_dist = summary.get('content_distribution', {})
            if content_dist:
                print(f"\n� CONTENT ANALYSIS:")
                print(f"   • Questions: {content_dist.get('questions', 0)}")
                print(f"   • Mentions: {content_dist.get('mentions', 0)}")
                print(f"   • Hashtags: {content_dist.get('hashtags', 0)}")
                print(f"   • With Emojis: {content_dist.get('emojis', 0)}")
                print(f"   • Long Comments (>20 words): {content_dist.get('long_comments', 0)}")
            
            top_authors = summary.get('top_authors', [])
            if top_authors:
                print(f"\n👥 TOP ACTIVE AUTHORS:")
                for author, count in top_authors[:5]:
                    print(f"   • {author}: {count} comments")
        
        print(f"\n🤖 AI STATUS: Ready for semantic search and natural language queries")
        
        if 'export_path' in results:
            print(f"💾 AI-ready data saved to: {results['export_path']}")
        
        print("="*60 + "\n")
    
    def search_comments(
        self, 
        query: str, 
        video_id: str, 
        n_results: int = 10
    ) -> Dict[str, Any]:
        """
        Perform semantic search on stored comments.
        
        Args:
            query: Search query in Tamil, English, or mixed (Tanglish)
            video_id: YouTube video ID
            n_results: Number of results to return
            
        Returns:
            Search results with similarity scores
        """
        self.logger.info(f"Performing semantic search: '{query}' on video {video_id}")
        
        results = self.vector_db.semantic_search(
            query=query,
            video_id=video_id,
            n_results=n_results
        )
        
        if "error" in results:
            self.logger.error(f"Search error: {results['error']}")
            return results
        
        self.logger.info(f"Found {results['total_results']} relevant comments")
        return results
    
    def advanced_search_comments(
        self,
        query: str,
        video_id: str,
        min_likes: Optional[int] = None,
        has_replies: Optional[bool] = None,
        is_question: Optional[bool] = None,
        n_results: int = 10
    ) -> Dict[str, Any]:
        """
        Advanced semantic search with filters.
        
        Args:
            query: Search query
            video_id: YouTube video ID
            min_likes: Minimum likes filter
            has_replies: Filter comments with/without replies
            is_question: Filter questions only
            n_results: Number of results
            
        Returns:
            Filtered search results
        """
        self.logger.info(f"Advanced search: '{query}' with filters")
        
        results = self.vector_db.advanced_search(
            query=query,
            video_id=video_id,
            min_likes=min_likes,
            has_replies=has_replies,
            is_question=is_question,
            n_results=n_results
        )
        
        if "error" in results:
            self.logger.error(f"Advanced search error: {results['error']}")
            return results
        
        self.logger.info(f"Found {results['total_results']} filtered results")
        return results
    
    def get_vector_db_stats(self, video_id: str) -> Dict[str, Any]:
        """Get vector database statistics for a video."""
        return self.vector_db.get_collection_stats(video_id)
    
    def list_stored_videos(self) -> List[Dict[str, Any]]:
        """List all stored videos in the vector database."""
        return self.vector_db.list_collections()

    def get_video_analysis(self, video_id: str) -> Dict[str, Any]:
        """
        Get existing analysis results for a video from the database.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary containing analysis results or None if not found
        """
        try:
            # Get collection stats to check if video exists in DB
            stats = self.vector_db.get_collection_stats(video_id)
            
            if 'error' in stats:
                self.logger.info(f"No analysis found for video {video_id} in database")
                return None
            
            # Get processed comments from vector DB
            processed_comments = self.vector_db.get_all_comments(video_id)
            
            if not processed_comments:
                self.logger.warning(f"No comments found in database for video {video_id}")
                return None
            
            # Convert DB comments to ProcessedComment objects for AI processing
            processed_comment_objects = []
            for comment in processed_comments:
                # Create a minimal ProcessedComment object from DB data
                from src.ai_processing import ProcessedComment
                pc = ProcessedComment(
                    id=comment.get('id', ''),
                    text=comment.get('text', ''),
                    cleaned_text=comment.get('cleaned_text', ''),
                    embedding_ready=comment.get('text', ''),  # Use original text if embedding text not available
                    author=comment.get('author', ''),
                    likes=comment.get('likes', 0),
                    replies_count=comment.get('replies_count', 0),
                    timestamp=comment.get('timestamp', ''),
                    video_id=comment.get('video_id', video_id),
                    metadata={}
                )
                processed_comment_objects.append(pc)
            
            # Generate AI-ready data from database comments
            ai_ready_data = self.comment_processor.create_ai_ready_export(processed_comment_objects)
            
            # Reconstruct the results structure
            results = {
                'video_id': video_id,
                'analysis_timestamp': stats.get('collection_metadata', {}).get('created_at', ''),
                'total_comments_analyzed': len(processed_comments),
                'ai_ready_data': ai_ready_data,
                'processed_comments': processed_comment_objects,
                'vector_storage': stats
            }
            
            self.logger.info(f"Successfully loaded analysis for {video_id} from database: {len(processed_comments)} comments")
            return results
            
        except Exception as e:
            self.logger.error(f"Error loading analysis for {video_id} from database: {str(e)}")
            return None
    
    def get_chroma_client_for_rag(self, video_id: str) -> tuple:
        """
        Get ChromaDB client and collection info for RAG system setup.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Tuple of (chroma_client, collection_name, db_path) or (None, None, None) if not found
        """
        try:
            # Check if collection exists
            stats = self.vector_db.get_collection_stats(video_id)
            
            if 'error' in stats:
                return None, None, None
            
            chroma_client = self.vector_db.get_chroma_client()
            collection_name = f"comments_{video_id}"
            db_path = self.vector_db.db_path
            
            return chroma_client, collection_name, db_path
            
        except Exception as e:
            self.logger.error(f"Error getting ChromaDB client for RAG: {str(e)}")
            return None, None, None

    def reset_collection(self, video_id: str):
        """
        Reset/clear the vector database collection for a specific video.
        
        Args:
            video_id: YouTube video ID
        """
        try:
            self.vector_db.reset_collection(video_id)
            self.logger.info(f"Reset collection for video: {video_id}")
        except Exception as e:
            self.logger.error(f"Error resetting collection for {video_id}: {str(e)}")
            raise


def main():
    """Main application entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='YouTube Comments Analytics Tool')
    parser.add_argument('video_url', help='YouTube video URL or ID to analyze')
    parser.add_argument('--max-comments', type=int, default=500, 
                       help='Maximum number of comments to analyze (default: 500)')
    parser.add_argument('--format', choices=['json', 'csv'], default='json',
                       help='Export format (default: json)')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Configuration file path (default: config/config.yaml)')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = YouTubeAnalyzer(args.config)
        
        # Analyze video
        print(f"🚀 Starting analysis of: {args.video_url}")
        results = analyzer.analyze_video(
            args.video_url, 
            args.max_comments, 
            args.format
        )
        
        # Print summary
        analyzer.print_summary(results)
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
