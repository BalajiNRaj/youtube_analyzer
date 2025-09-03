"""
AI-ready comment processing integrated with extraction.
"""

import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ProcessedComment:
    """Structured comment data ready for AI processing."""
    id: str
    text: str
    author: str
    timestamp: str
    likes: int
    replies_count: int
    video_id: str
    # AI-ready fields
    cleaned_text: str
    embedding_ready: str  # Prepared for vector embedding
    metadata: Dict[str, Any]


class CommentProcessor:
    """Process YouTube comments for AI consumption during extraction."""
    
    def __init__(self):
        """Initialize comment processor."""
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }
    
    def process_comment_data(self, comment_data: Dict[str, Any], video_id: str) -> Optional[ProcessedComment]:
        """
        Process a single comment from CommentsExtractor._extract_comment_data into AI-ready format.
        
        Args:
            comment_data: Processed comment data from CommentsExtractor._extract_comment_data
            video_id: YouTube video ID (may be overridden by comment_data['video_id'])
            
        Returns:
            ProcessedComment object or None if invalid
        """
        try:
            # Extract fields from the processed comment data structure
            # This expects the format returned by CommentsExtractor._extract_comment_data()
            comment_id = comment_data.get('id', '')
            text = comment_data.get('text', '')
            author = comment_data.get('author', 'Unknown')
            timestamp = comment_data.get('published_at', '')
            likes = comment_data.get('like_count', 0)
            reply_count = comment_data.get('reply_count', 0)
            
            # Use video_id from comment data if available, otherwise use provided video_id
            actual_video_id = comment_data.get('video_id', video_id)
            
            # Skip empty or very short comments
            if not text or len(text.strip()) < 3:
                return None
            
            # Clean and process text
            cleaned_text = self._clean_text(text)

            # Create metadata for AI context
            metadata = self._create_metadata(text, cleaned_text, timestamp, likes, reply_count)
            
            embedding_ready = self._prepare_for_embedding(cleaned_text, author, likes, metadata)
            
            return ProcessedComment(
                id=comment_id,
                text=text,
                author=author,
                timestamp=timestamp,
                likes=likes,
                replies_count=reply_count,
                video_id=actual_video_id,
                cleaned_text=cleaned_text,
                embedding_ready=embedding_ready,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error processing comment {comment_data.get('id', 'unknown')}: {e}")
            print(f"Comment data structure: {list(comment_data.keys()) if isinstance(comment_data, dict) else type(comment_data)}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean text for AI processing."""
        if not text:
            return ""
        
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z0-9]+;', '', text)
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation and emojis
        text = re.sub(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF.,!?\-@#]', '', text)
        
        return text.strip()
    
    def _prepare_for_embedding(self, text: str, author: str, likes: int, metadata: Dict[str, Any]) -> str:
        """Prepare text for vector embedding with context."""
        # Add engagement context for better semantic search
        engagement_level = "high" if likes > 10 else "medium" if likes > 2 else "low"

        characteristics = []
        if metadata.get('is_question', False):
            characteristics.append("question")
        if metadata.get('has_emojis', False):
            characteristics.append("with emojis")
        if metadata.get('is_long_comment', False):
            characteristics.append("detailed comment")
        if metadata.get('replies_count', 0) > 0:
            characteristics.append(f"{metadata['replies_count']} replies")
        
        # Build enhanced embedding text
        char_desc = f" ({', '.join(characteristics)})" if characteristics else ""

        # Time context
        time_parsed = metadata.get('timestamp_parsed', {})
        if time_parsed:
            time_context = f"{time_parsed.get('year')}-{time_parsed.get('month'):02d}-{time_parsed.get('day'):02d}"
        else:
            time_context = "unknown date"

        context = f"Comment by {author} on {time_context} ({engagement_level} engagement, {likes} likes {char_desc}): {text}"
        return context
    
    def _create_metadata(self, original_text: str, cleaned_text: str, timestamp: str, likes: int, replies: int) -> Dict[str, Any]:
        """Create metadata for AI context and filtering."""
        # Tokenize for word count (simple split)
        words = [word.lower() for word in cleaned_text.split() if len(word) > 2 and word.lower() not in self.stop_words]
        
        # Calculate engagement score
        engagement_score = (likes * 1.0) + (replies * 2.0)  # Replies weighted higher
        
        # Parse timestamp
        timestamp_data = self._parse_timestamp(timestamp)
        
        # Content analysis
        content_flags = {
            'is_question': '?' in original_text,
            'has_mentions': '@' in original_text,
            'has_hashtags': '#' in original_text,
            'has_emojis': bool(re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', original_text)),
            'is_long_comment': len(words) > 20,
            'is_caps': original_text.isupper() and len(original_text) > 10
        }
        
        return {
            'word_count': len(words),
            'character_count': len(cleaned_text),
            'engagement_score': engagement_score,
            'processed_words': words[:50],  # First 50 words for topic analysis
            'timestamp_parsed': timestamp_data,
            **content_flags
        }
    
    def _parse_timestamp(self, timestamp: str) -> Dict[str, Any]:
        """Parse timestamp into useful components."""
        if not timestamp:
            return {}
        
        try:
            # Handle ISO format timestamps from YouTube API
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return {
                'year': dt.year,
                'month': dt.month,
                'day': dt.day,
                'hour': dt.hour,
                'weekday': dt.weekday(),
                'unix_timestamp': dt.timestamp(),
                'formatted': dt.strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception:
            return {'error': 'Could not parse timestamp'}
    
    def create_ai_ready_export(self, processed_comments: List[ProcessedComment]) -> Dict[str, Any]:
        """
        Create AI-ready export format.
        
        Args:
            processed_comments: List of processed comments
            
        Returns:
            AI-ready dataset
        """
        if not processed_comments:
            return {'error': 'No processed comments provided'}
        
        # Create dataset summary for AI context
        total_comments = len(processed_comments)
        total_words = sum(c.metadata.get('word_count', 0) for c in processed_comments)
        avg_engagement = sum(c.metadata.get('engagement_score', 0) for c in processed_comments) / total_comments
        
        # Top authors by comment count
        author_counts = {}
        for comment in processed_comments:
            author_counts[comment.author] = author_counts.get(comment.author, 0) + 1
        top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Content type distribution
        content_stats = {
            'questions': sum(1 for c in processed_comments if c.metadata.get('is_question', False)),
            'mentions': sum(1 for c in processed_comments if c.metadata.get('has_mentions', False)),
            'hashtags': sum(1 for c in processed_comments if c.metadata.get('has_hashtags', False)),
            'emojis': sum(1 for c in processed_comments if c.metadata.get('has_emojis', False)),
            'long_comments': sum(1 for c in processed_comments if c.metadata.get('is_long_comment', False))
        }
        
        return {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'total_comments': total_comments,
                'format_version': '1.0',
                'ai_ready': True
            },
            'dataset_summary': {
                'total_comments': total_comments,
                'total_words': total_words,
                'average_words_per_comment': total_words / total_comments if total_comments > 0 else 0,
                'average_engagement_score': avg_engagement,
                'top_authors': top_authors,
                'content_distribution': content_stats,
                'date_range': {
                    'earliest': min(c.timestamp for c in processed_comments if c.timestamp),
                    'latest': max(c.timestamp for c in processed_comments if c.timestamp)
                } if any(c.timestamp for c in processed_comments) else {}
            },
            'comments': [
                {
                    'id': c.id,
                    'embedding_text': c.embedding_ready,  # Primary text for AI/vector search
                    'original_text': c.text,
                    'cleaned_text': c.cleaned_text,
                    'author': c.author,
                    'likes': c.likes,
                    'replies_count': c.replies_count,
                    'timestamp': c.timestamp,
                    'video_id': c.video_id,
                    'ai_metadata': c.metadata
                } for c in processed_comments
            ]
        }