"""
YouTube Comments API module for extracting and processing comments.
"""

from typing import List, Dict, Any, Optional, Generator
from googleapiclient.errors import HttpError
from .client import YouTubeClient


class CommentsExtractor:
    """
    Class for extracting YouTube comments using the YouTube Data API v3.
    """
    
    def __init__(self, client: YouTubeClient):
        """
        Initialize comments extractor.
        
        Args:
            client: Authenticated YouTube API client
        """
        self.client = client
        self.service = client.get_service()
    
    def get_video_comments(
        self, 
        video_id: str, 
        max_results: int = 100,
        text_format: str = 'plainText',
        order: str = 'time'
    ) -> List[Dict[str, Any]]:
        """
        Extract comments from a YouTube video.
        
        Args:
            video_id: YouTube video ID
            max_results: Maximum number of comments to retrieve
            text_format: Format of comment text ('plainText' or 'html')
            order: Order of comments ('time', 'relevance')
            
        Returns:
            List of comment dictionaries (includes both comments and replies)
        """
        comments = []
        next_page_token = None
        seen_comment_ids = set()  # Track seen comment IDs to prevent duplicates
        comment_threads_processed = 0
        
        try:
            while comment_threads_processed < max_results:
                # Calculate remaining comment threads needed
                remaining = max_results - comment_threads_processed
                page_size = min(remaining, 100)  # API max per request
                
                # Get comment threads (top-level comments)
                request = self.service.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    maxResults=page_size,
                    order=order,
                    textFormat=text_format,
                    pageToken=next_page_token
                )
                
                response = request.execute()
                
                if not response.get('items'):
                    break
                
                # Process comment threads
                for item in response.get('items', []):
                    comment_data = self._extract_comment_data(item)
                    
                    # Add main comment if not already seen
                    if comment_data['id'] not in seen_comment_ids:
                        comments.append(comment_data)
                        seen_comment_ids.add(comment_data['id'])
                    
                    # Add replies if they exist
                    if 'replies' in item:
                        replies = self._extract_replies(item['replies'])
                        for reply in replies:
                            if reply['id'] not in seen_comment_ids:
                                comments.append(reply)
                                seen_comment_ids.add(reply['id'])
                    
                    comment_threads_processed += 1
                
                # Check for next page
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
        except HttpError as e:
            print(f"Error fetching comments for video {video_id}: {e}")
            
        return comments
    
    def get_comments_by_ids(
        self, 
        comment_ids: List[str],
        text_format: str = 'plainText'
    ) -> List[Dict[str, Any]]:
        """
        Get specific comments by their IDs.
        
        Args:
            comment_ids: List of comment IDs
            text_format: Format of comment text ('plainText' or 'html')
            
        Returns:
            List of comment dictionaries
        """
        comments = []
        
        try:
            # Process in batches of 50 (API limit)
            for i in range(0, len(comment_ids), 50):
                batch_ids = comment_ids[i:i+50]
                
                request = self.service.comments().list(
                    part='snippet',
                    id=','.join(batch_ids),
                    textFormat=text_format
                )
                
                response = request.execute()
                
                for item in response.get('items', []):
                    comment_data = {
                        'id': item['id'],
                        'text': item['snippet']['textDisplay'],
                        'author': item['snippet']['authorDisplayName'],
                        'author_channel_id': item['snippet']['authorChannelId']['value'] 
                            if 'authorChannelId' in item['snippet'] else None,
                        'like_count': item['snippet']['likeCount'],
                        'published_at': item['snippet']['publishedAt'],
                        'updated_at': item['snippet']['updatedAt'],
                        'parent_id': item['snippet'].get('parentId'),
                        'type': 'reply' if 'parentId' in item['snippet'] else 'comment'
                    }
                    comments.append(comment_data)
                    
        except HttpError as e:
            print(f"Error fetching comments by IDs: {e}")
            
        return comments
    
    def get_comment_replies(
        self, 
        parent_id: str, 
        max_results: int = 100,
        text_format: str = 'plainText'
    ) -> List[Dict[str, Any]]:
        """
        Get replies to a specific comment.
        
        Args:
            parent_id: Parent comment ID
            max_results: Maximum number of replies to retrieve
            text_format: Format of comment text ('plainText' or 'html')
            
        Returns:
            List of reply dictionaries
        """
        replies = []
        next_page_token = None
        
        try:
            while len(replies) < max_results:
                remaining = max_results - len(replies)
                page_size = min(remaining, 100)
                
                request = self.service.comments().list(
                    part='snippet',
                    parentId=parent_id,
                    maxResults=page_size,
                    textFormat=text_format,
                    pageToken=next_page_token
                )
                
                response = request.execute()
                
                for item in response.get('items', []):
                    reply_data = {
                        'id': item['id'],
                        'text': item['snippet']['textDisplay'],
                        'author': item['snippet']['authorDisplayName'],
                        'author_channel_id': item['snippet']['authorChannelId']['value'] 
                            if 'authorChannelId' in item['snippet'] else None,
                        'like_count': item['snippet']['likeCount'],
                        'published_at': item['snippet']['publishedAt'],
                        'updated_at': item['snippet']['updatedAt'],
                        'parent_id': parent_id,
                        'type': 'reply'
                    }
                    replies.append(reply_data)
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
        except HttpError as e:
            print(f"Error fetching replies for comment {parent_id}: {e}")
            
        return replies
    
    def _extract_comment_data(self, comment_thread: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract comment data from API response.
        
        Args:
            comment_thread: Comment thread item from API response
            
        Returns:
            Processed comment data dictionary
        """
        snippet = comment_thread['snippet']['topLevelComment']['snippet']
        
        return {
            'id': comment_thread['snippet']['topLevelComment']['id'],
            'text': snippet['textDisplay'],
            'author': snippet['authorDisplayName'],
            'author_channel_id': snippet['authorChannelId']['value'] 
                if 'authorChannelId' in snippet else None,
            'like_count': snippet['likeCount'],
            'reply_count': comment_thread['snippet']['totalReplyCount'],
            'published_at': snippet['publishedAt'],
            'updated_at': snippet['updatedAt'],
            'video_id': snippet['videoId'],
            'type': 'comment'
        }
    
    def _extract_replies(self, replies_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract reply data from comment thread.
        
        Args:
            replies_data: Replies data from comment thread
            
        Returns:
            List of processed reply dictionaries
        """
        replies = []
        
        for comment in replies_data.get('comments', []):
            snippet = comment['snippet']
            reply_data = {
                'id': comment['id'],
                'text': snippet['textDisplay'],
                'author': snippet['authorDisplayName'],
                'author_channel_id': snippet['authorChannelId']['value'] 
                    if 'authorChannelId' in snippet else None,
                'like_count': snippet['likeCount'],
                'published_at': snippet['publishedAt'],
                'updated_at': snippet['updatedAt'],
                'parent_id': snippet['parentId'],
                'video_id': snippet['videoId'],
                'type': 'reply'
            }
            replies.append(reply_data)
            
        return replies
    
    def stream_comments(
        self, 
        video_id: str, 
        batch_size: int = 50,
        text_format: str = 'plainText',
        max_batches: Optional[int] = None
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Stream comments in batches to handle large datasets efficiently.
        
        Args:
            video_id: YouTube video ID
            batch_size: Number of comment threads per batch
            text_format: Format of comment text ('plainText' or 'html')
            max_batches: Maximum number of batches to yield (None for unlimited)
            
        Yields:
            Batches of comment dictionaries (includes both comments and replies)
        """
        next_page_token = None
        seen_comment_ids = set()  # Track seen comment IDs across all batches
        batches_yielded = 0
        
        try:
            while True:
                if max_batches and batches_yielded >= max_batches:
                    break
                    
                request = self.service.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    maxResults=batch_size,
                    textFormat=text_format,
                    pageToken=next_page_token,
                    order='time'
                )
                
                response = request.execute()
                
                if not response.get('items'):
                    break
                
                batch_comments = []
                for item in response['items']:
                    comment_data = self._extract_comment_data(item)
                    
                    # Add main comment if not already seen
                    if comment_data['id'] not in seen_comment_ids:
                        batch_comments.append(comment_data)
                        seen_comment_ids.add(comment_data['id'])
                    
                    # Add replies if they exist
                    if 'replies' in item:
                        replies = self._extract_replies(item['replies'])
                        for reply in replies:
                            if reply['id'] not in seen_comment_ids:
                                batch_comments.append(reply)
                                seen_comment_ids.add(reply['id'])
                
                # Only yield if we have new comments
                if batch_comments:
                    yield batch_comments
                    batches_yielded += 1
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
        except HttpError as e:
            print(f"Error streaming comments for video {video_id}: {e}")
            return

    def get_all_comments(
        self, 
        video_id: str, 
        max_comments: Optional[int] = None,
        text_format: str = 'plainText',
        order: str = 'time'
    ) -> List[Dict[str, Any]]:
        """
        Get all comments for a video without duplicates.
        
        Args:
            video_id: YouTube video ID
            max_comments: Maximum number of comments to retrieve (None for all)
            text_format: Format of comment text ('plainText' or 'html')
            order: Order of comments ('time', 'relevance')
            
        Returns:
            List of all comment dictionaries without duplicates
        """
        all_comments = []
        seen_comment_ids = set()
        batch_count = 0
        
        try:
            for batch in self.stream_comments(video_id, batch_size=100, text_format=text_format):
                batch_count += 1
                
                # Add comments from this batch
                for comment in batch:
                    if comment['id'] not in seen_comment_ids:
                        all_comments.append(comment)
                        seen_comment_ids.add(comment['id'])
                
                # Check if we've reached the maximum
                if max_comments and len(all_comments) >= max_comments:
                    break
                    
                # Log progress for large extractions
                if batch_count % 10 == 0:
                    print(f"Processed {batch_count} batches, {len(all_comments)} unique comments")
                    
        except Exception as e:
            print(f"Error getting all comments for video {video_id}: {e}")
            
        # Trim to max_comments if specified
        if max_comments:
            all_comments = all_comments[:max_comments]
            
        print(f"Successfully extracted {len(all_comments)} unique comments")
        return all_comments
