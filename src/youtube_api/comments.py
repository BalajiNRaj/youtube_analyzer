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
            max_results: Maximum number of comments to retrieve (1-100)
            text_format: Format of comment text ('plainText' or 'html')
            order: Order of comments ('time', 'relevance')
            
        Returns:
            List of comment dictionaries
        """
        comments = []
        next_page_token = None
        
        try:
            while len(comments) < max_results:
                # Calculate remaining comments needed
                remaining = max_results - len(comments)
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
                
                # Process comment threads
                for item in response.get('items', []):
                    comment_data = self._extract_comment_data(item)
                    comments.append(comment_data)
                    
                    # Add replies if they exist
                    if 'replies' in item:
                        replies = self._extract_replies(item['replies'])
                        comments.extend(replies)
                
                # Check for next page
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
        except HttpError as e:
            print(f"Error fetching comments for video {video_id}: {e}")
            
        return comments[:max_results]
    
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
        text_format: str = 'plainText'
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Stream comments in batches to handle large datasets efficiently.
        
        Args:
            video_id: YouTube video ID
            batch_size: Number of comments per batch
            text_format: Format of comment text ('plainText' or 'html')
            
        Yields:
            Batches of comment dictionaries
        """
        next_page_token = None
        
        try:
            while True:
                request = self.service.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    maxResults=batch_size,
                    textFormat=text_format,
                    pageToken=next_page_token
                )
                
                response = request.execute()
                
                if not response.get('items'):
                    break
                
                batch_comments = []
                for item in response['items']:
                    comment_data = self._extract_comment_data(item)
                    batch_comments.append(comment_data)
                    
                    if 'replies' in item:
                        replies = self._extract_replies(item['replies'])
                        batch_comments.extend(replies)
                
                yield batch_comments
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
        except HttpError as e:
            print(f"Error streaming comments for video {video_id}: {e}")
            return
