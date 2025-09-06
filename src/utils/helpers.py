"""
Utility functions and helpers for YouTube Analytics.
"""

import os
import re
import json
import csv
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging


class DataExporter:
    """Export processed data to various formats."""
    
    @staticmethod
    def to_json(data: Union[Dict, List], filename: str, indent: int = 2) -> str:
        """
        Export data to JSON file.
        
        Args:
            data: Data to export
            filename: Output filename
            indent: JSON indentation
            
        Returns:
            Full path to exported file
        """
        output_path = f"data/exports/{filename}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        
        return output_path
    
    @staticmethod
    def to_csv(data: List[Dict[str, Any]], filename: str) -> str:
        """
        Export data to CSV file.
        
        Args:
            data: List of dictionaries to export
            filename: Output filename
            
        Returns:
            Full path to exported file
        """
        if not data:
            raise ValueError("No data to export")
        
        output_path = f"data/exports/{filename}.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get all unique keys from all dictionaries
        fieldnames = set()
        for item in data:
            fieldnames.update(item.keys())
        
        fieldnames = sorted(list(fieldnames))
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        return output_path


class VideoIDExtractor:
    """Extract YouTube video IDs from various URL formats."""
    
    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """
        Extract video ID from YouTube URL.
        
        Args:
            url: YouTube URL or video ID
            
        Returns:
            Video ID or None if not found
        """
        # If it's already a video ID (11 characters)
        if len(url) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', url):
            return url
        
        # YouTube URL patterns
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/v\/([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/embed\/([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    @staticmethod
    def validate_video_id(video_id: str) -> bool:
        """
        Validate YouTube video ID format.
        
        Args:
            video_id: Video ID to validate
            
        Returns:
            True if valid format
        """
        if not video_id or len(video_id) != 11:
            return False
        
        return bool(re.match(r'^[a-zA-Z0-9_-]{11}$', video_id))


class TextCleaner:
    """Text cleaning utilities for comment processing."""
    
    @staticmethod
    def clean_comment_text(text: str) -> str:
        """
        Clean comment text for analysis.
        
        Args:
            text: Raw comment text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions but keep the @
        text = re.sub(r'@[\w]+', '@user', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation and emojis
        text = re.sub(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF.,!?@]', '', text)
        
        return text.strip()
    
    @staticmethod
    def extract_hashtags(text: str) -> List[str]:
        """
        Extract hashtags from text.
        
        Args:
            text: Text to extract hashtags from
            
        Returns:
            List of hashtags without # symbol
        """
        if not text:
            return []
        
        hashtags = re.findall(r'#(\w+)', text.lower())
        return list(set(hashtags))  # Remove duplicates
    
    @staticmethod
    def extract_mentions(text: str) -> List[str]:
        """
        Extract mentions from text.
        
        Args:
            text: Text to extract mentions from
            
        Returns:
            List of mentioned usernames without @ symbol
        """
        if not text:
            return []
        
        mentions = re.findall(r'@(\w+)', text.lower())
        return list(set(mentions))  # Remove duplicates


class Logger:
    """Logging utility for the application."""
    
    @staticmethod
    def setup_logger(name: str, log_file: str = 'youtube_analyzer.log', level: str = 'INFO') -> logging.Logger:
        """
        Set up logger with file and console handlers.
        
        Args:
            name: Logger name
            log_file: Log file path
            level: Logging level
            
        Returns:
            Configured logger
        """
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent duplicate messages from propagating to parent loggers
        logger.propagate = False
        
        # Remove existing handlers to prevent duplicates
        for handler in logger.handlers[:]:  # Use slice copy to avoid modification during iteration
            logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(f'logs/{log_file}')
        file_handler.setLevel(getattr(logging, level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger


class RateLimiter:
    """Simple rate limiter for API requests."""
    
    def __init__(self, max_requests: int = 100, time_window: int = 3600):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per time window
            time_window: Time window in seconds (default: 1 hour)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def can_make_request(self) -> bool:
        """
        Check if a request can be made within rate limits.
        
        Returns:
            True if request is allowed
        """
        now = datetime.now()
        
        # Remove old requests outside the time window
        self.requests = [req_time for req_time in self.requests 
                        if (now - req_time).total_seconds() < self.time_window]
        
        return len(self.requests) < self.max_requests
    
    def make_request(self) -> bool:
        """
        Record a request if within rate limits.
        
        Returns:
            True if request was recorded
        """
        if self.can_make_request():
            self.requests.append(datetime.now())
            return True
        return False
    
    def get_wait_time(self) -> int:
        """
        Get time to wait before next request is allowed.
        
        Returns:
            Wait time in seconds
        """
        if not self.requests:
            return 0
        
        oldest_request = min(self.requests)
        wait_time = self.time_window - (datetime.now() - oldest_request).total_seconds()
        
        return max(0, int(wait_time))


class ConfigManager:
    """Configuration management utility."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON or YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            elif config_path.endswith(('.yml', '.yaml')):
                import yaml
                return yaml.safe_load(f)
            else:
                raise ValueError("Unsupported configuration file format")
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save configuration
        """
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.endswith('.json'):
                json.dump(config, f, indent=2, ensure_ascii=False)
            elif config_path.endswith(('.yml', '.yaml')):
                import yaml
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


class DataValidator:
    """Validate and sanitize data."""
    
    @staticmethod
    def validate_comment_data(comment: Dict[str, Any]) -> bool:
        """
        Validate comment data structure.
        
        Args:
            comment: Comment dictionary
            
        Returns:
            True if valid
        """
        required_fields = ['id', 'text', 'author']
        
        for field in required_fields:
            if field not in comment or not comment[field]:
                return False
        
        # Validate data types
        if not isinstance(comment['id'], str):
            return False
        
        if not isinstance(comment['text'], str):
            return False
        
        if 'like_count' in comment and not isinstance(comment['like_count'], int):
            return False
        
        return True
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename for safe file system operations.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Trim and remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        
        # Ensure filename is not too long
        if len(sanitized) > 255:
            sanitized = sanitized[:255]
        
        return sanitized or 'unnamed_file'


def format_number(num: Union[int, float]) -> str:
    """
    Format numbers for display (e.g., 1.2K, 1.5M).
    
    Args:
        num: Number to format
        
    Returns:
        Formatted number string
    """
    if num < 1000:
        return str(int(num))
    elif num < 1000000:
        return f"{num/1000:.1f}K"
    elif num < 1000000000:
        return f"{num/1000000:.1f}M"
    else:
        return f"{num/1000000000:.1f}B"


def calculate_engagement_score(comment: Dict[str, Any]) -> float:
    """
    Calculate engagement score for a comment.
    
    Args:
        comment: Comment dictionary
        
    Returns:
        Engagement score
    """
    like_count = comment.get('like_count', 0)
    reply_count = comment.get('reply_count', 0)
    text_length = len(comment.get('text', ''))
    
    # Weight factors
    like_weight = 1.0
    reply_weight = 2.0  # Replies are more valuable
    length_bonus = min(text_length / 100, 1.0)  # Bonus for longer, thoughtful comments
    
    score = (like_count * like_weight) + (reply_count * reply_weight) + length_bonus
    
    return score
