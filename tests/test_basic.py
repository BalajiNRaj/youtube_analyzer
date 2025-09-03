"""
Basic tests for YouTube Analytics components.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import VideoIDExtractor, TextCleaner, DataValidator


class TestVideoIDExtractor(unittest.TestCase):
    """Test video ID extraction functionality."""
    
    def test_extract_from_watch_url(self):
        """Test extraction from standard watch URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = VideoIDExtractor.extract_video_id(url)
        self.assertEqual(video_id, "dQw4w9WgXcQ")
    
    def test_extract_from_short_url(self):
        """Test extraction from short URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        video_id = VideoIDExtractor.extract_video_id(url)
        self.assertEqual(video_id, "dQw4w9WgXcQ")
    
    def test_extract_from_embed_url(self):
        """Test extraction from embed URL."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        video_id = VideoIDExtractor.extract_video_id(url)
        self.assertEqual(video_id, "dQw4w9WgXcQ")
    
    def test_extract_from_video_id(self):
        """Test extraction from plain video ID."""
        video_id = VideoIDExtractor.extract_video_id("dQw4w9WgXcQ")
        self.assertEqual(video_id, "dQw4w9WgXcQ")
    
    def test_invalid_url(self):
        """Test with invalid URL."""
        video_id = VideoIDExtractor.extract_video_id("clearly_invalid_url_12345")
        self.assertIsNone(video_id)
    
    def test_validate_video_id(self):
        """Test video ID validation."""
        self.assertTrue(VideoIDExtractor.validate_video_id("dQw4w9WgXcQ"))
        self.assertFalse(VideoIDExtractor.validate_video_id("invalid"))
        self.assertFalse(VideoIDExtractor.validate_video_id(""))


class TestTextCleaner(unittest.TestCase):
    """Test text cleaning functionality."""
    
    def test_clean_comment_text(self):
        """Test comment text cleaning."""
        text = "Check this out! https://example.com @user #hashtag   "
        cleaned = TextCleaner.clean_comment_text(text)
        expected = "Check this out! @user hashtag"  # Updated expected result
        self.assertEqual(cleaned, expected)
    
    def test_extract_hashtags(self):
        """Test hashtag extraction."""
        text = "Great video! #awesome #youtube #content"
        hashtags = TextCleaner.extract_hashtags(text)
        self.assertEqual(set(hashtags), {"awesome", "youtube", "content"})
    
    def test_extract_mentions(self):
        """Test mention extraction."""
        text = "Thanks @creator and @friend for this!"
        mentions = TextCleaner.extract_mentions(text)
        self.assertEqual(set(mentions), {"creator", "friend"})


class TestDataValidator(unittest.TestCase):
    """Test data validation functionality."""
    
    def test_validate_valid_comment(self):
        """Test validation of valid comment data."""
        comment = {
            'id': 'test123',
            'text': 'Great video!',
            'author': 'TestUser',
            'like_count': 5
        }
        self.assertTrue(DataValidator.validate_comment_data(comment))
    
    def test_validate_invalid_comment(self):
        """Test validation of invalid comment data."""
        comment = {
            'id': 'test123',
            'text': '',  # Empty text
            'author': 'TestUser'
        }
        self.assertFalse(DataValidator.validate_comment_data(comment))
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        filename = 'test<file>name?.txt'
        sanitized = DataValidator.sanitize_filename(filename)
        self.assertEqual(sanitized, 'test_file_name_.txt')


if __name__ == '__main__':
    unittest.main()
