#!/usr/bin/env python3
"""
Demo script for YouTube Analytics without requiring API credentials.
This demonstrates the core functionality using mock data.
"""

import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics.sentiment import SentimentAnalyzer
from src.analytics.insights import InsightGenerator
from src.utils.helpers import VideoIDExtractor, format_number


def create_mock_comments():
    """Create mock comment data for demonstration."""
    return [
        {
            'id': 'comment1',
            'text': 'This video is absolutely amazing! Love the content 😍',
            'author': 'User1',
            'like_count': 25,
            'reply_count': 3,
            'published_at': '2024-01-15T10:30:00Z',
            'type': 'comment'
        },
        {
            'id': 'comment2',
            'text': 'Not really impressed with this one. Could be better.',
            'author': 'User2',
            'like_count': 2,
            'reply_count': 0,
            'published_at': '2024-01-15T11:45:00Z',
            'type': 'comment'
        },
        {
            'id': 'comment3',
            'text': 'Great explanation! This really helped me understand the topic.',
            'author': 'User3',
            'like_count': 18,
            'reply_count': 1,
            'published_at': '2024-01-15T12:20:00Z',
            'type': 'comment'
        },
        {
            'id': 'comment4',
            'text': 'Thanks for sharing this! Very informative and well presented.',
            'author': 'User4',
            'like_count': 12,
            'reply_count': 0,
            'published_at': '2024-01-15T13:10:00Z',
            'type': 'comment'
        },
        {
            'id': 'comment5',
            'text': 'This is terrible content. Waste of time 👎',
            'author': 'User5',
            'like_count': 1,
            'reply_count': 2,
            'published_at': '2024-01-15T14:00:00Z',
            'type': 'comment'
        }
    ]


def demo_video_id_extraction():
    """Demo video ID extraction functionality."""
    print("🔍 Video ID Extraction Demo")
    print("-" * 40)
    
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "dQw4w9WgXcQ"
    ]
    
    for url in test_urls:
        video_id = VideoIDExtractor.extract_video_id(url)
        print(f"URL: {url}")
        print(f"Video ID: {video_id}")
        print(f"Valid: {VideoIDExtractor.validate_video_id(video_id) if video_id else False}")
        print()


def demo_sentiment_analysis():
    """Demo sentiment analysis functionality."""
    print("🎭 Sentiment Analysis Demo")
    print("-" * 40)
    
    comments = create_mock_comments()
    analyzer = SentimentAnalyzer()
    
    # Analyze comments
    analyzed_comments = analyzer.analyze_comments_batch(comments)
    
    for comment in analyzed_comments:
        print(f"Comment: {comment['text'][:50]}...")
        print(f"Sentiment: {comment['sentiment'].title()}")
        print(f"Polarity: {comment['polarity']:.3f}")
        print(f"Confidence: {comment['confidence']:.3f}")
        print()
    
    # Get summary
    summary = analyzer.get_sentiment_summary(analyzed_comments)
    print("📊 Sentiment Summary:")
    print(f"Total Comments: {summary['total_comments']}")
    
    for sentiment, count in summary['sentiment_distribution'].items():
        percentage = summary['sentiment_percentages'][sentiment]
        print(f"{sentiment.title()}: {count} ({percentage:.1f}%)")
    
    print(f"Average Polarity: {summary['average_polarity']:.3f}")
    print()


def demo_insights_generation():
    """Demo insights generation functionality."""
    print("💡 Insights Generation Demo")
    print("-" * 40)
    
    comments = create_mock_comments()
    analyzer = SentimentAnalyzer()
    insights_generator = InsightGenerator()
    
    # Analyze comments first
    analyzed_comments = analyzer.analyze_comments_batch(comments)
    
    # Generate insights
    insights = insights_generator.generate_comprehensive_insights(analyzed_comments)
    
    # Display key insights
    overview = insights.get('overview', {})
    print("📈 Overview:")
    print(f"Total Comments: {overview.get('total_comments', 0)}")
    print(f"Total Likes: {format_number(overview.get('total_likes', 0))}")
    print(f"Avg Likes/Comment: {overview.get('average_likes_per_comment', 0):.1f}")
    print()
    
    engagement = insights.get('engagement_metrics', {})
    if engagement and engagement.get('top_liked_comments'):
        print("🔥 Top Liked Comment:")
        top_comment = engagement['top_liked_comments'][0]
        print(f"Text: {top_comment['text'][:80]}...")
        print(f"Likes: {top_comment['likes']}")
        print(f"Sentiment: {top_comment['sentiment'].title()}")
        print()
    
    # Show recommendations
    recommendations = insights.get('recommendations', {})
    if recommendations and recommendations.get('recommendations'):
        print("📝 Recommendations:")
        for rec in recommendations['recommendations'][:2]:
            print(f"• {rec['recommendation']}")
        print()


def demo_complete_analysis():
    """Demo complete analysis workflow."""
    print("🚀 Complete Analysis Demo")
    print("=" * 60)
    
    # Mock video info
    video_id = "dQw4w9WgXcQ"
    print(f"Analyzing Video ID: {video_id}")
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run all demos
    demo_video_id_extraction()
    demo_sentiment_analysis()
    demo_insights_generation()
    
    print("✅ Demo completed successfully!")
    print("\nTo analyze real YouTube videos:")
    print("1. Set up YouTube API credentials (see config/credentials.json.template)")
    print("2. Run: python main.py [video_url]")
    print("\nExample:")
    print("python main.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --max-comments 100")


if __name__ == "__main__":
    demo_complete_analysis()
