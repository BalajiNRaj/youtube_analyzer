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
from src.analytics.sentiment import SentimentAnalyzer
from src.analytics.insights import InsightGenerator
from src.utils.helpers import (
    VideoIDExtractor, DataExporter, Logger, RateLimiter, 
    ConfigManager, format_number
)


class YouTubeAnalyzer:
    """Main YouTube Analytics application."""
    
    def __init__(self, config_path: str = 'config/config.yaml', video_url: str = None):
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
        
        video_id = VideoIDExtractor.extract_video_id(video_url)
        logger_file = 'youtube_analyzer_' + video_id
        # Set up logging
        self.logger = Logger.setup_logger(
            logger_file, 
            self.config.get('logging', {}).get('file', logger_file + '.log'),
            self.config.get('logging', {}).get('level', 'INFO')
        )
        
        # Initialize components
        self.client = YouTubeClient(config_path)
        self.comments_extractor = None
        self.sentiment_analyzer = SentimentAnalyzer()
        self.insights_generator = InsightGenerator()
        
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
        export_format: str = 'json'
    ) -> Dict[str, Any]:
        """
        Analyze a YouTube video's comments and generate insights.
        
        Args:
            video_url: YouTube video URL or ID
            max_comments: Maximum number of comments to analyze
            export_format: Export format for results
            
        Returns:
            Dictionary containing analysis results
        """
        # Extract video ID
        video_id = VideoIDExtractor.extract_video_id(video_url)
        if not video_id:
            raise ValueError(f"Could not extract video ID from: {video_url}")
        
        self.logger.info(f"Starting analysis for video: {video_id}")
        
        # Check authentication
        if not self.comments_extractor:
            if not self.authenticate():
                raise Exception("Authentication failed")
        
        try:
            # Extract comments with rate limiting
            self.logger.info(f"Extracting up to {max_comments} comments...")
            comments = self._extract_comments_with_rate_limit(video_id, max_comments)
            
            if not comments:
                self.logger.warning("No comments found for analysis")
                return {"error": "No comments found"}
            
            self.logger.info(f"Extracted {len(comments)} comments")
            
            # Perform sentiment analysis
            # self.logger.info("Performing sentiment analysis...")
            analyzed_comments = comments # self.sentiment_analyzer.analyze_comments_batch(comments)
            
            # Generate insights
            # self.logger.info("Generating insights...")
            # insights = self.insights_generator.generate_comprehensive_insights(analyzed_comments)
            
            # Prepare results
            results = {
              'video_id': video_id,
              'analysis_timestamp': datetime.now().isoformat(),
              'total_comments_analyzed': len(analyzed_comments),
              'comments': analyzed_comments,
              # 'insights': insights
            }
            
            # Export results
            if export_format:
                filename = f"youtube_analysis_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if export_format.lower() == 'json':
                    export_path = DataExporter.to_json(results, filename)
                elif export_format.lower() == 'csv':
                    export_path = DataExporter.to_csv(analyzed_comments, filename)
                else:
                    export_path = DataExporter.to_json(results, filename)
                
                self.logger.info(f"Results exported to: {export_path}")
                results['export_path'] = export_path
            
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
                all_comments.extend(result.get('comments', []))
                
            except Exception as e:
                self.logger.error(f"Failed to analyze video {video_url}: {str(e)}")
                continue
        
        # Generate combined insights
        if all_comments:
            self.logger.info("Generating combined insights...")
            combined_insights = self.insights_generator.generate_comprehensive_insights(all_comments)
            
            # Export combined results
            combined_results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_videos_analyzed': len(all_results),
                'total_comments_analyzed': len(all_comments),
                'individual_results': all_results,
                'combined_insights': combined_insights
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
    ) -> List[Dict[str, Any]]:
        """Extract comments with rate limiting."""
        comments = []
        batch_size = min(100, max_comments)  # API maximum
        
        try:
            for batch in self.comments_extractor.stream_comments(video_id, batch_size):
                # Check rate limit
                if not self.rate_limiter.can_make_request():
                    wait_time = self.rate_limiter.get_wait_time()
                    self.logger.warning(f"Rate limit reached. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                
                self.rate_limiter.make_request()
                comments.extend(batch)
                
                self.logger.info(f"Extracted {len(comments)} comments so far...")
                
                if len(comments) >= max_comments:
                    break
            
            return comments[:max_comments]
            
        except Exception as e:
            self.logger.error(f"Error extracting comments: {str(e)}")
            return comments  # Return what we have so far
    
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
        
        insights = results.get('insights', {})
        overview = insights.get('overview', {})
        sentiment = insights.get('sentiment_insights', {})
        
        print("\n" + "="*60)
        print("📊 YOUTUBE COMMENTS ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"🎥 Video ID: {results.get('video_id', 'N/A')}")
        print(f"📅 Analysis Date: {results.get('analysis_timestamp', 'N/A')}")
        print(f"💬 Comments Analyzed: {format_number(results.get('total_comments_analyzed', 0))}")
        
        if overview:
            print(f"\n📈 ENGAGEMENT METRICS:")
            print(f"   • Total Likes: {format_number(overview.get('total_likes', 0))}")
            print(f"   • Avg Likes/Comment: {overview.get('average_likes_per_comment', 0)}")
            print(f"   • Avg Word Count: {overview.get('average_word_count', 0)}")
        
        if sentiment:
            overall_sentiment = sentiment.get('overall_sentiment', {})
            distribution = overall_sentiment.get('distribution', {})
            
            print(f"\n🎭 SENTIMENT ANALYSIS:")
            print(f"   • Positive: {distribution.get('positive', 0)} ({distribution.get('positive', 0)/sum(distribution.values())*100:.1f}%)")
            print(f"   • Neutral: {distribution.get('neutral', 0)} ({distribution.get('neutral', 0)/sum(distribution.values())*100:.1f}%)")
            print(f"   • Negative: {distribution.get('negative', 0)} ({distribution.get('negative', 0)/sum(distribution.values())*100:.1f}%)")
            print(f"   • Dominant: {overall_sentiment.get('dominant_sentiment', 'N/A').title()}")
        
        if 'export_path' in results:
            print(f"\n💾 Results saved to: {results['export_path']}")
        
        print("="*60 + "\n")


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
        analyzer = YouTubeAnalyzer(args.config, args.video_url)
        
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
