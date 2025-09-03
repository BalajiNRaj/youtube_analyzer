# YouTube Analytics Integration Example

This example demonstrates how to use the YouTube Analytics tool to extract comments and generate insights.

## Quick Start Example

```python
from main import YouTubeAnalyzer

# Initialize the analyzer
analyzer = YouTubeAnalyzer()

# Analyze a video (replace with actual video URL)
video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
results = analyzer.analyze_video(video_url, max_comments=100)

# Print summary
analyzer.print_summary(results)
```

## Example Output

```
============================================================
📊 YOUTUBE COMMENTS ANALYSIS SUMMARY
============================================================
🎥 Video ID: dQw4w9WgXcQ
📅 Analysis Date: 2025-01-03T10:30:00
💬 Comments Analyzed: 100

📈 ENGAGEMENT METRICS:
   • Total Likes: 1.2K
   • Avg Likes/Comment: 12.5
   • Avg Word Count: 15.2

🎭 SENTIMENT ANALYSIS:
   • Positive: 65 (65.0%)
   • Neutral: 25 (25.0%)
   • Negative: 10 (10.0%)
   • Dominant: Positive

💾 Results saved to: data/exports/youtube_analysis_dQw4w9WgXcQ_20250103_103000.json
============================================================
```

## Advanced Usage

### Batch Analysis
```python
# Analyze multiple videos
video_urls = [
    "https://www.youtube.com/watch?v=video1",
    "https://www.youtube.com/watch?v=video2",
    "https://www.youtube.com/watch?v=video3"
]

results = analyzer.analyze_multiple_videos(video_urls, max_comments_per_video=200)
```

### Custom Configuration
```python
# Use custom configuration
analyzer = YouTubeAnalyzer('path/to/custom/config.yaml')

# Get video information first
video_info = analyzer.get_video_info(video_url)
print(f"Analyzing: {video_info['title']}")
print(f"Channel: {video_info['channel_title']}")
print(f"Views: {video_info['view_count']:,}")
```

### Export Formats
```python
# Export as JSON (default)
results = analyzer.analyze_video(video_url, export_format='json')

# Export as CSV
results = analyzer.analyze_video(video_url, export_format='csv')
```

## Available Insights

The analysis provides comprehensive insights including:

- **Sentiment Analysis**: Positive, negative, neutral distribution
- **Engagement Metrics**: Likes, replies, engagement scores
- **Content Analysis**: Word frequency, common phrases, topics
- **Author Insights**: Most active commenters, author engagement patterns
- **Temporal Patterns**: Activity by hour, day of week, time trends
- **Trending Topics**: Popular keywords and themes
- **Recommendations**: Actionable insights based on analysis

## Error Handling

The tool includes robust error handling:

```python
try:
    results = analyzer.analyze_video("invalid_url")
except ValueError as e:
    print(f"Invalid video URL: {e}")
except Exception as e:
    print(f"Analysis failed: {e}")
```
