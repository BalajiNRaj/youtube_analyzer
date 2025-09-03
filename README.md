# YouTube Analytics Integration

A comprehensive Python application to extract YouTube comments and generate detailed insights using the YouTube Data API v3.

## Features

- **Comment Extraction**: Extract comments from YouTube videos with rate limiting
- **Sentiment Analysis**: Analyze comment sentiment using TextBlob and custom indicators
- **Comprehensive Insights**: Generate detailed analytics and recommendations
- **Multiple Export Formats**: Export data as JSON or CSV
- **Batch Processing**: Analyze multiple videos simultaneously
- **Real-time Processing**: Stream comments for large datasets

## Quick Start

### 1. Installation

```bash
# Clone the repository (or extract the files)
cd youtube-analyzer

# Install dependencies
pip install -r requirements.txt
```

### 2. Set up YouTube API Credentials

**Option A: API Key (Recommended for read-only access)**
```bash
export YOUTUBE_API_KEY="your_api_key_here"
```

**Option B: OAuth 2.0 Credentials**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable YouTube Data API v3
4. Create OAuth 2.0 credentials
5. Download the JSON file and save as `config/credentials.json`

### 3. Run Analysis

```bash
# Analyze a single video
python main.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --max-comments 500

# Run demo without API credentials
python demo.py
```

## Usage Examples

### Basic Usage
```python
from main import YouTubeAnalyzer

# Initialize analyzer
analyzer = YouTubeAnalyzer()

# Analyze video
results = analyzer.analyze_video(
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    max_comments=100
)

# Print summary
analyzer.print_summary(results)
```

### Advanced Features
```python
# Get video information
video_info = analyzer.get_video_info(video_url)

# Batch analysis
video_urls = ["url1", "url2", "url3"]
batch_results = analyzer.analyze_multiple_videos(video_urls)

# Custom configuration
analyzer = YouTubeAnalyzer('config/custom_config.yaml')
```

## Available Insights

The tool provides comprehensive analytics including:

- **📊 Overview Statistics**: Comment counts, likes, engagement rates
- **🎭 Sentiment Analysis**: Positive/negative/neutral distribution with confidence scores
- **📈 Engagement Metrics**: Top liked comments, reply patterns, engagement scores  
- **📝 Content Analysis**: Word frequency, common phrases, trending topics
- **👥 Author Insights**: Most active commenters, author engagement patterns
- **⏰ Temporal Patterns**: Activity by hour, day of week, posting trends
- **🔥 Trending Topics**: Popular keywords and emerging themes
- **💡 Recommendations**: Actionable insights based on analysis patterns

## Project Structure

```
youtube-analyzer/
├── src/
│   ├── youtube_api/          # YouTube API integration
│   │   ├── client.py         # API client and authentication
│   │   └── comments.py       # Comment extraction logic
│   ├── analytics/            # Analysis modules
│   │   ├── sentiment.py      # Sentiment analysis
│   │   └── insights.py       # Insights generation
│   └── utils/               # Utility functions
│       └── helpers.py       # Helper functions and utilities
├── config/                  # Configuration files
│   ├── config.yaml         # Main configuration
│   └── credentials.json.template
├── data/exports/           # Exported analysis results
├── tests/                  # Unit tests
├── main.py                # Main application entry point
├── demo.py                # Demo script (no API required)
└── requirements.txt       # Python dependencies
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
api:
  quota_limit: 10000
  requests_per_minute: 60
  
analysis:
  sentiment:
    confidence_threshold: 0.1
    enable_emotion_analysis: true
  
processing:
  export:
    default_format: json
```

## Output Examples

### Terminal Output
```
============================================================
📊 YOUTUBE COMMENTS ANALYSIS SUMMARY  
============================================================
🎥 Video ID: dQw4w9WgXcQ
📅 Analysis Date: 2025-01-03T10:30:00
💬 Comments Analyzed: 500

📈 ENGAGEMENT METRICS:
   • Total Likes: 2.1K
   • Avg Likes/Comment: 4.2
   • Avg Word Count: 15.8

🎭 SENTIMENT ANALYSIS:
   • Positive: 325 (65.0%)
   • Neutral: 125 (25.0%) 
   • Negative: 50 (10.0%)
   • Dominant: Positive

💾 Results saved to: data/exports/youtube_analysis_dQw4w9WgXcQ_20250103_103000.json
============================================================
```

### JSON Export Structure
```json
{
  "video_id": "dQw4w9WgXcQ",
  "analysis_timestamp": "2025-01-03T10:30:00",
  "total_comments_analyzed": 500,
  "insights": {
    "overview": {
      "total_comments": 500,
      "total_likes": 2100,
      "average_likes_per_comment": 4.2
    },
    "sentiment_insights": {
      "overall_sentiment": {
        "distribution": {"positive": 325, "neutral": 125, "negative": 50},
        "dominant_sentiment": "positive"
      }
    }
  }
}
```

## API Endpoints Used

- `GET /youtube/v3/commentThreads` - List comment threads for videos
- `GET /youtube/v3/comments` - List individual comments  
- `GET /youtube/v3/videos` - Get video metadata

## Rate Limiting & Quotas

- **API Quota**: 10,000 units per day (default)
- **Rate Limiting**: Built-in rate limiting to prevent quota exhaustion
- **Batch Processing**: Efficient batch requests to maximize quota usage

## Error Handling

The application includes comprehensive error handling:

- API authentication failures
- Rate limit exceeded scenarios  
- Invalid video URLs or IDs
- Network connectivity issues
- Quota exhaustion handling

## Testing

```bash
# Run unit tests
python tests/test_basic.py

# Run demo (no API required)
python demo.py
```

## Contributing

The project is structured for easy extension:

- Add new analysis methods to `src/analytics/`
- Extend API functionality in `src/youtube_api/`
- Add utility functions to `src/utils/helpers.py`
- Update configuration in `config/config.yaml`

## Requirements

- Python 3.8+
- YouTube Data API v3 access
- Internet connection for API requests

## License

MIT License - see LICENSE file for details.
