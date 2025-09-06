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

## Streamlit Dashboard Usage

### Running the Dashboard
```bash
# Start the interactive dashboard
streamlit run streamlit_app.py

# Access at http://localhost:8501
```

### Dashboard Features
1. **📊 Video Analysis**: Performance KPIs, health scores, engagement metrics
2. **🔍 Comment Search**: AI-powered semantic search with filters
3. **📈 Analytics**: Deep-dive analytics with interactive visualizations
4. **💬 Comment Explorer**: Advanced filtering and exploration tools
5. **🤖 AI Query Engine**: RAG-based conversational AI for data insights

## AI Query Engine Examples

### Setup
1. Navigate to the "🤖 AI Query Engine" tab
2. Enter your Anthropic API key
3. Select a Claude model (Sonnet recommended)
4. Click "🚀 Initialize AI System"

### Example Conversations

**Query**: "What are the main themes in the comments?"

**AI Response**: "Based on the comment analysis, I can see several main themes emerging:

1. **Video Quality** (45% of comments): Viewers are praising the high production value and clear audio
2. **Educational Content** (30% of comments): Many appreciate the learning value and clear explanations
3. **Technical Discussion** (15% of comments): Detailed discussions about the topic specifics
4. **Requests for More** (10% of comments): Viewers asking for follow-up videos or related content

The overall sentiment is very positive (78% positive, 18% neutral, 4% negative), indicating strong audience satisfaction."

---

**Query**: "Are there any controversial topics mentioned?"

**AI Response**: "Looking through the comment data, there are minimal controversial topics. The few contentious areas I found:

- **Technical Disagreements** (3% of comments): Some viewers debating specific technical points
- **Alternative Approaches** (2% of comments): Suggestions for different methods or viewpoints
- **Spam/Off-topic** (1% of comments): A small amount of irrelevant content

Overall, this video has a very healthy comment section with constructive discussions. The controversy level is very low, which suggests good community management and content that resonates well with your audience."

---

**Query**: "What improvements do viewers suggest?"

**AI Response**: "Based on viewer feedback, here are the main improvement suggestions:

**Most Requested**:
1. **Longer Format** (mentioned 23 times): Viewers want more detailed explanations
2. **More Examples** (mentioned 18 times): Request for additional practical examples
3. **Follow-up Topics** (mentioned 15 times): Specific related topics they'd like covered

**Technical Suggestions**:
- Better screen recording quality in some sections
- Timestamps for easier navigation
- Downloadable resources or references

**Content Ideas**:
- Part 2 continuation of this topic
- Behind-the-scenes content
- Q&A format addressing common questions

These suggestions indicate high engagement and a desire for more content, which is excellent for channel growth!"

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
