"""
Sentiment Analysis module for YouTube comments.
"""

from typing import List, Dict, Any, Tuple
import re
from textblob import TextBlob
import pandas as pd
from collections import Counter


class SentimentAnalyzer:
    """
    Sentiment analysis for YouTube comments using TextBlob and custom rules.
    """
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        # Common YouTube-specific positive and negative indicators
        self.positive_indicators = [
            'love', 'amazing', 'awesome', 'great', 'fantastic', 'excellent',
            'wonderful', 'brilliant', 'perfect', 'incredible', 'outstanding',
            '❤️', '😍', '👏', '🔥', '💯', '🎉', '😊', '😃', '😆'
        ]
        
        self.negative_indicators = [
            'hate', 'terrible', 'awful', 'worst', 'horrible', 'disgusting',
            'stupid', 'dumb', 'trash', 'garbage', 'disappointing',
            '😡', '😠', '😒', '👎', '💩', '😢', '😭', '🤮', '😤'
        ]
        
        # Compile regex patterns for efficiency
        self.positive_pattern = re.compile(
            '|'.join(re.escape(word) for word in self.positive_indicators), 
            re.IGNORECASE
        )
        
        self.negative_pattern = re.compile(
            '|'.join(re.escape(word) for word in self.negative_indicators), 
            re.IGNORECASE
        )
    
    def analyze_comment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single comment.
        
        Args:
            text: Comment text to analyze
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        # TextBlob sentiment analysis
        blob = TextBlob(cleaned_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Custom indicator analysis
        positive_matches = len(self.positive_pattern.findall(text))
        negative_matches = len(self.negative_pattern.findall(text))
        
        # Determine overall sentiment
        if polarity > 0.1 or positive_matches > negative_matches:
            sentiment = 'positive'
        elif polarity < -0.1 or negative_matches > positive_matches:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Confidence score based on polarity magnitude and indicator matches
        confidence = abs(polarity) + (abs(positive_matches - negative_matches) * 0.1)
        confidence = min(confidence, 1.0)  # Cap at 1.0
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'confidence': confidence,
            'positive_indicators': positive_matches,
            'negative_indicators': negative_matches,
            'word_count': len(cleaned_text.split()),
            'cleaned_text': cleaned_text
        }
    
    def analyze_comments_batch(self, comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of comments.
        
        Args:
            comments: List of comment dictionaries
            
        Returns:
            List of comments with sentiment analysis added
        """
        analyzed_comments = []
        
        for comment in comments:
            # Perform sentiment analysis
            sentiment_data = self.analyze_comment(comment.get('text', ''))
            
            # Add sentiment data to comment
            enhanced_comment = comment.copy()
            enhanced_comment.update(sentiment_data)
            
            analyzed_comments.append(enhanced_comment)
        
        return analyzed_comments
    
    def get_sentiment_summary(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate sentiment summary for a collection of comments.
        
        Args:
            comments: List of analyzed comments
            
        Returns:
            Dictionary containing sentiment summary statistics
        """
        if not comments:
            return {}
        
        sentiments = [comment.get('sentiment', 'neutral') for comment in comments]
        polarities = [comment.get('polarity', 0) for comment in comments]
        subjectivities = [comment.get('subjectivity', 0) for comment in comments]
        
        sentiment_counts = Counter(sentiments)
        total_comments = len(comments)
        
        return {
            'total_comments': total_comments,
            'sentiment_distribution': {
                'positive': sentiment_counts.get('positive', 0),
                'negative': sentiment_counts.get('negative', 0),
                'neutral': sentiment_counts.get('neutral', 0)
            },
            'sentiment_percentages': {
                'positive': (sentiment_counts.get('positive', 0) / total_comments) * 100,
                'negative': (sentiment_counts.get('negative', 0) / total_comments) * 100,
                'neutral': (sentiment_counts.get('neutral', 0) / total_comments) * 100
            },
            'average_polarity': sum(polarities) / len(polarities),
            'average_subjectivity': sum(subjectivities) / len(subjectivities),
            'most_positive_comment': self._get_extreme_comment(comments, 'polarity', max),
            'most_negative_comment': self._get_extreme_comment(comments, 'polarity', min)
        }
    
    def get_temporal_sentiment(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment trends over time.
        
        Args:
            comments: List of analyzed comments with timestamps
            
        Returns:
            Dictionary containing temporal sentiment analysis
        """
        if not comments:
            return {}
        
        # Convert to DataFrame for easier time-based analysis
        df = pd.DataFrame(comments)
        
        # Convert published_at to datetime
        df['published_at'] = pd.to_datetime(df['published_at'])
        df = df.sort_values('published_at')
        
        # Group by date and calculate daily sentiment
        daily_sentiment = df.groupby(df['published_at'].dt.date).agg({
            'sentiment': lambda x: Counter(x).most_common(1)[0][0],
            'polarity': 'mean',
            'id': 'count'
        }).rename(columns={'id': 'comment_count'})
        
        return {
            'daily_sentiment_trend': daily_sentiment.to_dict(),
            'sentiment_over_time': df[['published_at', 'sentiment', 'polarity']].to_dict('records')
        }
    
    def _clean_text(self, text: str) -> str:
        """
        Clean comment text for better analysis.
        
        Args:
            text: Raw comment text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags but keep the content
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep emojis and basic punctuation
        text = re.sub(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF.,!?]', '', text)
        
        return text.strip()
    
    def _get_extreme_comment(self, comments: List[Dict[str, Any]], field: str, func) -> Dict[str, Any]:
        """
        Get comment with extreme value for a given field.
        
        Args:
            comments: List of comments
            field: Field to find extreme value for
            func: Function to apply (min or max)
            
        Returns:
            Comment with extreme value
        """
        if not comments:
            return {}
        
        extreme_comment = func(comments, key=lambda x: x.get(field, 0))
        return {
            'text': extreme_comment.get('text', ''),
            'author': extreme_comment.get('author', ''),
            'value': extreme_comment.get(field, 0),
            'published_at': extreme_comment.get('published_at', '')
        }
    
    def get_emotion_analysis(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform basic emotion analysis on comments.
        
        Args:
            comments: List of analyzed comments
            
        Returns:
            Dictionary containing emotion analysis
        """
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'thrilled', 'delighted', '😊', '😃', '😆', '🎉'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'pissed', '😡', '😠', '🤬'],
            'sadness': ['sad', 'depressed', 'disappointed', 'upset', '😢', '😭', '😞'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', '😰', '😨'],
            'surprise': ['surprised', 'shocked', 'amazed', 'wow', 'omg', '😲', '😮'],
            'disgust': ['disgusted', 'gross', 'awful', 'terrible', '🤮', '😷']
        }
        
        emotion_counts = {emotion: 0 for emotion in emotion_keywords}
        
        for comment in comments:
            text = comment.get('text', '').lower()
            
            for emotion, keywords in emotion_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in text:
                        emotion_counts[emotion] += 1
                        break
        
        total_emotional_comments = sum(emotion_counts.values())
        
        return {
            'emotion_distribution': emotion_counts,
            'total_emotional_comments': total_emotional_comments,
            'emotion_percentages': {
                emotion: (count / total_emotional_comments) * 100 if total_emotional_comments > 0 else 0
                for emotion, count in emotion_counts.items()
            }
        }
