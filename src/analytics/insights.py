"""
Insights Generator for YouTube comments analysis.
"""

from typing import List, Dict, Any, Tuple
import json
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns


class InsightGenerator:
    """
    Generate comprehensive insights from YouTube comments data.
    """
    
    def __init__(self):
        """Initialize insights generator."""
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }
    
    def generate_comprehensive_insights(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive insights from analyzed comments.
        
        Args:
            comments: List of analyzed comments with sentiment data
            
        Returns:
            Dictionary containing all insights
        """
        if not comments:
            return {"error": "No comments provided for analysis"}
        
        insights = {
            'overview': self._get_overview_stats(comments),
            'engagement_metrics': self._get_engagement_metrics(comments),
            'content_analysis': self._get_content_analysis(comments),
            'author_insights': self._get_author_insights(comments),
            'temporal_patterns': self._get_temporal_patterns(comments),
            'sentiment_insights': self._get_sentiment_insights(comments),
            'trending_topics': self._get_trending_topics(comments),
            'recommendations': self._generate_recommendations(comments)
        }
        
        return insights
    
    def _get_overview_stats(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overview statistics."""
        total_comments = len(comments)
        total_replies = len([c for c in comments if c.get('type') == 'reply'])
        total_top_level = total_comments - total_replies
        
        total_likes = sum(c.get('like_count', 0) for c in comments)
        avg_likes = total_likes / total_comments if total_comments > 0 else 0
        
        word_counts = [c.get('word_count', 0) for c in comments if c.get('word_count', 0) > 0]
        avg_word_count = sum(word_counts) / len(word_counts) if word_counts else 0
        
        return {
            'total_comments': total_comments,
            'top_level_comments': total_top_level,
            'replies': total_replies,
            'total_likes': total_likes,
            'average_likes_per_comment': round(avg_likes, 2),
            'average_word_count': round(avg_word_count, 1),
            'engagement_rate': round((total_likes / total_comments) if total_comments > 0 else 0, 2)
        }
    
    def _get_engagement_metrics(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze engagement patterns."""
        like_counts = [c.get('like_count', 0) for c in comments]
        reply_counts = [c.get('reply_count', 0) for c in comments if c.get('type') == 'comment']
        
        # Sort comments by engagement
        highly_liked = sorted(comments, key=lambda x: x.get('like_count', 0), reverse=True)[:10]
        most_replied = sorted(
            [c for c in comments if c.get('type') == 'comment'],
            key=lambda x: x.get('reply_count', 0), 
            reverse=True
        )[:10]
        
        return {
            'like_distribution': {
                'max_likes': max(like_counts) if like_counts else 0,
                'min_likes': min(like_counts) if like_counts else 0,
                'median_likes': np.median(like_counts) if like_counts else 0,
                'percentile_90': np.percentile(like_counts, 90) if like_counts else 0
            },
            'reply_distribution': {
                'max_replies': max(reply_counts) if reply_counts else 0,
                'median_replies': np.median(reply_counts) if reply_counts else 0,
                'total_threads_with_replies': len([c for c in reply_counts if c > 0])
            },
            'top_liked_comments': [
                {
                    'text': c.get('text', '')[:200] + '...' if len(c.get('text', '')) > 200 else c.get('text', ''),
                    'author': c.get('author', ''),
                    'likes': c.get('like_count', 0),
                    'sentiment': c.get('sentiment', 'neutral')
                }
                for c in highly_liked[:5]
            ],
            'most_replied_comments': [
                {
                    'text': c.get('text', '')[:200] + '...' if len(c.get('text', '')) > 200 else c.get('text', ''),
                    'author': c.get('author', ''),
                    'replies': c.get('reply_count', 0),
                    'sentiment': c.get('sentiment', 'neutral')
                }
                for c in most_replied[:5]
            ]
        }
    
    def _get_content_analysis(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze comment content patterns."""
        all_text = ' '.join([c.get('cleaned_text', '') for c in comments])
        words = [word.lower() for word in all_text.split() if len(word) > 2 and word.lower() not in self.stop_words]
        
        word_freq = Counter(words)
        common_words = word_freq.most_common(20)
        
        # Analyze comment lengths
        lengths = [len(c.get('text', '')) for c in comments]
        
        # Find questions and exclamations
        questions = len([c for c in comments if '?' in c.get('text', '')])
        exclamations = len([c for c in comments if '!' in c.get('text', '')])
        
        return {
            'most_common_words': common_words,
            'unique_words': len(set(words)),
            'total_words': len(words),
            'comment_length_stats': {
                'average_length': round(np.mean(lengths), 1) if lengths else 0,
                'median_length': np.median(lengths) if lengths else 0,
                'max_length': max(lengths) if lengths else 0,
                'short_comments': len([l for l in lengths if l < 50]),
                'long_comments': len([l for l in lengths if l > 200])
            },
            'content_patterns': {
                'questions': questions,
                'exclamations': exclamations,
                'question_percentage': round((questions / len(comments)) * 100, 1) if comments else 0,
                'exclamation_percentage': round((exclamations / len(comments)) * 100, 1) if comments else 0
            }
        }
    
    def _get_author_insights(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze author engagement patterns."""
        authors = [c.get('author', 'Unknown') for c in comments]
        author_stats = Counter(authors)
        
        # Calculate author engagement
        author_engagement = defaultdict(lambda: {'comments': 0, 'total_likes': 0, 'avg_sentiment': []})
        
        for comment in comments:
            author = comment.get('author', 'Unknown')
            author_engagement[author]['comments'] += 1
            author_engagement[author]['total_likes'] += comment.get('like_count', 0)
            if comment.get('polarity') is not None:
                author_engagement[author]['avg_sentiment'].append(comment.get('polarity', 0))
        
        # Calculate average sentiment for each author
        for author in author_engagement:
            sentiments = author_engagement[author]['avg_sentiment']
            author_engagement[author]['avg_sentiment'] = np.mean(sentiments) if sentiments else 0
            author_engagement[author]['avg_likes'] = (
                author_engagement[author]['total_likes'] / author_engagement[author]['comments']
                if author_engagement[author]['comments'] > 0 else 0
            )
        
        most_active = sorted(author_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_unique_authors': len(set(authors)),
            'most_active_authors': [
                {
                    'author': author,
                    'comment_count': count,
                    'avg_likes': round(author_engagement[author]['avg_likes'], 2),
                    'avg_sentiment': round(author_engagement[author]['avg_sentiment'], 3)
                }
                for author, count in most_active
            ],
            'engagement_distribution': {
                'single_comment_authors': len([count for count in author_stats.values() if count == 1]),
                'multiple_comment_authors': len([count for count in author_stats.values() if count > 1]),
                'very_active_authors': len([count for count in author_stats.values() if count >= 5])
            }
        }
    
    def _get_temporal_patterns(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in comments."""
        if not comments or not any(c.get('published_at') for c in comments):
            return {'error': 'No timestamp data available'}
        
        # Convert timestamps
        timestamps = []
        for comment in comments:
            if comment.get('published_at'):
                try:
                    timestamp = pd.to_datetime(comment['published_at'])
                    timestamps.append(timestamp)
                except:
                    continue
        
        if not timestamps:
            return {'error': 'Could not parse timestamp data'}
        
        df = pd.DataFrame({'timestamp': timestamps})
        
        # Analyze by hour of day
        df['hour'] = df['timestamp'].dt.hour
        hourly_counts = df['hour'].value_counts().sort_index()
        
        # Analyze by day of week
        df['day_of_week'] = df['timestamp'].dt.day_name()
        daily_counts = df['day_of_week'].value_counts()
        
        # Find peak activity periods
        peak_hour = hourly_counts.idxmax()
        peak_day = daily_counts.idxmax()
        
        return {
            'activity_by_hour': hourly_counts.to_dict(),
            'activity_by_day': daily_counts.to_dict(),
            'peak_activity': {
                'hour': int(peak_hour),
                'day': peak_day,
                'comments_at_peak_hour': int(hourly_counts[peak_hour]),
                'comments_on_peak_day': int(daily_counts[peak_day])
            },
            'time_span': {
                'first_comment': min(timestamps).isoformat(),
                'last_comment': max(timestamps).isoformat(),
                'total_days': (max(timestamps) - min(timestamps)).days
            }
        }
    
    def _get_sentiment_insights(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment patterns and insights."""
        sentiments = [c.get('sentiment', 'neutral') for c in comments]
        polarities = [c.get('polarity', 0) for c in comments if c.get('polarity') is not None]
        
        sentiment_counts = Counter(sentiments)
        
        # Find sentiment by engagement
        high_engagement = [c for c in comments if c.get('like_count', 0) > np.percentile([c.get('like_count', 0) for c in comments], 75)]
        high_engagement_sentiment = Counter([c.get('sentiment', 'neutral') for c in high_engagement])
        
        return {
            'overall_sentiment': {
                'distribution': dict(sentiment_counts),
                'dominant_sentiment': sentiment_counts.most_common(1)[0][0] if sentiment_counts else 'neutral'
            },
            'sentiment_strength': {
                'average_polarity': round(np.mean(polarities), 3) if polarities else 0,
                'polarity_std': round(np.std(polarities), 3) if polarities else 0,
                'strong_positive': len([p for p in polarities if p > 0.5]),
                'strong_negative': len([p for p in polarities if p < -0.5])
            },
            'engagement_sentiment_correlation': {
                'high_engagement_sentiment': dict(high_engagement_sentiment),
                'positive_engagement_ratio': (
                    high_engagement_sentiment.get('positive', 0) / len(high_engagement)
                    if high_engagement else 0
                )
            }
        }
    
    def _get_trending_topics(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify trending topics and themes."""
        all_text = ' '.join([c.get('cleaned_text', '') for c in comments])
        
        # Extract potential topics (words appearing frequently together)
        words = [word.lower() for word in all_text.split() 
                if len(word) > 3 and word.lower() not in self.stop_words]
        
        word_freq = Counter(words)
        trending_words = word_freq.most_common(15)
        
        # Simple topic extraction based on word co-occurrence
        topics = []
        for word, freq in trending_words[:10]:
            related_comments = [c for c in comments if word in c.get('cleaned_text', '').lower()]
            avg_sentiment = np.mean([c.get('polarity', 0) for c in related_comments if c.get('polarity') is not None])
            
            topics.append({
                'topic': word,
                'frequency': freq,
                'related_comments': len(related_comments),
                'average_sentiment': round(avg_sentiment, 3),
                'sentiment_label': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral'
            })
        
        return {
            'trending_words': trending_words,
            'topic_analysis': topics,
            'total_topics_identified': len(topics)
        }
    
    def _generate_recommendations(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate actionable recommendations based on analysis."""
        sentiment_counts = Counter([c.get('sentiment', 'neutral') for c in comments])
        total_comments = len(comments)
        
        recommendations = []
        
        # Sentiment-based recommendations
        positive_ratio = sentiment_counts.get('positive', 0) / total_comments
        negative_ratio = sentiment_counts.get('negative', 0) / total_comments
        
        if positive_ratio > 0.6:
            recommendations.append({
                'category': 'Sentiment',
                'priority': 'high',
                'recommendation': 'Excellent positive sentiment! Consider highlighting positive feedback and creating similar content.',
                'metric': f'{positive_ratio:.1%} positive comments'
            })
        elif negative_ratio > 0.4:
            recommendations.append({
                'category': 'Sentiment',
                'priority': 'high',
                'recommendation': 'High negative sentiment detected. Consider addressing common concerns in future content.',
                'metric': f'{negative_ratio:.1%} negative comments'
            })
        
        # Engagement recommendations
        avg_likes = np.mean([c.get('like_count', 0) for c in comments])
        if avg_likes < 1:
            recommendations.append({
                'category': 'Engagement',
                'priority': 'medium',
                'recommendation': 'Low engagement detected. Consider asking questions or encouraging interaction in your content.',
                'metric': f'Average {avg_likes:.1f} likes per comment'
            })
        
        # Content recommendations
        questions = len([c for c in comments if '?' in c.get('text', '')])
        if questions / total_comments > 0.2:
            recommendations.append({
                'category': 'Content',
                'priority': 'medium',
                'recommendation': 'Many questions in comments. Consider creating FAQ content or follow-up videos.',
                'metric': f'{questions} questions asked'
            })
        
        return {
            'recommendations': recommendations,
            'total_recommendations': len(recommendations)
        }
    
    def export_insights(self, insights: Dict[str, Any], format: str = 'json', filename: str = None) -> str:
        """
        Export insights to file.
        
        Args:
            insights: Insights dictionary
            format: Export format ('json', 'csv')
            filename: Output filename
            
        Returns:
            Filename of exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"youtube_insights_{timestamp}"
        
        if format.lower() == 'json':
            output_file = f"{filename}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(insights, f, indent=2, ensure_ascii=False, default=str)
        
        elif format.lower() == 'csv':
            # Flatten insights for CSV export
            flattened = self._flatten_dict(insights)
            df = pd.DataFrame([flattened])
            output_file = f"{filename}.csv"
            df.to_csv(output_file, index=False)
        
        return output_file
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                # Handle list of dictionaries by taking first item or count
                items.append((f"{new_key}_count", len(v)))
                if v:
                    items.extend(self._flatten_dict(v[0], f"{new_key}_first", sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
