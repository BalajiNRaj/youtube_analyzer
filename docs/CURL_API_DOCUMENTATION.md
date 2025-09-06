# 🚀 YouTube Comments Vector Database API - curl Examples

## 📋 Quick Start

### 1. Start API Server
```bash
cd "/Users/balajin/Documents/Workspace/youtube analyzer"
python api_server.py
```

Server runs at: **http://localhost:8000**

### 2. Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🎯 API Endpoints with curl Examples

### **1. API Information & Health**

#### Get API Information
```bash
curl -X GET "http://localhost:8000/" \
  -H "accept: application/json"
```

#### Health Check
```bash
curl -X GET "http://localhost:8000/health" \
  -H "accept: application/json"
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-04T15:30:00",
  "chromadb_status": "connected",
  "total_collections": 3,
  "embedding_model": "intfloat/multilingual-e5-large"
}
```

---

### **2. Video Analysis & Storage**

#### Analyze YouTube Video
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -H "accept: application/json" \
  -d '{
    "video_url": "https://youtu.be/OzHvOB137k0",
    "max_comments": 100,
    "reset_collection": true
  }'
```

**Response:**
```json
{
  "video_id": "OzHvOB137k0",
  "status": "success",
  "message": "Successfully analyzed and stored 100 comments",
  "analysis_summary": {
    "total_comments": 100,
    "vector_storage": {
      "collection_name": "comments_OzHvOB137k0",
      "embedding_model": "intfloat/multilingual-e5-large",
      "total_comments_stored": 100
    }
  }
}
```

---

### **3. Semantic Search**

#### Basic Search - Tamil
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -H "accept: application/json" \
  -d '{
    "query": "படிச்சிருந்தா தெரிஞ்சிருக்கும்",
    "video_id": "OzHvOB137k0",
    "n_results": 5
  }'
```

#### Basic Search - English
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -H "accept: application/json" \
  -d '{
    "query": "great explanation",
    "video_id": "OzHvOB137k0",
    "n_results": 5
  }'
```

#### Basic Search - Tanglish (Tamil-English Mix)
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -H "accept: application/json" \
  -d '{
    "query": "semma video bro",
    "video_id": "OzHvOB137k0",
    "n_results": 10
  }'
```

**Search Response:**
```json
{
  "query": "semma video bro",
  "video_id": "OzHvOB137k0",
  "total_results": 10,
  "search_timestamp": "2025-09-04T15:35:00",
  "results": [
    {
      "id": "comment_123",
      "similarity_score": 0.892,
      "original_text": "semma video bro! அருமையான explanation",
      "cleaned_text": "semma video bro அருமையான explanation",
      "author": "@user123",
      "likes": 15,
      "replies_count": 3,
      "timestamp": "2025-08-31T08:11:14Z",
      "engagement_score": 21.0
    }
  ]
}
```

---

### **4. Advanced Search with Filters**

#### Search Popular Questions
```bash
curl -X POST "http://localhost:8000/search/advanced" \
  -H "Content-Type: application/json" \
  -H "accept: application/json" \
  -d '{
    "query": "how to explain",
    "video_id": "OzHvOB137k0",
    "n_results": 5,
    "min_likes": 3,
    "is_question": true,
    "has_replies": true
  }'
```

#### Search Recent Comments with Emojis
```bash
curl -X POST "http://localhost:8000/search/advanced" \
  -H "Content-Type: application/json" \
  -H "accept: application/json" \
  -d '{
    "query": "good video",
    "video_id": "OzHvOB137k0",
    "n_results": 10,
    "year": 2025,
    "month": 8,
    "has_emojis": true
  }'
```

#### Search High Engagement Comments
```bash
curl -X POST "http://localhost:8000/search/advanced" \
  -H "Content-Type: application/json" \
  -H "accept: application/json" \
  -d '{
    "query": "அருமையான விளக்கம்",
    "video_id": "OzHvOB137k0",
    "n_results": 10,
    "min_likes": 10,
    "has_replies": true
  }'
```

---

### **5. Collection Management**

#### List All Collections
```bash
curl -X GET "http://localhost:8000/collections" \
  -H "accept: application/json"
```

**Response:**
```json
{
  "total_collections": 3,
  "collections": [
    {
      "name": "comments_OzHvOB137k0",
      "video_id": "OzHvOB137k0",
      "created_at": "2025-09-04T10:00:00",
      "total_comments": 100
    },
    {
      "name": "comments_t865KSKdSDk",
      "video_id": "t865KSKdSDk",
      "created_at": "2025-09-04T11:00:00",
      "total_comments": 250
    }
  ],
  "timestamp": "2025-09-04T15:40:00"
}
```

#### Get Collection Statistics
```bash
curl -X GET "http://localhost:8000/collections/OzHvOB137k0/stats" \
  -H "accept: application/json"
```

**Response:**
```json
{
  "collection_name": "comments_OzHvOB137k0",
  "video_id": "OzHvOB137k0",
  "total_comments": 100,
  "embedding_model": "intfloat/multilingual-e5-large",
  "collection_metadata": {
    "video_id": "OzHvOB137k0",
    "created_at": "2025-09-04T10:00:00"
  }
}
```

#### Get Collection Details
```bash
curl -X GET "http://localhost:8000/collections/OzHvOB137k0" \
  -H "accept: application/json"
```

---

### **6. Sample Data Exploration**

#### Get Sample Comments
```bash
curl -X GET "http://localhost:8000/collections/OzHvOB137k0/sample?limit=5" \
  -H "accept: application/json"
```

**Response:**
```json
{
  "video_id": "OzHvOB137k0",
  "sample_size": 5,
  "sample_comments": [
    {
      "id": "comment_456",
      "similarity_score": 0.95,
      "original_text": "Great tutorial! Very helpful 👍",
      "author": "@learner123",
      "likes": 8,
      "replies_count": 2
    }
  ],
  "note": "These are sample comments from the collection."
}
```

---

### **7. Metadata-Only Filtering**

#### Filter High Engagement Comments
```bash
curl -X GET "http://localhost:8000/collections/OzHvOB137k0/filter?min_likes=5&has_replies=true&limit=10" \
  -H "accept: application/json"
```

#### Filter Questions with Emojis
```bash
curl -X GET "http://localhost:8000/collections/OzHvOB137k0/filter?is_question=true&has_emojis=true&limit=15" \
  -H "accept: application/json"
```

#### Filter Long Comments
```bash
curl -X GET "http://localhost:8000/collections/OzHvOB137k0/filter?is_long_comment=true&min_likes=2&limit=20" \
  -H "accept: application/json"
```

#### Filter by Engagement Range
```bash
curl -X GET "http://localhost:8000/collections/OzHvOB137k0/filter?min_likes=5&max_likes=50&has_replies=false&limit=25" \
  -H "accept: application/json"
```

**Metadata Filter Response:**
```json
{
  "video_id": "OzHvOB137k0",
  "filters_applied": {
    "likes": {"$gte": 5},
    "replies_count": {"$gt": 0}
  },
  "total_results": 15,
  "results": [
    {
      "id": "comment_789",
      "original_text": "அட்டகாசமான explanation! Keep it up 👌",
      "author": "@tamiluser",
      "likes": 12,
      "replies_count": 4,
      "timestamp": "2025-08-31T10:15:00Z"
    }
  ]
}
```

---

## 🔧 Advanced curl Usage

### **Pretty Print JSON Response**
```bash
curl -X GET "http://localhost:8000/health" \
  -H "accept: application/json" | python -m json.tool
```

### **Save Response to File**
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "great tutorial",
    "video_id": "OzHvOB137k0",
    "n_results": 20
  }' \
  -o search_results.json
```

### **Include Response Headers**
```bash
curl -X GET "http://localhost:8000/collections" \
  -H "accept: application/json" \
  -i
```

### **Verbose Output (Debug)**
```bash
curl -X GET "http://localhost:8000/health" \
  -H "accept: application/json" \
  -v
```

---

## 🎯 Real-World Usage Examples

### **1. Find Popular Tamil Comments**
```bash
curl -X POST "http://localhost:8000/search/advanced" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "அருமை",
    "video_id": "YOUR_VIDEO_ID",
    "n_results": 10,
    "min_likes": 5
  }' | python -m json.tool
```

### **2. Find Questions in Tanglish**
```bash
curl -X POST "http://localhost:8000/search/advanced" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how to bro",
    "video_id": "YOUR_VIDEO_ID",
    "n_results": 15,
    "is_question": true,
    "min_likes": 2
  }' | python -m json.tool
```

### **3. Analyze Multiple Search Patterns**
```bash
# Search for positive feedback
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "excellent work", "video_id": "YOUR_VIDEO_ID", "n_results": 10}' > positive_comments.json

# Search for questions
curl -X POST "http://localhost:8000/search/advanced" \
  -H "Content-Type: application/json" \
  -d '{"query": "doubt", "video_id": "YOUR_VIDEO_ID", "is_question": true, "n_results": 10}' > questions.json

# Search for engagement
curl -X GET "http://localhost:8000/collections/YOUR_VIDEO_ID/filter?min_likes=10&has_replies=true" > high_engagement.json
```

---

## 🛠️ Testing & Debugging

### **Test API Availability**
```bash
curl -f http://localhost:8000/health || echo "API is down"
```

### **Test with Different Languages**
```bash
# Tamil
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "நல்ல விளக்கம்", "video_id": "YOUR_VIDEO_ID", "n_results": 3}'

# English  
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "good explanation", "video_id": "YOUR_VIDEO_ID", "n_results": 3}'

# Tanglish
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "super video bro", "video_id": "YOUR_VIDEO_ID", "n_results": 3}'
```

### **Error Testing**
```bash
# Test invalid video ID
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "video_id": "INVALID_ID", "n_results": 5}'

# Test collection not found
curl -X GET "http://localhost:8000/collections/nonexistent/stats"
```

---

## 📊 Response Status Codes

| Status Code | Meaning | Example |
|-------------|---------|---------|
| 200 | Success | Search results found |
| 404 | Not Found | Collection/Video not found |
| 401 | Unauthorized | YouTube API authentication failed |
| 400 | Bad Request | Invalid video URL |
| 500 | Server Error | Internal processing error |

---

## 🚀 Batch Operations

### **Sequential Analysis of Multiple Videos**
```bash
#!/bin/bash
VIDEOS=("OzHvOB137k0" "t865KSKdSDk" "VIDEO_ID_3")

for video in "${VIDEOS[@]}"; do
  echo "Analyzing video: $video"
  curl -X POST "http://localhost:8000/analyze" \
    -H "Content-Type: application/json" \
    -d "{\"video_url\": \"https://youtu.be/$video\", \"max_comments\": 100}"
  echo "Completed: $video"
  sleep 5  # Wait between requests
done
```

### **Batch Search Queries**
```bash
#!/bin/bash
VIDEO_ID="YOUR_VIDEO_ID"
QUERIES=("அருமை" "great" "semma" "doubt" "how to")

for query in "${QUERIES[@]}"; do
  echo "Searching for: $query"
  curl -X POST "http://localhost:8000/search" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"$query\", \"video_id\": \"$VIDEO_ID\", \"n_results\": 5}" \
    > "results_${query// /_}.json"
done
```

---

## 🔐 Production Considerations

### **Add Timeouts**
```bash
curl --max-time 30 -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://youtu.be/VIDEO_ID", "max_comments": 1000}'
```

### **Retry Logic**
```bash
#!/bin/bash
for i in {1..3}; do
  if curl -f "http://localhost:8000/health"; then
    echo "API is healthy"
    break
  else
    echo "Attempt $i failed, retrying..."
    sleep 5
  fi
done
```

---

## 📝 Quick Reference

**Base URL**: `http://localhost:8000`

**Key Endpoints**:
- `GET /health` - Health check
- `POST /analyze` - Analyze video
- `POST /search` - Semantic search  
- `POST /search/advanced` - Advanced search
- `GET /collections` - List collections
- `GET /collections/{video_id}/stats` - Collection stats
- `GET /collections/{video_id}/filter` - Metadata filtering

**Supported Languages**: Tamil, English, Tanglish (Tamil-English mix)

**Embedding Model**: `intfloat/multilingual-e5-large` (1024 dimensions)

---

This documentation provides comprehensive curl examples for all API endpoints. The API supports multilingual semantic search with advanced filtering capabilities! 🎉
