# Media Ingestion API

The media ingestion API enables uploading and managing media files (images, video, audio) for perception workflows in Sophia.

## Overview

Media files are:
1. Validated for type and size
2. Stored to disk in organized directories
3. Metadata extracted (dimensions, duration, etc.)
4. Indexed in Neo4j for querying and linking to simulations
5. **Automatically processed by JEPA runner** for physical world understanding
6. Embeddings stored in Milvus for semantic search
7. Available for reference in simulations via `media_sample_id`

## Processing Pipeline

```
Upload → Storage → Neo4j Index → JEPA Processing → Milvus Embeddings → Simulation Context
```

### JEPA Integration

When a media file is uploaded, the JEPA runner automatically:
- Generates a **visual embedding** (768-dim) capturing visual features
- Generates a **physics embedding** (768-dim) for physical property understanding
- Stores both embeddings in Milvus vector database
- Creates Neo4j relationships: `MediaSample -[:has_embedding]-> Embedding`

These embeddings enable:
- Cross-modal semantic search (match text descriptions to images)
- Visual context for simulations
- Grounding language in physical observations

See [JEPA_SIMULATION.md](./JEPA_SIMULATION.md) for details on media processing.

## API Endpoints

### POST `/ingest/media`

Upload a media file for processing.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Authentication: Bearer token required

**Parameters:**
- `file` (required): The media file to upload
- `media_type` (required): One of `image`, `video`, or `audio`
- `question` (optional): A perception question associated with the upload

**Supported File Types:**
- **Images**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`
- **Video**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`
- **Audio**: `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`

**Size Limit:** 100 MB per file

**Response (201 Created):**
```json
{
  "sample_id": "ms_abc123xyz",
  "media_type": "image",
  "file_path": "/app/media_storage/image/ms_abc123xyz.jpg",
  "file_size": 245678,
  "timestamp": "2025-01-15T10:30:00Z",
  "metadata": {
    "width": 1920,
    "height": 1080,
    "format": "JPEG"
  },
  "neo4j_node_id": "node456",
  "message": "Media sample ingested successfully"
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:8000/ingest/media \
  -H "Authorization: Bearer dev-token-change-in-production" \
  -F "file=@/path/to/image.jpg" \
  -F "media_type=image" \
  -F "question=What objects are visible?"
```

**Example (Python):**
```python
import requests

url = "http://localhost:8000/ingest/media"
headers = {"Authorization": "Bearer dev-token-change-in-production"}

files = {"file": ("photo.jpg", open("photo.jpg", "rb"), "image/jpeg")}
data = {
    "media_type": "image",
    "question": "Describe this scene"
}

response = requests.post(url, headers=headers, files=files, data=data)
result = response.json()
print(f"Sample ID: {result['sample_id']}")
```

---

### GET `/media/samples`

List media samples with optional filtering and pagination.

**Request:**
- Method: `GET`
- Authentication: Bearer token required

**Query Parameters:**
- `media_type` (optional): Filter by type (`image`, `video`, `audio`)
- `after_timestamp` (optional): Filter samples after this ISO timestamp
- `limit` (optional): Maximum results (1-100, default 50)
- `offset` (optional): Number of results to skip (default 0)

**Response (200 OK):**
```json
{
  "samples": [
    {
      "sample_id": "ms_abc123",
      "media_type": "image",
      "file_path": "/app/media_storage/image/ms_abc123.jpg",
      "file_size": 245678,
      "file_hash": "a1b2c3d4...",
      "timestamp": "2025-01-15T10:30:00Z",
      "simulation_count": 3,
      "metadata": {
        "width": 1920,
        "height": 1080,
        "format": "JPEG"
      }
    },
    ...
  ],
  "total": 42,
  "limit": 50,
  "offset": 0
}
```

**Example (curl):**
```bash
# List all images
curl -X GET "http://localhost:8000/media/samples?media_type=image&limit=10" \
  -H "Authorization: Bearer dev-token-change-in-production"

# List samples after a timestamp
curl -X GET "http://localhost:8000/media/samples?after_timestamp=2025-01-15T00:00:00Z" \
  -H "Authorization: Bearer dev-token-change-in-production"
```

**Example (Python):**
```python
import requests

url = "http://localhost:8000/media/samples"
headers = {"Authorization": "Bearer dev-token-change-in-production"}
params = {
    "media_type": "image",
    "limit": 20,
    "offset": 0
}

response = requests.get(url, headers=headers, params=params)
result = response.json()
print(f"Found {result['total']} samples")
```

---

### GET `/media/samples/{sample_id}`

Retrieve details for a specific media sample.

**Request:**
- Method: `GET`
- Authentication: Bearer token required

**Path Parameters:**
- `sample_id` (required): The sample identifier

**Response (200 OK):**
```json
{
  "sample_id": "ms_abc123",
  "media_type": "image",
  "file_path": "/app/media_storage/image/ms_abc123.jpg",
  "file_size": 245678,
  "file_hash": "a1b2c3d4...",
  "timestamp": "2025-01-15T10:30:00Z",
  "simulation_count": 3,
  "metadata": {
    "width": 1920,
    "height": 1080,
    "format": "JPEG"
  }
}
```

**Response (404 Not Found):**
```json
{
  "detail": "Media sample not found: ms_nonexistent"
}
```

**Example (curl):**
```bash
curl -X GET http://localhost:8000/media/samples/ms_abc123 \
  -H "Authorization: Bearer dev-token-change-in-production"
```

**Example (Python):**
```python
import requests

sample_id = "ms_abc123"
url = f"http://localhost:8000/media/samples/{sample_id}"
headers = {"Authorization": "Bearer dev-token-change-in-production"}

response = requests.get(url, headers=headers)
if response.status_code == 200:
    sample = response.json()
    print(f"Sample used in {sample['simulation_count']} simulations")
else:
    print(f"Sample not found: {response.json()['detail']}")
```

---

## Architecture

### Storage Structure

Media files are stored on disk in type-based directories:

```
media_storage/
├── image/
│   ├── ms_abc123.jpg
│   ├── ms_def456.png
│   └── ...
├── video/
│   ├── ms_ghi789.mp4
│   └── ...
└── audio/
    ├── ms_jkl012.mp3
    └── ...
```

### Neo4j Schema

Media samples are stored as nodes with the following properties:

```cypher
CREATE (m:MediaSample {
  sample_id: "ms_abc123",
  node_type: "media_sample",
  media_type: "image",
  file_path: "/app/media_storage/image/ms_abc123.jpg",
  file_size: 245678,
  file_hash: "a1b2c3d4...",
  timestamp: "2025-01-15T10:30:00Z",
  ingested_at: "2025-01-15T10:30:00Z",
  
  // Extracted metadata (flat properties)
  metadata_width: 1920,
  metadata_height: 1080,
  metadata_format: "JPEG",
  
  // Optional perception question
  question: "What objects are visible?"
})
```

Samples can be linked to simulations:

```cypher
MATCH (sample:MediaSample {sample_id: "ms_abc123"})
MATCH (sim:Simulation {simulation_id: "sim_xyz"})
CREATE (sample)-[:USED_IN]->(sim)
```

### Metadata Extraction

- **Images**: Width, height, format extracted using Pillow
- **Video**: Duration, frame rate, codec (TODO: requires ffmpeg-python)
- **Audio**: Duration, sample rate, channels (TODO: requires mutagen/ffmpeg-python)

### JEPA Integration

When media is ingested, a notification hook is triggered (currently logs only):

```python
# Future implementation will:
# 1. Publish sample to message queue
# 2. JEPA runner processes media for perception
# 3. Generates embeddings stored in Milvus
# 4. Creates predicted states in Neo4j
```

---

## Docker Setup

The media storage is mounted as a Docker volume for persistence:

```yaml
# docker-compose.yml
services:
  sophia:
    environment:
      - MEDIA_STORAGE_ROOT=/app/media_storage
    volumes:
      - ./volumes/media_storage:/app/media_storage
```

Start the service:

```bash
cd sophia
docker-compose up -d
```

Verify the service:

```bash
curl http://localhost:8000/health
```

---

## Complete Example Workflow

See `examples/media_ingest_example.py` for a complete Python client demonstrating:

1. Creating and uploading an image
2. Listing samples with filters
3. Retrieving sample details
4. Linking samples to simulations

Run the example:

```bash
cd sophia
python examples/media_ingest_example.py
```

Expected output:

```
============================================================
Media Ingestion Example
============================================================

1. Creating sample image...
   Created image: 12845 bytes

2. Uploading image to media ingestion service...
   ✓ Upload successful!
   Sample ID: ms_abc123xyz
   Media Type: image
   File Path: /app/media_storage/image/ms_abc123xyz.jpg
   File Size: 12845 bytes
   Timestamp: 2025-01-15T10:30:00Z
   Neo4j Node: node456
   Metadata:
     - Width: 800
     - Height: 600
     - Format: JPEG

3. Listing all image samples...
   ✓ Found 5 image samples
   ...

============================================================
Example completed successfully!
============================================================
```

---

## Error Handling

### Validation Errors (400 Bad Request)

```json
{
  "detail": "Invalid file extension .txt for media type image"
}
```

```json
{
  "detail": "File size 104857600 bytes exceeds maximum size 100MB"
}
```

### Service Unavailable (503)

```json
{
  "detail": "Media ingestion service not available"
}
```

Occurs when Neo4j or storage service fails to initialize.

### Not Found (404)

```json
{
  "detail": "Media sample not found: ms_nonexistent"
}
```

---

## Future Enhancements

- [ ] Video/audio metadata extraction (ffmpeg-python)
- [ ] Deduplication using file hashes
- [ ] Thumbnail generation for images/videos
- [ ] Direct JEPA processing integration
- [ ] WebRTC streaming support
- [ ] File watcher for batch ingestion
- [ ] Milvus embedding storage
- [ ] Sample deletion endpoint
- [ ] Batch upload support
