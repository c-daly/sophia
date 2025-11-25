"""Example client demonstrating media ingestion API usage.

This example shows how to:
1. Upload an image file to the media ingestion service
2. List all media samples with filtering
3. Retrieve a specific media sample by ID
4. Link a media sample to a simulation
"""

import requests
from pathlib import Path
from PIL import Image
import io

# Configuration
API_BASE_URL = "http://localhost:8000"
API_TOKEN = "dev-token-change-in-production"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}


def create_sample_image() -> bytes:
    """Create a sample image for testing."""
    img = Image.new("RGB", (800, 600), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def ingest_image(image_data: bytes, filename: str, question: str = None):
    """Upload an image to the media ingestion service.

    Args:
        image_data: Image file bytes
        filename: Name for the uploaded file
        question: Optional perception question associated with the upload

    Returns:
        Response JSON containing sample metadata
    """
    url = f"{API_BASE_URL}/ingest/media"

    files = {"file": (filename, image_data, "image/jpeg")}

    data = {"media_type": "image"}

    if question:
        data["question"] = question

    response = requests.post(url, files=files, data=data, headers=HEADERS)
    response.raise_for_status()

    return response.json()


def list_media_samples(media_type: str = None, limit: int = 50, offset: int = 0):
    """List media samples with optional filtering.

    Args:
        media_type: Filter by media type (image, video, audio)
        limit: Maximum number of results
        offset: Number of results to skip

    Returns:
        Response JSON containing paginated sample list
    """
    url = f"{API_BASE_URL}/media/samples"

    params = {"limit": limit, "offset": offset}

    if media_type:
        params["media_type"] = media_type

    response = requests.get(url, params=params, headers=HEADERS)
    response.raise_for_status()

    return response.json()


def get_media_sample(sample_id: str):
    """Retrieve a specific media sample by ID.

    Args:
        sample_id: Sample identifier

    Returns:
        Response JSON containing sample metadata and usage count
    """
    url = f"{API_BASE_URL}/media/samples/{sample_id}"

    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    return response.json()


def main():
    """Run the media ingestion example workflow."""
    print("=" * 60)
    print("Media Ingestion Example")
    print("=" * 60)

    # Step 1: Create and upload an image
    print("\n1. Creating sample image...")
    image_data = create_sample_image()
    print(f"   Created image: {len(image_data)} bytes")

    print("\n2. Uploading image to media ingestion service...")
    ingest_result = ingest_image(
        image_data=image_data,
        filename="demo_image.jpg",
        question="What colors are present in this image?",
    )

    sample_id = ingest_result["sample_id"]
    print(f"   ✓ Upload successful!")
    print(f"   Sample ID: {sample_id}")
    print(f"   Media Type: {ingest_result['media_type']}")
    print(f"   File Path: {ingest_result['file_path']}")
    print(f"   File Size: {ingest_result['file_size']} bytes")
    print(f"   Timestamp: {ingest_result['timestamp']}")
    print(f"   Neo4j Node: {ingest_result['neo4j_node_id']}")

    if ingest_result.get("metadata"):
        metadata = ingest_result["metadata"]
        print(f"   Metadata:")
        print(f"     - Width: {metadata.get('width')}")
        print(f"     - Height: {metadata.get('height')}")
        print(f"     - Format: {metadata.get('format')}")

    # Step 2: List all image samples
    print("\n3. Listing all image samples...")
    list_result = list_media_samples(media_type="image", limit=10)

    print(f"   ✓ Found {list_result['total']} image samples")
    print(f"   Showing {len(list_result['samples'])} samples:")

    for idx, sample in enumerate(list_result["samples"], 1):
        print(f"   {idx}. {sample['sample_id']}")
        print(f"      - Type: {sample['media_type']}")
        print(f"      - Size: {sample['file_size']} bytes")
        print(f"      - Simulations: {sample['simulation_count']}")
        print(f"      - Uploaded: {sample['timestamp']}")

    # Step 3: Retrieve the specific sample we just uploaded
    print(f"\n4. Retrieving sample {sample_id}...")
    sample_detail = get_media_sample(sample_id)

    print(f"   ✓ Sample details:")
    print(f"   Sample ID: {sample_detail['sample_id']}")
    print(f"   Media Type: {sample_detail['media_type']}")
    print(f"   File Path: {sample_detail['file_path']}")
    print(f"   File Size: {sample_detail['file_size']} bytes")
    print(f"   Simulations Using This Sample: {sample_detail['simulation_count']}")

    if sample_detail.get("metadata"):
        metadata = sample_detail["metadata"]
        print(f"   Metadata:")
        for key, value in metadata.items():
            if value is not None:
                print(f"     - {key}: {value}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. Sophia service is running (docker-compose up -d)")
        print("2. API token is correct")
        print("3. Service is accessible at http://localhost:8000")
