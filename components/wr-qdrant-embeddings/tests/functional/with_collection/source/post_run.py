import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from datadirtest import TestDataDir


def run(context: TestDataDir):
    """Set up database connection and store it in context."""
    client = QdrantClient(url=os.getenv("QDRANT_HOST"))
    print("Connected to Qdrant at", os.getenv("QDRANT_HOST"))

    # print available collections
    collections = [collection.name for collection in client.get_collections().collections]
    print("Available collections:", collections)

    # print vector count in collection
    collection_name = os.getenv("QDRANT_COLLECTION")
    if collection_name in collections:
        collection_info = client.get_collection(collection_name)
        assert collection_info.points_count == 5, (f"Expected 5 vectors in {collection_name},"
                                                   f" found {collection_info.points_count} points.")
        print(f"Collection {collection_name} already exists with {collection_info.points_count} points.")

    # create collection if it does not exist
    collection_name = os.getenv("QDRANT_COLLECTION")
    if collection_name in collections:
        # client.delete_collection(collection_name)
        print(f"Collection {collection_name} deleted.")
