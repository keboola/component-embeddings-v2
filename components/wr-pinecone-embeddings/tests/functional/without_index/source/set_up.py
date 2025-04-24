import os
import time

from pinecone.grpc import PineconeGRPC, GRPCClientConfig
from pinecone import ServerlessSpec
from datadirtest import TestDataDir


def run(context: TestDataDir):
    """Set up database connection and store it in context."""
    # Prepare connection parameters
    print(f"Connection to Pinecone...{os.getenv('PINECONE_CONTROLLER_HOST')}")
    pc = PineconeGRPC(
        api_key="pclocal",
        host=os.getenv('PINECONE_CONTROLLER_HOST')
    )
    print("Connection to Pinecone established")

    # Wait for index to be ready
    print(f"Waiting for index list is available...")
    start_time = time.time()
    timeout = 120  # seconds
    while True:
        try:
            indexes = pc.list_indexes()
            if len(indexes) >= 0:
                print(f"Indexes are available")
                break
        except Exception as e:
            # Handle potential exceptions during describe_index call if needed
            print(f"Error listing indexes: {e}. Retrying...")

        if time.time() - start_time > timeout:
            raise TimeoutError(f"Indexes are not ready within {timeout} seconds.")

        time.sleep(10)  # Wait 10 seconds before checking again

    if pc.has_index(os.getenv("PINECONE_INDEX_NAME")):
        pc.delete_index(
            name=os.getenv("PINECONE_INDEX_NAME")
        )
        print(f"Index {os.getenv('PINECONE_INDEX_NAME')} deleted")
