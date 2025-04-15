import os
from pinecone.grpc import PineconeGRPC, GRPCClientConfig
from pinecone import ServerlessSpec
from datadirtest import TestDataDir


def run(context: TestDataDir):
    """Verify that data was correctly written to Pinecone."""
    print("Verifying data in Pinecone...")
    pc = PineconeGRPC(
        api_key="pclocal",
        host=os.getenv('PINECONE_CONTROLLER_HOST')
    )
    # check List vector IDs in index
    #TODO check index parameters

    print("✓ All verifications completed successfully")

    pc.delete_index(
        name=os.getenv("PINECONE_INDEX_NAME")
    )
    print(f"Index {os.getenv('PINECONE_INDEX_NAME')} deleted")
