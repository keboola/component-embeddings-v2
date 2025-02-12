"""Script to fetch data from PGVector database."""
import asyncio
import json
import sys
from typing import Any, Dict, List, Tuple

import psycopg
from psycopg.rows import dict_row


async def truncate_pgvector_data(
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "vectordb",
        user: str = "postgres",
        password: str = "postgres",
        collection_table: str = "langchain_pg_collection",
        embedding_table: str = "langchain_pg_embedding"
) -> None:
    """Truncate (delete all data) from PGVector tables."""

    # Connect to database
    conn_string = f"host={host} port={port} dbname={dbname} user={user} password={password}"

    async with await psycopg.AsyncConnection.connect(
            conn_string
    ) as aconn:
        async with aconn.cursor() as acur:
            # Check and truncate collection table
            await acur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (collection_table,))

            exists = await acur.fetchone()
            if exists and exists[0]:
                print(f"\nTruncating table {collection_table}...")
                await acur.execute(f"TRUNCATE TABLE {collection_table} CASCADE")
                print(f"Table {collection_table} truncated successfully")
            else:
                print(f"Table {collection_table} does not exist!")

            # Check and truncate embedding table
            await acur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (embedding_table,))

            exists = await acur.fetchone()
            if exists and exists[0]:
                print(f"\nTruncating table {embedding_table}...")
                await acur.execute(f"TRUNCATE TABLE {embedding_table} CASCADE")
                print(f"Table {embedding_table} truncated successfully")
            else:
                print(f"Table {embedding_table} does not exist!")

            await aconn.commit()
            print("\nAll data has been deleted successfully!")


async def fetch_pgvector_data(
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "vectordb",
        user: str = "postgres",
        password: str = "postgres",
        collection_table: str = "langchain_pg_collection",
        embedding_table: str = "langchain_pg_embedding"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fetch all data from PGVector tables."""

    # Connect to database
    conn_string = f"host={host} port={port} dbname={dbname} user={user} password={password}"

    async with await psycopg.AsyncConnection.connect(
            conn_string,
            row_factory=dict_row
    ) as aconn:
        async with aconn.cursor() as acur:
            collection_data = []
            embedding_data = []

            # Fetch collection data
            print(f"\nFetching data from {collection_table}:")
            print("=" * 80)

            # Check if collection table exists
            await acur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (collection_table,))

            exists = await acur.fetchone()
            if exists and exists['exists']:
                # Show table structure
                await acur.execute(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = %s
                """, (collection_table,))
                columns = await acur.fetchall()
                print("\nCollection table structure:")
                for col in columns:
                    print(f"- {col['column_name']}: {col['data_type']}")

                # Fetch data
                await acur.execute(f"""
                    SELECT *
                    FROM {collection_table}
                    LIMIT 100
                """)

                collection_data = await acur.fetchall()
                print(f"\nFetched {len(collection_data)} records from collection table")
            else:
                print(f"Table {collection_table} does not exist!")

            # Fetch embedding data
            print(f"\nFetching data from {embedding_table}:")
            print("=" * 80)

            # Check if embedding table exists
            await acur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (embedding_table,))

            exists = await acur.fetchone()
            if exists and exists['exists']:
                # Show table structure
                await acur.execute(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = %s
                """, (embedding_table,))
                columns = await acur.fetchall()
                print("\nEmbedding table structure:")
                for col in columns:
                    print(f"- {col['column_name']}: {col['data_type']}")

                # Fetch data
                await acur.execute(f"""
                    SELECT *
                    FROM {embedding_table}
                    LIMIT 100
                """)

                embedding_data = await acur.fetchall()
                print(f"\nFetched {len(embedding_data)} records from embedding table")
            else:
                print(f"Table {embedding_table} does not exist!")

            # Process the data
            processed_collection = []
            for row in collection_data:
                processed_row = {}
                for key, value in row.items():
                    if isinstance(value, str) and (key == 'metadata' or key.endswith('_metadata')):
                        try:
                            processed_row[key] = json.loads(value)
                        except json.JSONDecodeError:
                            processed_row[key] = value
                    else:
                        processed_row[key] = value
                processed_collection.append(processed_row)

            processed_embeddings = []
            for row in embedding_data:
                processed_row = {}
                for key, value in row.items():
                    if isinstance(value, str) and (key == 'metadata' or key.endswith('_metadata')):
                        try:
                            processed_row[key] = json.loads(value)
                        except json.JSONDecodeError:
                            processed_row[key] = value
                    else:
                        processed_row[key] = value
                processed_embeddings.append(processed_row)

            return processed_collection, processed_embeddings


def print_usage():
    """Print script usage."""
    print("""
Usage: python fetch_pgvector_data.py [command]

Commands:
  fetch    Fetch and display data from PGVector tables (default)
  delete   Delete all data from PGVector tables
  help     Show this help message
    """)


async def main():
    """Main function."""
    try:
        # Parse command line arguments
        command = "fetch"  # default command
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()

        if command == "help":
            print_usage()
            return

        if command == "delete":
            # Ask for confirmation
            confirm = input("Are you sure you want to delete ALL data from the vector store? (yes/no): ")
            if confirm.lower() != "yes":
                print("Operation cancelled.")
                return

            await truncate_pgvector_data(
                host="localhost",
                port=5432,
                dbname="vectordb",
                user="postgres",
                password="postgres"
            )
            return

        if command == "fetch":
            # Fetch data
            collection_data, embedding_data = await fetch_pgvector_data(
                host="localhost",
                port=5432,
                dbname="vectordb",
                user="postgres",
                password="postgres"
            )

            # Print collection results
            if collection_data:
                print("\nCollection Data:")
                print("=" * 80)
                for row in collection_data:
                    print("\nRecord:")
                    for key, value in row.items():
                        print(f"{key}: {value}")
                    print("-" * 80)

            # Print embedding results
            if embedding_data:
                print("\nEmbedding Data:")
                print("=" * 80)
                for row in embedding_data:
                    print("\nRecord:")
                    for key, value in row.items():
                        if key == 'embedding':
                            print(f"{key}: [first 5 values: {value[:5]}...]")
                        else:
                            print(f"{key}: {value}")
                    print("-" * 80)
        else:
            print(f"Unknown command: {command}")
            print_usage()

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
