import os
import psycopg2
from datadirtest import TestDataDir

def run(context: TestDataDir):
    """Set up database connection and store it in context."""
    # Prepare connection parameters
    pg_connection = {
        'host': os.getenv('POSTGRES_HOST'),
        'port': int(os.getenv('POSTGRES_PORT')),
        'user': os.getenv('POSTGRES_USER'),
        'password': os.getenv('POSTGRES_PASSWORD'),
        'dbname': os.getenv('POSTGRES_DB')
    }
    
    # Store in context
    context.context_parameters['pg_connection'] = pg_connection
    
    # Create connection and store it
    conn = psycopg2.connect(**pg_connection)
    context.context_parameters['pg_conn'] = conn
    
    # Initialization steps
    with conn:
        with conn.cursor() as cur:
            # Ensure pgvector extension is available
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("pgvector extension is ready") 