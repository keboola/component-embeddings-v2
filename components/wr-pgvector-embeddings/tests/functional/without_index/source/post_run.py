from datadirtest import TestDataDir


def run(context: TestDataDir):
    """Verify that data was correctly written to pgvector."""
    # Get connection from context
    conn = context.context_parameters['pg_conn']

    # Use connection for verification
    with conn:
        with conn.cursor() as cur:
            # list all vector tables
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;")
            tables = cur.fetchall()
            # check if exists two tables
            assert len(tables) == 2, f"Expected 2 tables, found {len(tables)}"
            print("Tables in the database:")

            # check if the first table (collection) has one rows
            table = tables[0]
            cur.execute(f"SELECT COUNT(*) FROM {table[0]};")
            count = cur.fetchone()[0]
            assert count == 1, f"Table {table[0]} has wrong count of rows ({count} instead of 1)"
            print(f"Table {table[0]} has {count} rows")

            # check if the first table (embeddings) has five rows
            table = tables[1]
            cur.execute(f"SELECT COUNT(*) FROM {table[0]};")
            count = cur.fetchone()[0]
            assert count == 5, f"Table {table[0]} has wrong count of rows ({count} instead of 5)"
            print(f"Table {table[0]} has {count} rows")
