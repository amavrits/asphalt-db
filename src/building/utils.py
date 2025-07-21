import psycopg
from psycopg.errors import DuplicateDatabase


def create_database_if_not_exists(dbname, db_admin_config):
    with psycopg.connect(**db_admin_config, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            exists = cur.fetchone()
            if not exists:
                cur.execute(f'CREATE DATABASE "{dbname}";')
                print(f"Database '{dbname}' created.")
            else:
                print(f"Database '{dbname}' already exists.")


if __name__ == "__main__":

    pass