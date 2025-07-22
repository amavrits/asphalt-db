import psycopg
from peewee import PostgresqlDatabase, Model, CharField, ForeignKeyField


def create_db(db_config):

    admin_conn = psycopg.connect(
        dbname="postgres",  # connect to default DB to create your target DB
        user=db_config["user"],
        password=db_config["password"],
        host=db_config["host"],
        port=db_config["port"],
        autocommit=True
    )

    with admin_conn.cursor() as cur:

        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_config["database"],))
        exists = cur.fetchone()

        if not exists:
            cur.execute(f"CREATE DATABASE {db_config['database']}")
            print(f"Database '{db_config['database']}' created.")
        else:
            print(f"Database '{db_config['database']}' already exists.")

    admin_conn.close()


if __name__ == "__main__":

    pass

