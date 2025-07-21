import psycopg
from src.db_builder.schema import create_tables, create_database_if_not_exists
from src.config import DB_CONFIG


def main():

    admin_config = DB_CONFIG.copy()
    admin_config["dbname"] = "postgres"

    create_database_if_not_exists(DB_CONFIG["dbname"], admin_config)

    with psycopg.connect(**DB_CONFIG) as conn:
        create_tables(conn)
        print("Tables created successfully.")


if __name__ == "__main__":

    main()
