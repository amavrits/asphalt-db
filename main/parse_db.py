from src.config import DB_CONFIG
import psycopg


def main():

    # data = parse_file("data/raw_data.txt")

    # Example: connect to PostgreSQL
    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            print("Connected to database.")


if __name__ == "__main__":

    main()

