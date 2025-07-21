import os
from peewee import PostgresqlDatabase, Model, CharField, ForeignKeyField
import psycopg
from dotenv import load_dotenv
from pathlib import Path
from src.config import DB_CONFIG


if __name__ == "__main__":

    # -----------------------
    # Load environment variables
    # -----------------------

    SCRIPT_DIR = Path(__file__).parent
    dotenv_path = SCRIPT_DIR.parent / ".env"
    load_dotenv(dotenv_path)

    # -----------------------
    # Step 1: Create the database if it doesn't exist
    # -----------------------

    admin_conn = psycopg.connect(
        dbname="postgres",  # connect to default DB to create your target DB
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        autocommit=True
    )

    with admin_conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_CONFIG["database"],))
        exists = cur.fetchone()
        if not exists:
            cur.execute(f"CREATE DATABASE {DB_CONFIG['database']}")
            print(f"Database '{DB_CONFIG['database']}' created.")
        else:
            print(f"Database '{DB_CONFIG['database']}' already exists.")

    admin_conn.close()

    # -----------------------
    # Step 2: Connect via Peewee and define model
    # -----------------------

    db = PostgresqlDatabase(
        database=DB_CONFIG["database"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"]
    )


    class BaseModel(Model):
        class Meta:
            database = db


    class User(BaseModel):
        name = CharField()
        email = CharField(unique=True)


    class UserPet(BaseModel):
        owner = ForeignKeyField(User, backref='pets')
        name = CharField()
        type = CharField()


    # -----------------------
    # Step 3: Create table and populate data
    # -----------------------

    db.connect()
    db.create_tables([User, UserPet], safe=True)

    users = [
        {"name": "Alice", "email": "alice@example.com"},
        {"name": "Bob", "email": "bob@example.com"},
        {"name": "Charlie", "email": "charlie@example.com"},
    ]

    for user_data in users:
        User.get_or_create(**user_data)

    alice, _ = User.get_or_create(name="Alice", email="alice@example.com")
    bob, _ = User.get_or_create(name="Bob", email="bob@example.com")

    UserPet.get_or_create(name="Whiskers", type="cat", owner=alice)
    UserPet.get_or_create(name="Fido", type="dog", owner=alice)
    UserPet.get_or_create(name="Goldie", type="fish", owner=bob)

    print("Users inserted successfully!")
    db.close()