
import psycopg2
connection = psycopg2.connect(
    database="vector_db",
    user="postgres",
    password="postgres",
    host="172.17.0.2",
    port=5432
)

from sqlalchemy import create_engine

CONNECTION_STRING = "postgresql+psycopg2://postgres:postgres@localhost:5432/vector_db"
engine = create_engine(CONNECTION_STRING)

# Test connection
with engine.connect() as connection:
    result = connection.execute("SELECT version();")
    print(result.fetchone())