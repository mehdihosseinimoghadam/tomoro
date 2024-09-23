import sqlite3
from src.config import DB_NAME
import pickle


def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    return conn


def create_database():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings
                 (id TEXT PRIMARY KEY, pre_embedding BLOB, post_embedding BLOB, content TEXT)''')
    conn.commit()
    conn.close()


def save_embeddings(embeddings_data):
    conn = get_db_connection()
    c = conn.cursor()
    for item in embeddings_data:
        c.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)", item)
    conn.commit()
    conn.close()


def load_embeddings():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM embeddings")
    rows = c.fetchall()
    conn.close()
    return [(row[0], pickle.loads(row[1]), pickle.loads(row[2]), row[3]) for row in rows]
