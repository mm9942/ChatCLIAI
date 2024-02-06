import sqlite3
import os
from datetime import datetime

class DB:
    def __init__(self, db_filename="ai.db"):
        self.connection = sqlite3.connect(db_filename, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        with self.connection:
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER,
                    user_message TEXT,
                    ai_response TEXT,
                    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(chat_id) REFERENCES chats(id)
                )
            ''')
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT,
                    content TEXT
                )
            ''')
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text_id INTEGER,
                    embedding BLOB,
                    FOREIGN KEY(text_id) REFERENCES chat_messages(id)
                )
            ''')

    def add_chat(self, chat_name):
        with self.connection:
            cursor = self.connection.execute('''
                INSERT INTO chats (name) VALUES (?)
            ''', (chat_name,))
            self.connection.commit()
            return cursor.lastrowid

    def get_all_chats(self):
        with self.connection:
            cursor = self.connection.execute('SELECT id, name FROM chats')
            return cursor.fetchall()

    def add_chat_message(self, chat_id, user_message, ai_response):
        with self.connection:
            cursor = self.connection.execute('''
                INSERT INTO chat_messages (chat_id, user_message, ai_response) VALUES (?, ?, ?)
            ''', (chat_id, user_message, ai_response))
            self.connection.commit()
            return cursor.lastrowid

    def get_chat_messages(self, chat_id):
        with self.connection:
            cursor = self.connection.execute('SELECT user_message, ai_response FROM chat_messages WHERE chat_id = ?', (chat_id,))
            return cursor.fetchall()

    def save_document(self, path: str, content: str):
        with self.connection:
            cursor = self.connection.execute('''
                INSERT INTO documents (path, content) VALUES (?, ?)
            ''', (path, content))
            self.connection.commit()
            return cursor.lastrowid

    def load_document(self, path: str):
        with self.connection:
            cursor = self.connection.execute('SELECT content FROM documents WHERE path = ?', (path,))
            row = cursor.fetchone()
            return row[0] if row else None

    def save_embedding(self, text_id: int, embedding: bytes):
        with self.connection:
            self.connection.execute('''
                INSERT INTO embeddings (text_id, embedding) VALUES (?, ?)
            ''', (text_id, sqlite3.Binary(embedding)))
            self.connection.commit()

    def load_embedding(self, text_id: int):
        with self.connection:
            cursor = self.connection.execute('SELECT embedding FROM embeddings WHERE text_id = ?', (text_id,))
            row = cursor.fetchone()
            return row[0] if row else None

    def __del__(self):
        self.connection.close()