import sqlite3
import faiss #_cpu
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredWordDocumentLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import getpass
import os
# import fitz

class DocLoad:
    def __init__(self, db_filename="ai.db"):
        self.connection = sqlite3.connect(db_filename)
        self.create_tables()

        # Initialize the OpenAIEmbeddings with the API key
        self.setup_api_key()
        self.embeddings = OpenAIEmbeddings()

        # Define the embedding function
        embedding_function = self.embeddings.embed_query

        # Create the FAISS index
        # Assuming embeddings have a dimensionality of 768
        index = faiss.IndexFlatL2(768)  # L2 distance flat index

        # Set up the document store
        # For simplicity, we'll just use a dictionary
        self.docstore = {}

        # Function to map from index IDs to document store IDs
        index_to_docstore_id = lambda x: x

        # Initialize the FAISS vector store
        self.vector_store = FAISS(
            embedding_function=embedding_function,
            index=index,
            docstore=self.docstore,
            index_to_docstore_id=index_to_docstore_id
        )

    def setup_api_key(self):
        self.api_key =  os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            self.api_key = getpass.getpass('OpenAI API Key:')

        if os.environ.get("OPENAI_API_KEY") is None:
            os.environ["OPENAI_API_KEY"] = self.api_key
            self.append_openai_key_to_shell()

    def append_openai_key_to_shell(self):
        euid = os.geteuid()

        if euid != 0:
            print(
                f"\n\nScript not started as root. Running sudo... Your euid is: {euid}"
            )
            args = ["sudo", sys.executable] + sys.argv + [os.environ]
            os.execlpe("sudo", *args)

        print(f"Running. Your euid is {euid}\n\n")

        shell_config_map = {
            "/bash": ".bashrc",
            "/zsh": ".zshrc",
            "/csh": ".cshrc",
            "/tcsh": ".tcshrc",
            "/ksh": ".kshrc",
            "/fish": ".config/fish/config.fish",
            "/mksh": ".mkshrc",
            "/yash": ".yashrc",
            "/rc": ".rcrc",
        }

        home_dir = os.getenv("HOME")

        for shell, config_filename in shell_config_map.items():
            config_file = os.path.join(home_dir, config_filename)

            if os.path.isfile(config_file):
                with open(config_file, "a") as file:
                    file.write(f"\nexport OPENAI_API_KEY={self.api_key}\n")
                return

        raise ValueError("No recognized shell configuration file was found.")

    def create_tables(self):
        with self.connection:
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    path TEXT,
                    content TEXT
                )
            ''')
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY,
                    text_id INTEGER,
                    embedding BLOB,
                    FOREIGN KEY(text_id) REFERENCES documents(id)
                )
            ''')

    def embed_and_store_message(self, chat_id, user_message, ai_response):
        # Embed the user message and AI response
        user_message_embedding = self.embed_text(user_message)
        ai_response_embedding = self.embed_text(ai_response)

        # Save the user message and AI response to the database
        message_id = self.db.add_chat_message(chat_id, user_message, ai_response)

        # Save the embeddings to the database
        self.db.save_embedding(message_id, user_message_embedding)
        self.db.save_embedding(message_id, ai_response_embedding)

        # Integrate with FAISS
        self.integrate_with_faiss(message_id, user_message_embedding)
        self.integrate_with_faiss(message_id, ai_response_embedding)


    def __del__(self):
        self.connection.close()

    def save_document(self, path: str, content: str):
        with self.connection:
            cursor = self.connection.execute('''
                INSERT INTO documents (path, content) VALUES (?, ?)
            ''', (path, content))
            return cursor.lastrowid

    def load_document(self, path: str):
        with self.connection:
            cursor = self.connection.execute('SELECT content FROM documents WHERE path = ?', (path,))
            row = cursor.fetchone()
            if row:
                return row[0]
            else:
                return None

    def save_embedding(self, text_id: int, embedding: bytes):
        with self.connection:
            self.connection.execute('''
                INSERT INTO embeddings (text_id, embedding) VALUES (?, ?)
            ''', (text_id, sqlite3.Binary(embedding)))

    def load_embedding(self, text_id: int):
        with self.connection:
            cursor = self.connection.execute('SELECT embedding FROM embeddings WHERE text_id = ?', (text_id,))
            row = cursor.fetchone()
            if row:
                return row[0]
            else:
                return None

    def embed_text(self, text):
        embeddings = self.embeddings.embed_text(text)
        return embeddings

    def integrate_with_faiss(self, text_id, embedding):
        self.vector_store.add_vector(text_id, embedding)

    def search_in_doc(self, query, documents):
        query_embedding = self.embeddings.embed_text(query)
        results = self.vector_store.search_vectors(query_embedding, num_results=10)
        return [(doc_id, self.load_document(doc_id)) for doc_id, _ in results]

    def load_and_split(self, path: str):
        content = self.load_document(path)
        if content:
            splitter = CharacterTextSplitter(max_length=1024)
            documents = splitter.split_text(content)
            return documents
        else:
            return []

    def load_and_split(self, path: str):
        loaded_docs = self.doc_loader.load_document_from_path(path)
        splitter = CharacterTextSplitter(max_length=1024)
        documents = []
        for doc in loaded_docs:
            documents.extend(splitter.split_text(doc.page_content))
        return documents

    def __call__(self, text: str):
        embeddings = self.embed_text(text)
        text_id = self.save_document("embedded_text", text)
        self.save_embedding(text_id, embeddings)
        self.integrate_with_faiss(text_id, embeddings)