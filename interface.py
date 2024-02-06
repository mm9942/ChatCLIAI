import getpass
import os
import argparse
from prompt_toolkit import PromptSession
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from db import DB
from docload import DocLoad

class Interface:
    def __init__(self):
        self.db = DB()
        self.doc_loader = DocLoad()
        self.parse_arguments()
        self.setup_api_key()
        self.setup_chat_model()

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-k", "--key", type=str, help="Stores the API key")
        parser.add_argument("-s", "--system", type=str, help="Modify the system's role")
        parser.add_argument("-a", "--ai", type=str, help="Set an AI message prompt")
        parser.add_argument("-m", "--model", type=str, help="Select the OpenAI API model you want to use", default="gpt-3.5-turbo")
        parser.add_argument("-v", "--verbose", action="store_true", help="Set the llm output to verbose")
        parser.add_argument("-t", "--temp", type=float, help="Select a temperature")
        parser.add_argument("--human", type=str, help="Add a human message prompt")
        parser.add_argument("-f", "--file", nargs='+', type=str, help="Specify the file path(s) to load and embed")
        parser.add_argument("-l", "--list", action="store_true", help="List the available chats")
        parser.add_argument("-c", "--chat", type=str, help="Specify the chat")
        parser.add_argument("-n", "--new", action="store_true", help="Start a new chat")
        self.args = parser.parse_args()
        
        # Set additional attributes based on the parsed arguments
        self.system = self.args.system if self.args.system is not None else "you are a chatbot"
        self.AiMessage = self.args.ai if self.args.ai is not None else None
        self.model = self.args.model if self.args.model is not None else "gpt-3.5-turbo"
        self.verbose = self.args.verbose if self.args.verbose is not None else False
        self.temperature = self.args.temp if self.args.temp is not None else 0.7
        self.human_template = self.args.human if self.args.human is not None else "{human_input}"

    def setup_api_key(self):
        self.api_key = self.args.key if self.args.key is not None else os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            self.api_key = getpass.getpass('OpenAI API Key:')
        os.environ["OPENAI_API_KEY"] = self.api_key

    def setup_chat_model(self):
        self.llm = OpenAI(api_key=self.api_key, model=self.model)
        self.memory = ConversationBufferMemory()
        self.chain = LLMChain(llm=self.llm, memory=self.memory)

    def list_chats(self):
        chats = self.db.get_all_chats()
        for chat_id, chat_name in chats:
            print(f"{chat_id}: {chat_name}")

    def select_chat(self, chat_name):
        self.chat_name = chat_name
        # Retrieve chat_id based on chat_name from the database
        self.chat_id = self.db.get_chat_id(chat_name)

    def create_new_chat(self, chat_name):
        self.db.add_chat(chat_name)

    def __call__(self, message) -> str:
        user_message = message.strip()
        ai_response = self.chain.predict(human_input=user_message)
        self.doc_loader.embed_and_store_message(self.chat_id, user_message, ai_response)
        search_result = self.embed_and_search_message(user_message)
        new_message = f"FAISS Response: {search_result}\n\nAI: {ai_response}"
        return new_message

    def embed_and_search_message(self, message):
        message_embedding = self.doc_loader.embed_text(message)
        _, I = self.doc_loader.index.search(message_embedding, 1)
        similar_text_id = I.flatten()[0]
        similar_message = self.doc_loader.db.load_document(similar_text_id)
        return similar_message

def main():
    chat_interface = Interface()

    if chat_interface.args.list:
        chat_interface.list_chats()
        return

    if chat_interface.args.new:
        chat_name = input("Enter a name for the new chat: ")
        chat_interface.create_new_chat(chat_name)

    if chat_interface.args.chat:
        chat_interface.select_chat(chat_interface.args.chat)

    session = PromptSession()
    while True:
        user_message = session.prompt("You: ")
        if user_message.lower() == "/exit":
            break
        ai_response = chat_interface(user_message)
        print(f"AI: {ai_response}")

if __name__ == "__main__":
    main()
