import os
import re
import json
import chromadb
from flask import Flask, request, jsonify
from flask_cors import CORS
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv


#Loading environment variables from .env file
load_dotenv()


#Getting API key from environment variable
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")


#Configuring Gemini API
genai.configure(api_key=API_KEY)


app = Flask(__name__)
CORS(app)


# Initialize ChromaDB and load data
def initialize_chroma_db():
    client = chromadb.PersistentClient(path="./chroma_db")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection_name = "website_content"


    try:
        db_collection = client.get_collection(
            name=collection_name, embedding_function=sentence_transformer_ef
        )
        if db_collection.count() > 0:
            print("Vector database already populated")
            return db_collection
    except Exception:
        db_collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=sentence_transformer_ef
        )


    print("Populating vector database...")
   
    try:
        with open('website_data.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("website_data.json not found")
        return None


    if not data:
        print("website_data.json is empty")
        return None
    # Prepare documents for insertion
    documents = [d['text'] for d in data]
    ids = [d['id'] for d in data]
    metadatas = [d['metadata'] for d in data]


    db_collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Added {len(documents)} documents to database")


    return db_collection


db_collection = initialize_chroma_db()


def generate_response_and_suggestions(user_message, db_collection):
    results = db_collection.query(
        query_texts=[user_message],
        n_results=1
    )


    context = results['documents'][0][0]


    prompt = f"""
You are an expert chatbot representing Hush Solutions. Provide a helpful answer based on the context.


Speak as if you are the Hush Solutions team. Use "we" or "Hush Solutions", never "I" or "the company".


After your answer, suggest 2-3 related questions based only on the context.


Format your response as JSON with:
- "answer": The direct answer to the user's question
- "suggestions": Array of suggested related questions


If information is not in the context, state you don't have enough information and suggest contacting the company.


Context:
{context}


User question:
{user_message}
"""


    try:
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        response = model.generate_content(prompt)
        bot_text = response.text.strip()
        cleaned_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", bot_text, flags=re.DOTALL).strip()


        try:
            return json.loads(cleaned_text)
        except Exception:
            return {
                "answer": cleaned_text,
                "suggestions": []
            }


    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return {
            "answer": "Sorry, I am unable to generate a response at this time.",
            "suggestions": []
        }
#Bot API
@app.route('/chat', methods=['POST'])
def chat():
    if db_collection is None:
        return jsonify({"response": "Chatbot service not available"}), 503


    data = request.get_json(silent=True)


    if not data or 'message' not in data or not isinstance(data['message'], str):
        return jsonify({"error": "Invalid JSON or missing 'message' key"}), 400


    user_message = data['message']
    print(f"User message: {user_message}")
   
    bot_response_data = generate_response_and_suggestions(user_message, db_collection)
    print(f"Bot response data: {bot_response_data}")
   
    return jsonify(bot_response_data)


if __name__ == '__main__':
    app.run(debug=True)
    #json version from website_data.json
