import os
import re
import json
import chromadb
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from PyPDF2 import PdfReader


#Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")


#Configure Gemini
genai.configure(api_key=API_KEY)


app = Flask(__name__)
CORS(app)


#File paths
PDF_PATH = "andrewmiccai.pdf"
WEBSITE_JSON_PATH = "website_data.json"


#Helper: Load PDF text into chunks
def load_pdf_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text_chunks = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                for i in range(0, len(text), 1000):  # chunk size
                    chunk = text[i:i+1000]
                    text_chunks.append({
                        "id": f"page{page_num}_chunk{i}",
                        "text": chunk,
                        "metadata": {"source": f"PDF_page_{page_num+1}"}
                    })
        return text_chunks
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []


#Initialize PDF ChromaDB
def initialize_chroma_db_pdf():
    client = chromadb.PersistentClient(path="./chroma_db_pdf")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection_name = "datafrompdf_pdf"


    try:
        db_collection = client.get_collection(name=collection_name, embedding_function=ef)
        if db_collection.count() > 0:
            print("PDF Vector database already populated")
            return db_collection
    except Exception:
        db_collection = client.get_or_create_collection(name=collection_name, embedding_function=ef)


    print("Loading PDF into ChromaDB...")
    data = load_pdf_text(PDF_PATH)
    if not data:
        print("No text found in PDF")
        return None


    db_collection.add(
        documents=[d['text'] for d in data],
        metadatas=[d['metadata'] for d in data],
        ids=[d['id'] for d in data]
    )
    print(f"Added {len(data)} chunks from PDF into ChromaDB")
    return db_collection


#Initialize website JSON ChromaDB
def initialize_chroma_db_website():
    client = chromadb.PersistentClient(path="./chroma_db_website")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection_name = "website_content"


    try:
        db_collection = client.get_collection(name=collection_name, embedding_function=ef)
        if db_collection.count() > 0:
            print("Website Vector database already populated")
            return db_collection
    except Exception:
        db_collection = client.get_or_create_collection(name=collection_name, embedding_function=ef)


    print("Loading website JSON into ChromaDB...")
    try:
        with open(WEBSITE_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("website_data.json not found")
        return None


    if not data:
        print("website_data.json is empty")
        return None


    db_collection.add(
        documents=[d['text'] for d in data],
        metadatas=[d['metadata'] for d in data],
        ids=[d['id'] for d in data]
    )
    print(f"Added {len(data)} documents from website JSON into ChromaDB")
    return db_collection


#Initialize both DBs
db_pdf = initialize_chroma_db_pdf()
db_website = initialize_chroma_db_website()
db_collections = [db for db in [db_pdf, db_website] if db is not None]


#Generate response by querying both DBs
def generate_response_and_suggestions(user_message, db_collections):
    all_contexts = []


    for db in db_collections:
        results = db.query(query_texts=[user_message], n_results=3)
        if results["documents"] and results["documents"][0]:
            all_contexts.extend(results["documents"][0])


    if not all_contexts:
        return {
            "answer": "We don’t have enough information in the handbook. Please contact Hush Solutions.",
            "suggestions": []
        }


    context = " ".join(all_contexts)


    prompt = f"""
You are an expert chatbot representing Hush Solutions.
Answer based only on the provided context. Always speak as "we" or "Hush Solutions".


If the context is insufficient, say:
"We don’t have enough information in the handbook. Please contact Hush Solutions."


After your answer, suggest 2–3 related questions.


Format strictly as JSON:
- "answer": string
- "suggestions": array of strings


Context:
{context}


User question:
{user_message}
"""


    try:
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        response = model.generate_content(prompt)
        bot_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.text.strip(), flags=re.DOTALL).strip()
        try:
            return json.loads(bot_text)
        except Exception:
            return {"answer": bot_text, "suggestions": []}
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return {"answer": "Sorry, we are unable to generate a response right now.", "suggestions": []}


#Flask API endpoint
@app.route('/chat', methods=['POST'])
def chat():
    if not db_collections:
        return jsonify({"response": "Chatbot service not available"}), 503


    data = request.get_json(silent=True)
    if not data or 'message' not in data or not isinstance(data['message'], str):
        return jsonify({"error": "Invalid JSON or missing 'message' key"}), 400


    user_message = data['message']
    print(f"User message: {user_message}")


    bot_response_data = generate_response_and_suggestions(user_message, db_collections)
    print(f"Bot response data: {bot_response_data}")


    return jsonify(bot_response_data)


#Run the app
if __name__ == '__main__':
    app.run(debug=True)