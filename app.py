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


# Load env variables
load_dotenv()


API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")


# Configure Gemini
genai.configure(api_key=API_KEY)


app = Flask(__name__)
CORS(app)


PDF_PATH = "andrewmiccai.pdf"


#Function to load PDF text
def load_pdf_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text_chunks = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                # Split into smaller chunks (for embeddings)
                for i in range(0, len(text), 1000):
                    chunk = text[i:i+1000]
                    text_chunks.append({
                        "id": f"page{page_num}_chunk{i}",
                        "text": chunk,
                        "metadata": {"source": f"page_{page_num+1}"}
                    })
        return text_chunks
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []


#Initialize ChromaDB with PDF
def initialize_chroma_db():
    client = chromadb.PersistentClient(path="./chroma_db_pdf")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection_name = "insurance_pdf"


    try:
        db_collection = client.get_collection(
            name=collection_name, embedding_function=sentence_transformer_ef
        )
        if db_collection.count() > 0:
            print("PDF Vector database already populated")
            return db_collection
    except Exception:
        db_collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=sentence_transformer_ef
        )


    print("Loading PDF...")
    data = load_pdf_text(PDF_PATH)


    if not data:
        print("No text found in PDF")
        return None


    documents = [d['text'] for d in data]
    ids = [d['id'] for d in data]
    metadatas = [d['metadata'] for d in data]


    db_collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Added {len(documents)} chunks from PDF into ChromaDB")


    return db_collection


db_collection = initialize_chroma_db()


#Generate response
def generate_response_and_suggestions(user_message, db_collection):
    results = db_collection.query(
        query_texts=[user_message],
        n_results=3
    )


    if not results["documents"] or not results["documents"][0]:
        return {
            "answer": "Sorry, I couldn’t find information in the Insurance Handbook.",
            "suggestions": []
        }


    context = " ".join(results['documents'][0])


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
        bot_text = response.text.strip()
        cleaned_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", bot_text, flags=re.DOTALL).strip()


        try:
            return json.loads(cleaned_text)
        except Exception:
            return {"answer": cleaned_text, "suggestions": []}


    except Exception as e:
        print(f"Gemini API Error: {e}")
        return {
            "answer": "Sorry, I’m unable to generate a response right now.",
            "suggestions": []
        }


#Flask API endpoint
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
