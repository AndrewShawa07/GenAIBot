import os
import re
import json
import chromadb
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pyngrok import ngrok, conf
from flask_mail import Mail, Message
from datetime import datetime


# Loading environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")


# Adding intent detection function for greetings and farewells
def detect_intent(user_message):
    msg = user_message.strip().lower()
    if msg in ["bye", "exit", "quit", "goodbye"]:
        return "farewell"
    if msg in ["hi", "hello", "hey"]:
        return "greet"
    if msg in ["no", "nah", "not really"]:
        return "negative"
    return "unknown"


# Configure Gemini
genai.configure(api_key=API_KEY)


app = Flask(__name__)
# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'andrewshawa0420@gmail.com'
app.config['MAIL_PASSWORD'] = 'buwz shtu ycvu zocs'
app.config['MAIL_DEFAULT_SENDER'] = 'andrewshawa0420@gmail.com'

mail = Mail(app)
CORS(app)

# Path for website JSON
WEBSITE_JSON_PATH = "website_data.json"

# Initialize website JSON ChromaDB
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


# Initialize DB
db_website = initialize_chroma_db_website()
db_collections = [db for db in [db_website] if db is not None]

# Store pending questions that need email follow-up (in production, use a database)
pending_questions = {}

# Email validation function
def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def generate_response_and_suggestions(user_message, db_collections):
    all_contexts = []

    for db in db_collections:
        results = db.query(query_texts=[user_message], n_results=3)
        if results["documents"] and results["documents"][0]:
            all_contexts.extend(results["documents"][0])

    if not all_contexts:
        # Store the question temporarily and return response asking for email
        question_id = datetime.now().strftime("%Y%m%d%H%M%S")
        pending_questions[question_id] = user_message
        
        return {
            "answer": "Thank you for your question! Our team will review it. For a faster and more detailed response, please provide your email address so we can contact you directly.",
            "suggestions": ["Would you like to provide your email for follow-up?"],
            "needs_email": True,
            "question_id": question_id
        }
    
    context = " ".join(all_contexts)

    prompt = f"""
You are an expert chatbot representing Hush Solutions. Provide a helpful answer based on the context.
Speak as if you are the Hush Solutions team. Use "we" or "Hush Solutions", never "I" or "the company".

After your answer, suggest 2-3 related questions based only on the context.

Format your response as JSON with:
- "answer": The direct answer to the user's question
- "suggestions": Array of suggested related questions

If the context does NOT contain the answer, explicitly reply with:
"INSUFFICIENT_INFO"

Context:
{context}

User question:
{user_message}
"""

    try:
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        response = model.generate_content(prompt)
        bot_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.text.strip(), flags=re.DOTALL).strip()
        
        if "INSUFFICIENT_INFO" in bot_text.upper():
            # Store the question temporarily and return response asking for email
            question_id = datetime.now().strftime("%Y%m%d%H%M%S")
            pending_questions[question_id] = user_message
            
            return {
                "answer": "Thank you for your question! Our team will review it. For a faster and more detailed response, please provide your email address so we can contact you directly.",
                "suggestions": ["Would you like to provide your email for follow-up?"],
                "needs_email": True,
                "question_id": question_id
            }
        
        try:
            response_data = json.loads(bot_text)
            response_data["needs_email"] = False
            return response_data
        except Exception:
            return {
                "answer": bot_text, 
                "suggestions": [],
                "needs_email": False
            }
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return {
            "answer": "Sorry, we are unable to generate a response right now.", 
            "suggestions": [],
            "needs_email": False
        }

# Flask API endpoint for chat
@app.route('/chat', methods=['POST'])
def chat():
    if not db_collections:
        return jsonify({"response": "Chatbot service not available"}), 503

    data = request.get_json(silent=True)
    if not data or 'message' not in data or not isinstance(data['message'], str):
        return jsonify({"error": "Invalid JSON or missing 'message' key"}), 400

    user_message = data['message']
    print(f"User message: {user_message}")
    
    # Check for intents first
    intent = detect_intent(user_message)

    if intent == "farewell":
        return jsonify({
            "answer": "We understand you're ending the conversation. Thank you for chatting with Hush Solutions!",
            "suggestions": [
                "Can you tell me more about Hush Solutions' services?",
                "How can I contact Hush Solutions directly?"
            ],
            "needs_email": False
        })

    elif intent == "greet":
        return jsonify({
            "answer": "Hello! How can I assist you today?",
            "suggestions": [
                "Tell me about insurance coverage",
                "How do I file a claim?"
            ],
            "needs_email": False
        })

    else:
        # proceed to query ChromaDB and call Gemini
        bot_response_data = generate_response_and_suggestions(user_message, db_collections)
        print(f"Bot response data: {bot_response_data}")
        return jsonify(bot_response_data)

# New endpoint for email collection
@app.route('/submit_email', methods=['POST'])
def submit_email():
    data = request.get_json(silent=True)
    if not data or 'email' not in data or 'question_id' not in data:
        return jsonify({"error": "Missing email or question ID"}), 400
    
    user_email = data['email'].strip()
    question_id = data['question_id']
    
    # Validate email
    if not is_valid_email(user_email):
        return jsonify({"error": "Please provide a valid email address"}), 400
    
    # Check if question exists in pending questions
    if question_id not in pending_questions:
        return jsonify({"error": "Invalid question ID or question has expired"}), 400
    
    question = pending_questions[question_id]
    
    # Send email with both question and user email
    try:
        msg = Message(
            subject=f"Follow-up Question from Chatbot User - {user_email}",
            recipients=["quiverkhalifa@gmail.com"],
            body=f"""
New Unanswered Question from Chatbot:

User Email: {user_email}
Question: {question}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Question ID: {question_id}

This question could not be answered by the chatbot and requires human assistance.
Please follow up with the user directly.
"""
        )
        mail.send(msg)
        
        # Remove the question from pending list after successful email
        del pending_questions[question_id]
        
        print(f"🫱🏽‍🫲🏽 Sent follow-up email for question {question_id} from {user_email}")
        
        return jsonify({
            "success": True,
            "message": "Thank you! We've received your email and our team will contact you shortly."
        })
        
    except Exception as e:
        print(f"Email sending failed: {e}")
        return jsonify({"error": "Failed to submit email. Please try again later."}), 500

# Optional: Endpoint to check if a question is still pending
@app.route('/check_question/<question_id>', methods=['GET'])
def check_question(question_id):
    if question_id in pending_questions:
        return jsonify({
            "exists": True,
            "question": pending_questions[question_id]
        })
    else:
        return jsonify({"exists": False})

# Optional: Clean up old pending questions (could be run periodically)
def cleanup_pending_questions(hours=24):
    """Remove questions older than specified hours"""
    current_time = datetime.now()
    expired_questions = []
    
    for question_id in list(pending_questions.keys()):
        try:
            # Extract timestamp from question_id (format: YYYYMMDDHHMMSS)
            question_time = datetime.strptime(question_id, "%Y%m%d%H%M%S")
            if (current_time - question_time).total_seconds() > hours * 3600:
                expired_questions.append(question_id)
        except ValueError:
            # If ID format is invalid, remove it
            expired_questions.append(question_id)
    
    for question_id in expired_questions:
        del pending_questions[question_id]
    
    if expired_questions:
        print(f"Cleaned up {len(expired_questions)} expired questions")

# Running the app with tunneling
if __name__ == '__main__':
    # Configure ngrok with authtoken
    conf.get_default().auth_token = "33Q1qbwIdyZWgWFVlW7TceZBVvx_VV2RhytAWEFRBWfUdy1m"  
    
    # Creating a public tunnel for testing purposes
    print("Starting ngrok tunnel...")
    public_url = ngrok.connect(5000)
    print(f"The Flask app is now publicly accessible at: {public_url}")
    print(f"URL to use in Postman: {public_url}/chat")
    
    # Clean up any expired questions on startup
    cleanup_pending_questions()
    
    # Start Flask app WITHOUT debug mode to avoid instance conflicts
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down tunnel...")
        ngrok.kill()