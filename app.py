import json
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Load JSON dataset
with open("course_faqs.json", "r", encoding="utf-8") as file:
    data = json.load(file)

faqs = data["FAQs"]
questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert questions to embeddings
question_embeddings = model.encode(questions, convert_to_numpy=True)

# Build FAISS index for fast retrieval
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

# Initialize FastAPI
app = FastAPI()

# Request model
class ChatRequest(BaseModel):
    query: str

# Chatbot function
def chatbot_response(user_query):
    query_embedding = model.encode([user_query], convert_to_numpy=True)
    _, top_match = index.search(query_embedding, 1)
    best_match_index = top_match[0][0]
    return answers[best_match_index]

# API Route
@app.post("/chat")
def chat(request: ChatRequest):
    response = chatbot_response(request.query)
    return {"response": response}

# Run using: uvicorn app:app --reload
