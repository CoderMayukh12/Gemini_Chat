import os
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
# Load API key from .env file
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Configure generation settings
generation_config = {
    "temperature": 0.15,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize chatbot model with system instruction and an initial chat history
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    system_instruction=(
        "You are a highly knowledgeable medical chatbot specializing in brain tumors. \n"
        "You must provide detailed, factual, and to-the-point answers based on verified medical knowledge. \n"
        "If you do not know the answer, simply respond with: \"I don't know the answer yet.\""
    )
)

# Set an initial chat history
chat_history = [
    {"role": "user", "parts": ["hello"]},
    {"role": "model", "parts": ["Hello! How can I help you today with your questions about brain tumors?"]},
]

# Start the chat session
chat_session = model.start_chat(history=chat_history)

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
   allow_origins=[
    "https://gemini-chat-black-delta.vercel.app",
    "https://4385f656-a93f-443d-9dd5-127f60d8e5fa-00-2avmmtpp9s0l7.janeway.replit.dev",
    "http://localhost",  # Add these
    "http://localhost:3000",
        "http://localhost:5000",
       "file:///C:/Users/91977/OneDrive/Desktop/text.html",
    "http://127.0.0.1",
    "http://127.0.0.1:5500"  # Common port for live servers
],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
# Define the request model for the API
class ChatRequest(BaseModel):
    user_input: str

# API endpoint to handle chat requests
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global chat_session
    # Send user's message to the chatbot
    response = chat_session.send_message(request.user_input)
    # Optionally, you can update your chat history here if needed
    return {"response": response.text}

# For local testing (Vercel will automatically detect the "app" variable)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))