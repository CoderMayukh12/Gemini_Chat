import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# For local testing, load environment variables from .env
load_dotenv()

# Retrieve your Gemini API key from the environment (set this in Hugging Face Space settings too)
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please set it as an environment variable.")

# Configure the Gemini API with your API key
genai.configure(api_key=api_key)

# Generation configuration
generation_config = {
    "temperature": 0.15,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 256,
    "response_mime_type": "text/plain",
}

# Initialize the chatbot model with a system instruction
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    system_instruction=(
        "You are a highly knowledgeable medical chatbot specializing in brain tumors. "
        "Provide detailed, factual, and concise answers. "
        "If you do not know the answer, simply say: 'I don't know the answer yet.'"
    )
)

# Start a chat session
chat_session = model.start_chat()

# Initialize the FastAPI app
app = FastAPI()

# Define the request schema
class ChatRequest(BaseModel):
    user_input: str

# Define the /chat endpoint
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = chat_session.send_message(request.user_input)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local testing: run with `python main.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
