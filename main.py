import os
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env (for local testing)
load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Configure generation settings
generation_config = {
    "temperature": 0.15,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 256,
    "response_mime_type": "text/plain",
}

# Initialize the chatbot model with system instruction
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

# Initialize FastAPI app
app = FastAPI()

# Define request model
class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = chat_session.send_message(request.user_input)
        return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}

# For local testing, you can run: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
