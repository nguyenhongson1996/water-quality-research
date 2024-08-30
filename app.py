from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.conversation import ConversationManager

app = FastAPI()

USER_ID = "fc3d25b5-35ae-4ece-b5f7-8c8af7e9476c"

conversation_manager = ConversationManager()


@app.post("/chat")
async def chat() -> JSONResponse:
    """
    Add a new message to the section.
    """
    conversation = conversation_manager.create_conversation(user_id=USER_ID, title="test", init_message=[])
    return JSONResponse({"conversation_id": conversation.conversation_id})


@app.post("/add_message")
async def add_message(role: str, conversation_id: str, message_text: str) -> JSONResponse:
    """
    Add a new message to the section.
    """
    conversation_manager.get_conversation(conversation_id).add_message(role=role, message_text=message_text)
    return JSONResponse({"success": True})


@app.post("/get_history")
async def get_message(conversation_id: str) -> JSONResponse:
    """
    Get all messages for a conversation.
    """
    messages = conversation_manager.get_conversation(conversation_id).load_conversation()
    return JSONResponse({"history": messages})


@app.post("/get_response")
async def get_response(conversation_id: str) -> JSONResponse:
    """
    Generate response for a conversation.

    Pull session history from db.
    Extract keywords/intent.
    Document retrieval
    Policy retrieval.
    Build prompt.
    Generate response.
    """
    return JSONResponse({
        "content": conversation_manager.get_conversation(conversation_id).make_response()
    })


"""
Endpoint for insert/delete session/user
"""

if __name__ == "__main__":
    import sys

    import uvicorn

    if len(sys.argv) >= 2:
        port = int(sys.argv[1])
    else:
        port = 8100
    uvicorn.run("app:app", host="0.0.0.0", port=port)
