import os
import base64
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Arkcom AI Browser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str = ""
    # Optional image as base64 data URL or raw base64 with mime
    image_base64: Optional[str] = None
    mime_type: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = Field(default="gemini-2.5-flash")
    web_search: bool = Field(default=False)


class EnhanceRequest(BaseModel):
    draft: str
    model: str = Field(default="gemini-2.5-flash")


@app.get("/")
def root():
    return {"status": "ok", "service": "Arkcom API"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from Arkcom backend!"}


@app.post("/api/enhance")
def enhance_prompt(req: EnhanceRequest):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # Graceful fallback without key
        return {"enhanced": f"Improve this: {req.draft}"}
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        sys = (
            "You are a helpful prompt engineer. Rewrite the user's draft prompt to be clear, specific, and actionable. "
            "Keep the intent, add relevant details, structure with bullet points when helpful, and keep it concise."
        )
        model = genai.GenerativeModel(req.model)
        resp = model.generate_content([sys, req.draft])
        text = getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if resp.candidates else "")
        return {"enhanced": text.strip() if text else req.draft}
    except Exception as e:
        return {"enhanced": req.draft, "error": str(e)}


@app.post("/api/chat")
def chat(req: ChatRequest):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    # Build parts from messages
    def to_parts(msg: Message) -> List[Any]:
        parts: List[Any] = []
        if msg.content:
            parts.append({"text": msg.content})
        if msg.image_base64:
            b64 = msg.image_base64
            mime = msg.mime_type or "image/png"
            # Support data URLs
            if b64.startswith("data:") and "," in b64:
                header, data = b64.split(",", 1)
                if ";base64" in header:
                    mime = header.split(";")[0].split(":")[1]
                    b64 = data
            parts.append({
                "inline_data": {
                    "mime_type": mime,
                    "data": b64
                }
            })
        return parts

    history: List[Dict[str, Any]] = []
    for m in req.messages:
        role = "user" if m.role == "user" else "model"
        history.append({"role": role, "parts": to_parts(m)})

    if not api_key:
        # No key scenario: simple echo with mock sources
        user_last = next((m for m in reversed(req.messages) if m.role == "user"), None)
        return {
            "text": f"[Mock] You said: {user_last.content if user_last else ''}",
            "sources": [
                {"title": "Example Source", "url": "https://example.com"}
            ] if req.web_search else []
        }

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        tools = []
        # Best-effort enable Google Search grounding when requested
        if req.web_search:
            try:
                tools = [{"google_search": {}}]
            except Exception:
                tools = []

        model = genai.GenerativeModel(req.model, tools=tools if tools else None)
        # Use non-streaming for reliability in this environment
        resp = model.generate_content(history)

        # Extract text
        text = getattr(resp, "text", None)
        if not text and getattr(resp, "candidates", None):
            try:
                text = resp.candidates[0].content.parts[0].text
            except Exception:
                text = ""
        text = (text or "").strip()

        # Extract sources if provided
        sources: List[Dict[str, str]] = []
        try:
            grounding = getattr(resp, "grounding_metadata", None)
            if grounding and getattr(grounding, "grounding_chunks", None):
                for chunk in grounding.grounding_chunks:
                    url = getattr(chunk.web, "uri", None) if hasattr(chunk, "web") else None
                    title = getattr(chunk, "title", None) or (url or "Source")
                    if url:
                        sources.append({"title": title, "url": url})
        except Exception:
            pass

        return {"text": text, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        from database import db

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
