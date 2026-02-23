import httpx
import json
import logging
from app.core.config import settings

logging.basicConfig(level=logging.INFO)

async def stream_openrouter(model: str, messages: list, max_tokens: int, timeout: int):
    
    # --- GOOGLE GEMINI ---
    if model.startswith("google/"):
        # Auto-fix model names
        try:
            google_model_id = model.split("/")[1].replace(":free", "")
            if "2.5" in google_model_id: google_model_id = "gemini-2.0-flash" # Fallback for stability
        except:
            google_model_id = "gemini-2.0-flash"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{google_model_id}:streamGenerateContent?alt=sse&key={settings.GOOGLE_API_KEY}"

        # Separate System Prompt from User History
        system_prompt = "You are ZenoAi."
        conversation = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                # Standardize roles
                role = "user" if msg["role"] == "user" else "model"
                # Context Merging (Prevent "User, User" errors)
                if conversation and conversation[-1]["role"] == role:
                    conversation[-1]["parts"][0]["text"] += f"\n\n{msg['content']}"
                else:
                    conversation.append({"role": role, "parts": [{"text": msg['content']}]})

        # Final Payload
        payload = {
            "contents": conversation,
            "systemInstruction": { "parts": [{"text": system_prompt}] },
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7,
                "topP": 0.95,
                "topK": 40
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout + 20.0, connect=10.0)) as client:
            async with client.stream("POST", url, headers={"Content-Type": "application/json"}, json=payload) as response:
                if response.status_code != 200:
                    err = await response.aread()
                    raise Exception(f"Google Error {response.status_code}: {err.decode()}")
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            candidates = data.get("candidates", [])
                            if candidates and "content" in candidates[0]:
                                parts = candidates[0]["content"].get("parts", [])
                                for part in parts:
                                    if "text" in part: yield part["text"]
                        except: continue
        return

    # --- OPENROUTER FALLBACK ---
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://zenoai.com",
    }
    payload = { "model": model, "messages": messages, "max_tokens": max_tokens, "stream": True }

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                raise Exception(f"OpenRouter Error {response.status_code}")
            async for line in response.aiter_lines():
                if line.startswith("data: ") and "[DONE]" not in line:
                    try:
                        data = json.loads(line[6:])
                        if "content" in data["choices"][0]["delta"]:
                            yield data["choices"][0]["delta"]["content"]
                    except: continue
