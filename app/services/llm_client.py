import httpx
import json
from app.core.config import settings

async def stream_openrouter(model: str, messages: list, max_tokens: int, timeout: int):
    
    # ==========================================
    # 1. DIRECT GOOGLE GEMINI API (Primary)
    # ==========================================
    if model.startswith("google/"):
        
        # TRUST THE ADMIN PANEL: 
        # Extract "gemini-2.5-flash" directly from "google/gemini-2.5-flash:free"
        try:
            google_model_id = model.split("/")[1].replace(":free", "")
        except IndexError:
            google_model_id = "gemini-2.5-flash" # Default if string is malformed

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{google_model_id}:streamGenerateContent?alt=sse&key={settings.GOOGLE_API_KEY}"

        # 1. Extract System Prompt
        system_prompt = "You are ZenoAi, a helpful conversational AI."
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]

        # 2. CONTEXT FIXER (Crucial for "Looping" issues)
        # Google API throws errors or ignores context if you send two "User" messages in a row.
        # This block merges them so the AI actually remembers what you said.
        google_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue 
            
            # Map roles to strictly "user" or "model"
            role = "user" if msg["role"] == "user" else "model"
            
            # If the last message was from the SAME role, append to it instead of creating a new one.
            if google_messages and google_messages[-1]["role"] == role:
                google_messages[-1]["parts"][0]["text"] += f"\n\n{msg['content']}"
            else:
                google_messages.append({
                    "role": role,
                    "parts": [{"text": msg['content']}]
                })

        # 3. Payload Construction
        payload = {
            "contents": google_messages,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7, # Creativity (prevent robotic looping)
                "topP": 0.95,
                "topK": 40
            },
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            # Disable safety filters to ensure code generation isn't blocked
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }

        headers = {"Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout + 10.0, connect=10.0)) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    # If 2.5 fails, it will print the specific Google error here
                    raise Exception(f"HTTP {response.status_code} from Google ({google_model_id}): {body.decode()}")
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            candidates = data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                for part in parts:
                                    if "text" in part:
                                        yield part["text"]
                        except json.JSONDecodeError:
                            continue
        return

    # ==========================================
    # 2. OPENROUTER API (Fallback)
    # ==========================================
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://zenoai-wdy5.onrender.com",
        "X-Title": "ZenoAi"
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise Exception(f"HTTP {response.status_code} from OpenRouter: {body.decode()}")
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue
