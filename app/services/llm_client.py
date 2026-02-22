import httpx
import json
from app.core.config import settings

async def stream_openrouter(model: str, messages: list, max_tokens: int, timeout: int):
    
    # ==========================================
    # 1. DIRECT GOOGLE GEMINI API (Primary)
    # ==========================================
    if model.startswith("google/"):
        # Force the stable, incredibly fast Gemini 2.0 direct API model
        google_model_id = "gemini-2.0-flash" 
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{google_model_id}:streamGenerateContent?alt=sse&key={settings.GOOGLE_API_KEY}"

        system_prompt = ""
        google_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                google_messages.append({
                    "role": "user",
                    "parts": [{"text": msg["content"]}]
                })
            elif msg["role"] == "assistant":
                google_messages.append({
                    "role": "model",
                    "parts": [{"text": msg["content"]}]
                })

        payload = {
            "contents": google_messages,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
            }
        }

        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        headers = {"Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise Exception(f"HTTP {response.status_code} from Direct Google API: {body.decode()}")
                
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
        return # Exit here so it doesn't trigger OpenRouter code

    # ==========================================
    # 2. OPENROUTER API (Fallback for Llama/Mistral)
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
