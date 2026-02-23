import httpx
import json
import logging
from app.core.config import settings

logging.basicConfig(level=logging.INFO)

async def stream_openrouter(model: str, messages: list, max_tokens: int, timeout: int):
    
    # --- 1. ROUTING LOGIC ---
    provider = "openrouter" # Default
    api_key = settings.OPENROUTER_API_KEY
    base_url = "https://openrouter.ai/api/v1/chat/completions"
    clean_model = model

    # > GOOGLE DIRECT
    if model.startswith("google/"):
        # FIX: We must loop and yield, not return directly
        async for chunk in stream_google(model, messages, max_tokens, timeout):
            yield chunk
        return
    
    # > XAI (GROK) DIRECT
    elif model.startswith("xai/") or model.startswith("grok/"):
        provider = "xai"
        api_key = settings.XAI_API_KEY
        base_url = "https://api.x.ai/v1/chat/completions"
        clean_model = model.replace("xai/", "").replace("grok/", "")
        
    # > DEEPSEEK DIRECT
    elif model.startswith("deepseek/"):
        provider = "deepseek"
        api_key = settings.DEEPSEEK_API_KEY
        base_url = "https://api.deepseek.com/chat/completions"
        clean_model = model.replace("deepseek/", "")

    # --- 2. GENERIC CLIENT (Grok, DeepSeek, OpenRouter) ---
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://zenoai.com",
    }
    
    conversation = messages 
    
    payload = {
        "model": clean_model,
        "messages": conversation,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7 
    }

    if provider == "deepseek" and "reasoner" in clean_model:
        payload["max_tokens"] = 8192 

    print(f"[ZenoAi] Routing to {provider.upper()} -> Model: {clean_model}")

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as client:
        async with client.stream("POST", base_url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                err = await response.aread()
                print(f"[{provider.upper()} ERROR] {err.decode()}")
                raise Exception(f"{provider.upper()} Error {response.status_code}")
            
            async for line in response.aiter_lines():
                if line.startswith("data: ") and "[DONE]" not in line:
                    try:
                        data = json.loads(line[6:])
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except: continue

# --- 3. GOOGLE GEMINI SPECIAL CLIENT (With Memory Fix) ---
async def stream_google(model: str, messages: list, max_tokens: int, timeout: int):
    try:
        google_model_id = model.split("/")[1].replace(":free", "")
    except:
        google_model_id = "gemini-2.0-flash"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{google_model_id}:streamGenerateContent?alt=sse&key={settings.GOOGLE_API_KEY}"
    
    system_prompt = "You are ZenoAi."
    conversation = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
    
    for msg in messages:
        if msg["role"] == "system": continue
        
        role = "user" if msg["role"] == "user" else "model"
        
        if conversation and conversation[-1]["role"] == role:
            conversation[-1]["parts"][0]["text"] += f"\n\n{msg['content']}"
        else:
            conversation.append({"role": role, "parts": [{"text": msg['content']}]})

    payload = {
        "contents": conversation,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
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
                print(f"[GOOGLE ERROR] {err.decode()}")
                raise Exception(f"Google Error {response.status_code}: {err.decode()}")
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if data.get("candidates") and "content" in data["candidates"][0]:
                            parts = data["candidates"][0]["content"].get("parts", [])
                            for p in parts: 
                                if "text" in p: yield p["text"]
                    except: continue
