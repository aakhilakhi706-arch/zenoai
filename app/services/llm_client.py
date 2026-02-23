import httpx
import json
import logging
from app.core.config import settings

logging.basicConfig(level=logging.INFO)

async def stream_openrouter(model: str, messages: list, max_tokens: int, timeout: int):
    
    # --- 1. SETUP DEFAULT (OpenRouter) ---
    provider = "openrouter"
    clean_model = model

    # --- 2. ROUTING LOGIC ---
    
    # > GOOGLE DIRECT (Special Case)
    if model.startswith("google/"):
        # CRITICAL FIX: We must iterate and yield, NOT return
        async for chunk in stream_google(model, messages, max_tokens, timeout):
            yield chunk
        return # Stop function here

    # > XAI (GROK) DIRECT
    elif model.startswith("xai/") or model.startswith("grok/"):
        provider = "xai"
        clean_model = model.replace("xai/", "").replace("grok/", "")
        
    # > DEEPSEEK DIRECT
    elif model.startswith("deepseek/"):
        provider = "deepseek"
        clean_model = model.replace("deepseek/", "")

    # --- 3. GENERIC DELEGATION (OpenRouter, Grok, DeepSeek) ---
    # CRITICAL FIX: We yield from the generic streamer. 
    # This ensures this function acts as a Generator in ALL scenarios.
    async for chunk in stream_generic(clean_model, messages, max_tokens, timeout, provider):
        yield chunk


# =========================================================
# GENERIC CLIENT (Handles OpenRouter, Grok, DeepSeek)
# =========================================================
async def stream_generic(model: str, messages: list, max_tokens: int, timeout: int, provider: str):
    
    # Setup Keys & URLs
    if provider == "xai":
        api_key = settings.XAI_API_KEY
        base_url = "https://api.x.ai/v1/chat/completions"
    elif provider == "deepseek":
        api_key = settings.DEEPSEEK_API_KEY
        base_url = "https://api.deepseek.com/chat/completions"
    else: # OpenRouter
        api_key = settings.OPENROUTER_API_KEY
        base_url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://zenoai.com",
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7 
    }

    if provider == "deepseek" and "reasoner" in model:
        payload["max_tokens"] = 8192

    print(f"[ZenoAi] Routing via {provider.upper()} -> Model: {model}")

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
                        
                        # Handle standard OpenAI/OpenRouter format
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                                
                    except: continue


# =========================================================
# GOOGLE GEMINI SPECIAL CLIENT
# =========================================================
async def stream_google(model: str, messages: list, max_tokens: int, timeout: int):
    try:
        google_model_id = model.split("/")[1].replace(":free", "")
    except:
        google_model_id = "gemini-2.0-flash"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{google_model_id}:streamGenerateContent?alt=sse&key={settings.GOOGLE_API_KEY}"
    
    system_prompt = "You are ZenoAi."
    conversation = []
    
    for msg in messages:
        if msg["role"] == "system": system_prompt = msg["content"]
        else:
            role = "user" if msg["role"] == "user" else "model"
            if conversation and conversation[-1]["role"] == role:
                conversation[-1]["parts"][0]["text"] += f"\n\n{msg['content']}"
            else:
                conversation.append({"role": role, "parts": [{"text": msg['content']}]})

    payload = {
        "contents": conversation,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": { "maxOutputTokens": max_tokens, "temperature": 0.7 },
        "safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout + 20.0)) as client:
        async with client.stream("POST", url, headers={"Content-Type": "application/json"}, json=payload) as response:
            if response.status_code != 200:
                err = await response.aread()
                print(f"[GOOGLE ERROR] {err.decode()}")
                raise Exception(f"Google Error {response.status_code}")
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if data.get("candidates") and "content" in data["candidates"][0]:
                            parts = data["candidates"][0]["content"].get("parts", [])
                            for p in parts: 
                                if "text" in p: yield p["text"]
                    except: continue
