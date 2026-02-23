import time
import asyncio
import traceback
from app.db.database import get_memory, save_message, log_metric, get_config_db
from app.services.llm_client import stream_openrouter

class Orchestrator:
    async def process_chat_stream(self, session_id: str, user_message: str):
        config = await get_config_db()
        
        # 1. Save to DB (Background)
        await save_message(session_id, "user", user_message)
        
        # 2. Build Context Window
        # Fetch history, but we know the DB might be slightly laggy
        history = await get_memory(session_id, config["memory_window"])
        
        # 3. CRITICAL FIX: Check if the last message in history is actually the one we just sent.
        # If the DB was slow and didn't return our new message, we MANUALLY append it.
        # This guarantees the AI sees your "Make a table" request instead of just old history.
        if not history or history[-1]["content"] != user_message:
            history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": config["system_prompt"]}] + history
        
        fallback_order = config["fallback_order"]
        max_retries = config["retry_count"]
        
        # Routing State
        current_model_idx = 0
        start_time = time.time()
        fallback_triggered = False
        
        while current_model_idx < len(fallback_order):
            model = fallback_order[current_model_idx]
            # Use a default timeout of 15s if model config is missing
            model_conf = next((m for m in config.get("models", []) if m["id"] == model), {"timeout": 15})
            
            for attempt in range(max_retries):
                try:
                    full_response = ""
                    # Stream the response
                    async for chunk in stream_openrouter(model, messages, config["max_tokens"], model_conf["timeout"]):
                        full_response += chunk
                        yield chunk
                    
                    if not full_response:
                        raise Exception("Empty response from model")

                    latency = (time.time() - start_time) * 1000
                    await save_message(session_id, "assistant", full_response)
                    await log_metric(session_id, model, latency, fallback_triggered, len(full_response)//4, "success")
                    return
                
                except Exception as e:
                    print(f"[Warn] Model {model} attempt {attempt+1} failed: {str(e)}")
                    await asyncio.sleep(1.0) # Short wait before retry
            
            print(f"[Fallback] Swapping from {model}")
            fallback_triggered = True
            current_model_idx += 1
            
            if current_model_idx == len(fallback_order) - 1:
                messages[0]["content"] += " (Respond concisely. System in fallback mode)."
                config["max_tokens"] = max(256, int(config["max_tokens"] / 2))

        error_msg = "\n\n[ZenoAi Alert] All providers unavailable. Please try again."
        yield error_msg
        await log_metric(session_id, "none", (time.time()-start_time)*1000, True, 0, "failed", "All models failed")

# --- THIS IS THE LINE THAT WAS MISSING ---
orchestrator = Orchestrator()
