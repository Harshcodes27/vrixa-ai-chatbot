import os
import time
import base64
import logging
import asyncio
import re
from io import BytesIO
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from PIL import Image
from google import genai
from google.genai import types

def load_env_variables():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_paths = [
        os.path.join(base_dir, ".env"),
        os.path.join(base_dir, "..", ".env"),
        ".env"
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'").strip('"')
                            if not os.environ.get(k):
                                os.environ[k] = v
            except Exception:
                pass

load_env_variables()

logger = logging.getLogger("VRIXA_AI_ROUTER")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [AI-Router] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", (prompt or "").strip()).lower()

def is_conversational_request(prompt: str) -> bool:
    text = normalize_prompt(prompt)
    if not text:
        return True

    greeting_words = {
        "hi", "hello", "hey", "hii", "heyy", "hlo", "hlw", "hloo", "helo", "hy",
        "namaste", "hola", "sup", "yo", "vrixa", "suno", "sun", "bol", "bolo",
        "bhai", "bro", "ok", "acha", "theek", "hmm"
    }
    self_intro_phrases = (
        "tell me about yourself", "tell me about you", "introduce yourself",
        "who are you", "what are you", "what is your name", "what's your name",
        "who created you", "who made you", "your name", "who are u", "what are u",
        "how are you", "kaise ho", "kya haal hai", "kya haal h", "tum kaun ho", "tum kon ho"
    )

    if text in {"hi vrixa", "hello vrixa", "hey vrixa"}:
        return True
    if any(phrase in text for phrase in self_intro_phrases):
        return True
    tokens = set(re.findall(r"\w+", text))
    if len(tokens) <= 3 and tokens.issubset(greeting_words):
        return True
    return False

def is_explicit_wikipedia_request(prompt: str) -> bool:
    text = normalize_prompt(prompt)
    return bool(re.search(r"\b(?:wikipedia|wiki)\b", text))

def extract_wikipedia_query(prompt: str) -> str:
    text = normalize_prompt(prompt)
    if not text:
        return ""
    for marker in ["wikipedia ", "wiki ", "search wikipedia for ", "wikipedia search ", "search wiki for "]:
        if text.startswith(marker):
            text = text[len(marker):]
            break
    text = re.sub(r"^(according to wikipedia|look up on wikipedia|wikipedia page about|wiki page about)\s+", "", text)
    text = re.sub(r"^(who is|what is|tell me about|tell me something about|describe|explain|summarize)\s+", "", text)
    text = text.strip(" ?!.,;:-_")
    text = re.sub(r"\b(?:about|yourself|you|vrixa)\b", "", text)
    return re.sub(r"\s+", " ", text).strip()

@dataclass
class ProviderResponse:
    success: bool
    provider: str
    response: str
    model: str
    response_time_ms: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None

# =====================================================================
# Single Primary AI Provider: Google Gemini
# =====================================================================
class GeminiProvider:
    def __init__(self):
        self.provider_id = "gemini"
        self.display_name = "Google Gemini"
        self.priority = 1
        self.default_model = "gemini-3.8-flash"
        self.fallback_models = ["gemini-3.8-flash", "gemini-flash-latest", "gemini-3.6-flash"]
        self.is_enabled = True
        self.timeout_seconds = 12.0

    def get_api_key(self, custom_key: Optional[str] = None) -> str:
        if custom_key and custom_key.strip():
            return custom_key.strip()
        return os.environ.get("GEMINI_API_KEY", "").strip()

    def is_configured(self, custom_key: Optional[str] = None) -> bool:
        return bool(self.get_api_key(custom_key))

    def get_masked_key(self, custom_key: Optional[str] = None) -> str:
        key = self.get_api_key(custom_key)
        if not key:
            return ""
        if len(key) <= 8:
            return "••••••••"
        return f"••••...{key[-6:]}"

    async def generate(
        self,
        prompt: str,
        conversation_history: List[Dict[str, str]],
        image_base64: Optional[str] = None,
        system_instruction: Optional[str] = None,
        custom_key: Optional[str] = None,
        model: Optional[str] = None
    ) -> ProviderResponse:
        start = time.time()
        api_key = self.get_api_key(custom_key)

        if not api_key:
            return ProviderResponse(
                success=False,
                provider=self.provider_id,
                response="Gemini API Key is missing. Please set GEMINI_API_KEY in your environment or Settings.",
                model=model or self.default_model,
                response_time_ms=0,
                error_type="NOT_CONFIGURED",
                error_message="Gemini API key is not configured"
            )

        target_models = list(self.fallback_models)
        if model and model in target_models:
            target_models.remove(model)
            target_models.insert(0, model)
        elif model:
            target_models.insert(0, model)

        pil_image = None
        if image_base64:
            try:
                b64_clean = image_base64.split(",")[-1]
                pil_image = Image.open(BytesIO(base64.b64decode(b64_clean)))
            except Exception as img_err:
                logger.warning(f"[Gemini] Image decode error: {img_err}")

        convo_lines = []
        for turn in conversation_history[-6:]:
            r_str = "User" if turn.get("role") == "user" else "Vrixa"
            convo_lines.append(f"{r_str}: {turn.get('content', '')}")
        convo_lines.append(f"User: {prompt}\nVrixa:")
        formatted_prompt = "\n".join(convo_lines)

        gemini_contents = [pil_image, formatted_prompt] if pil_image else formatted_prompt

        last_error_type = "API_ERROR"
        last_error_msg = ""

        for m_name in target_models:
            try:
                client = genai.Client(api_key=api_key)
                cfg = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=1024,
                    temperature=0.7
                )

                resp = await asyncio.wait_for(
                    asyncio.to_thread(
                        client.models.generate_content,
                        model=m_name,
                        contents=gemini_contents,
                        config=cfg
                    ),
                    timeout=self.timeout_seconds
                )

                if resp and hasattr(resp, "text") and resp.text:
                    elapsed_ms = round((time.time() - start) * 1000, 1)
                    return ProviderResponse(
                        success=True,
                        provider=self.provider_id,
                        response=resp.text.strip(),
                        model=m_name,
                        response_time_ms=elapsed_ms
                    )
            except asyncio.TimeoutError:
                last_error_type = "TIMEOUT"
                last_error_msg = f"Model {m_name} timed out after {self.timeout_seconds}s"
                logger.warning(f"[Gemini] Model {m_name} timed out, trying fallback model...")
                continue
            except Exception as e:
                err_str = str(e)
                last_error_msg = err_str[:150]
                if "401" in err_str or "API_KEY_INVALID" in err_str or "403" in err_str:
                    return ProviderResponse(
                        success=False,
                        provider=self.provider_id,
                        response="Invalid Gemini API Key. Please verify your GEMINI_API_KEY.",
                        model=m_name,
                        response_time_ms=round((time.time() - start) * 1000, 1),
                        error_type="AUTH_ERROR",
                        error_message="Invalid Gemini API key"
                    )
                logger.warning(f"[Gemini] Model {m_name} error: {last_error_msg}, trying fallback...")
                continue

        elapsed_ms = round((time.time() - start) * 1000, 1)
        return ProviderResponse(
            success=False,
            provider=self.provider_id,
            response="I'm having trouble reaching Gemini API right now. Please try again shortly.",
            model=target_models[0],
            response_time_ms=elapsed_ms,
            error_type=last_error_type,
            error_message=last_error_msg
        )

    async def test_connection(
        self,
        custom_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        start = time.time()
        res = await self.generate(
            prompt="Reply with 'OK' in one word.",
            conversation_history=[],
            system_instruction="You are a test probe. Reply with only 'OK'.",
            custom_key=custom_key,
            model=model
        )
        elapsed_ms = round((time.time() - start) * 1000, 1)
        if res.success:
            return {
                "success": True,
                "provider": self.provider_id,
                "message": f"Connected successfully ({res.model})",
                "response_time_ms": elapsed_ms
            }
        else:
            return {
                "success": False,
                "provider": self.provider_id,
                "message": res.response or "Connection test failed.",
                "response_time_ms": elapsed_ms
            }

# =====================================================================
# Single-API AI Orchestrator
# =====================================================================
class SingleAIOrchestrator:
    def __init__(self):
        self.gemini = GeminiProvider()
        self.providers: Dict[str, Any] = {
            "gemini": self.gemini
        }

    def get_provider(self, provider_id: str) -> Optional[GeminiProvider]:
        if provider_id.lower() in ("gemini", "primary", "vrixa"):
            return self.gemini
        return None

    def get_providers_status(self, custom_keys: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        custom_keys = custom_keys or {}
        ckey = custom_keys.get("gemini", "")
        return [{
            "id": self.gemini.provider_id,
            "name": self.gemini.display_name,
            "priority": 1,
            "enabled": self.gemini.is_enabled,
            "configured": self.gemini.is_configured(ckey),
            "model": self.gemini.default_model,
            "masked_key": self.gemini.get_masked_key(ckey),
            "in_cooldown": False,
            "cooldown_remaining_seconds": 0
        }]

    async def generate_ai_response(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        image_base64: Optional[str] = None,
        system_instruction: Optional[str] = None,
        custom_keys: Optional[Dict[str, str]] = None,
        provider_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        custom_keys = custom_keys or {}
        provider_configs = provider_configs or {}

        gemini_conf = provider_configs.get("gemini", {})
        target_model = gemini_conf.get("model", self.gemini.default_model)
        custom_key = custom_keys.get("gemini", "")

        resp = await self.gemini.generate(
            prompt=user_message,
            conversation_history=conversation_history,
            image_base64=image_base64,
            system_instruction=system_instruction,
            custom_key=custom_key,
            model=target_model
        )

        return {
            "success": resp.success,
            "provider": resp.provider,
            "response": resp.response,
            "model": resp.model,
            "response_time_ms": resp.response_time_ms,
            "fallback_log": []
        }

# Global Orchestrator instance
ai_orchestrator = SingleAIOrchestrator()

# Helper export function matching Vrixa caller signatures
async def generateAIResponse(
    userMessage: str,
    conversationHistory: List[Dict[str, str]],
    imageBase64: Optional[str] = None,
    systemInstruction: Optional[str] = None,
    customKeys: Optional[Dict[str, str]] = None,
    providerConfigs: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    return await ai_orchestrator.generate_ai_response(
        user_message=userMessage,
        conversation_history=conversationHistory,
        image_base64=imageBase64,
        system_instruction=systemInstruction,
        custom_keys=customKeys,
        provider_configs=providerConfigs
    )
