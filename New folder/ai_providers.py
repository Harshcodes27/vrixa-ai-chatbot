import os
import time
import json
import base64
import logging
import asyncio
import re
from io import BytesIO
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from PIL import Image
import httpx
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
            break


load_env_variables()

# Setup clean logger for AI Router
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
        "tell me about yourself",
        "tell me about you",
        "introduce yourself",
        "who are you",
        "what are you",
        "what is your name",
        "what's your name",
        "who created you",
        "who made you",
        "your name",
        "who are u",
        "what are u",
        "how are you",
        "kaise ho",
        "kya haal hai",
        "kya haal h",
        "tum kaun ho",
        "tum kon ho"
    )

    if text in {"hi vrixa", "hello vrixa", "hey vrixa"}:
        return True
    if any(phrase in text for phrase in self_intro_phrases):
        return True
    if any(word in text for word in [
        "how are you", "what's up", "what up", "kya haal", "kaise ho",
        "kya chal raha", "kya kar rahe ho"
    ]):
        return True
    tokens = set(re.findall(r"\w+", text))
    if len(tokens) <= 3 and tokens.issubset(greeting_words):
        return True
    return False


def is_explicit_wikipedia_request(prompt: str) -> bool:
    text = normalize_prompt(prompt)
    if not text:
        return False

    # A normal fact question belongs to the main AI. Wikipedia is a tool,
    # so require the user to name that tool explicitly.
    return bool(re.search(r"\b(?:wikipedia|wiki)\b", text))


def extract_wikipedia_query(prompt: str) -> str:
    text = normalize_prompt(prompt)
    if not text:
        return ""

    for marker in [
        "wikipedia ", "wiki ", "search wikipedia for ", "wikipedia search ", "search wiki for "
    ]:
        if text.startswith(marker):
            text = text[len(marker):]
            break
    text = re.sub(r"^(according to wikipedia|look up on wikipedia|wikipedia page about|wiki page about)\s+", "", text)
    text = re.sub(r"^(who is|what is|tell me about|tell me something about|describe|explain|summarize)\s+", "", text)
    text = text.strip(" ?!.,;:-_")
    text = re.sub(r"\b(?:about|yourself|you|vrixa)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class ProviderResponse:
    success: bool
    provider: str
    response: str
    model: str
    response_time_ms: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class BaseAIProvider:
    def __init__(self, provider_id: str, display_name: str, priority: int, default_model: str):
        self.provider_id = provider_id
        self.display_name = display_name
        self.priority = priority
        self.default_model = default_model
        self.is_enabled = True
        self.timeout_seconds = 8.0

    def get_api_key(self, custom_key: Optional[str] = None) -> str:
        if custom_key and custom_key.strip():
            return custom_key.strip()
        env_var_name = f"{self.provider_id.upper()}_API_KEY"
        return os.environ.get(env_var_name, "").strip()

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
        raise NotImplementedError

    async def test_connection(
        self,
        custom_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        start = time.time()
        try:
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
                    "message": "Connection unavailable. Check the provider configuration and try again.",
                    "response_time_ms": elapsed_ms
                }
        except Exception as e:
            elapsed_ms = round((time.time() - start) * 1000, 1)
            return {
                "success": False,
                "provider": self.provider_id,
                "message": "Connection test failed. Check the provider configuration and try again.",
                "response_time_ms": elapsed_ms
            }

# =====================================================================
# 1. Google Gemini Provider (Primary)
# =====================================================================
class GeminiProvider(BaseAIProvider):
    def __init__(self):
        super().__init__("gemini", "Google Gemini", 1, "gemini-3.6-flash")
        self.fallback_models = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.7-flash"]

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
        import re

        raw_keys = []
        if custom_key and custom_key.strip():
            raw_keys.extend([k.strip() for k in re.split(r'[,;\n]+', custom_key) if k.strip()])
        env_key = os.environ.get("GEMINI_API_KEY", "")
        if env_key and env_key.strip():
            raw_keys.extend([k.strip() for k in re.split(r'[,;\n]+', env_key) if k.strip()])
        env_keys = os.environ.get("GEMINI_API_KEYS", "")
        if env_keys and env_keys.strip():
            raw_keys.extend([k.strip() for k in re.split(r'[,;\n]+', env_keys) if k.strip()])

        keys_to_try = list(dict.fromkeys(raw_keys))
        if not keys_to_try:
            return ProviderResponse(
                success=False,
                provider=self.provider_id,
                response="",
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
        for turn in conversation_history[-4:]:
            r_str = "User" if turn.get("role") == "user" else "Vrixa"
            convo_lines.append(f"{r_str}: {turn.get('content', '')}")
        convo_lines.append(f"User: {prompt}\nVrixa:")
        formatted_prompt = "\n".join(convo_lines)

        gemini_contents = [pil_image, formatted_prompt] if pil_image else formatted_prompt

        last_error_type = "UNKNOWN_ERROR"
        last_error_msg = ""

        for key_idx, current_key in enumerate(keys_to_try):
            for m_name in target_models:
                try:
                    client = genai.Client(api_key=current_key)
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

                    if resp and hasattr(resp, 'text') and resp.text:
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
                    last_error_msg = f"Gemini model {m_name} timed out after {self.timeout_seconds}s"
                    logger.warning(f"[Gemini] {last_error_msg}")
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                        last_error_type = "QUOTA_EXCEEDED"
                        last_error_msg = f"Gemini quota reached on key #{key_idx+1} ({m_name})"
                    elif "401" in err_str or "API_KEY_INVALID" in err_str:
                        last_error_type = "AUTH_ERROR"
                        last_error_msg = f"Invalid Gemini API Key #{key_idx+1}"
                        break
                    elif "503" in err_str or "UNAVAILABLE" in err_str:
                        last_error_type = "SERVER_ERROR"
                        last_error_msg = f"Gemini {m_name} temporarily unavailable"
                    else:
                        last_error_type = "API_ERROR"
                        last_error_msg = err_str[:120]
                    logger.warning(f"[Gemini] {last_error_msg}")

        elapsed_ms = round((time.time() - start) * 1000, 1)
        return ProviderResponse(
            success=False,
            provider=self.provider_id,
            response="",
            model=target_models[0],
            response_time_ms=elapsed_ms,
            error_type=last_error_type,
            error_message=last_error_msg
        )

# =====================================================================
# 2. Anthropic Claude Provider (1st Fallback)
# =====================================================================
class ClaudeProvider(BaseAIProvider):
    def __init__(self):
        super().__init__("claude", "Anthropic Claude", 2, "claude-3-5-haiku-20241022")
        self.endpoint = "https://api.anthropic.com/v1/messages"

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
                response="",
                model=model or self.default_model,
                response_time_ms=0,
                error_type="NOT_CONFIGURED",
                error_message="Anthropic API key is not configured"
            )

        use_model = model or self.default_model

        messages = []
        for turn in conversation_history[-4:]:
            role = "user" if turn.get("role") == "user" else "assistant"
            content = turn.get("content", "").strip()
            if content:
                messages.append({"role": role, "content": content})

        if image_base64:
            media_type = "image/jpeg"
            b64_clean = image_base64
            if ";base64," in image_base64:
                prefix, b64_clean = image_base64.split(";base64,", 1)
                if "image/png" in prefix: media_type = "image/png"
                elif "image/webp" in prefix: media_type = "image/webp"
            user_content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_clean
                    }
                },
                {"type": "text", "text": prompt or "Analyze this image."}
            ]
        else:
            user_content = prompt or "Hello"

        messages.append({"role": "user", "content": user_content})

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        payload: Dict[str, Any] = {
            "model": use_model,
            "max_tokens": 1024,
            "temperature": 0.7,
            "messages": messages
        }
        if system_instruction:
            payload["system"] = system_instruction

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as http_client:
                res = await http_client.post(self.endpoint, headers=headers, json=payload)
                elapsed_ms = round((time.time() - start) * 1000, 1)

                if res.status_code == 200:
                    data = res.json()
                    content_blocks = data.get("content", [])
                    text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
                    reply = "\n".join(text_parts).strip()
                    return ProviderResponse(
                        success=True,
                        provider=self.provider_id,
                        response=reply,
                        model=use_model,
                        response_time_ms=elapsed_ms
                    )
                elif res.status_code == 429:
                    return ProviderResponse(
                        success=False,
                        provider=self.provider_id,
                        response="",
                        model=use_model,
                        response_time_ms=elapsed_ms,
                        error_type="QUOTA_EXCEEDED",
                        error_message="Claude rate limit / balance exhausted (429)"
                    )
                elif res.status_code == 401:
                    return ProviderResponse(
                        success=False,
                        provider=self.provider_id,
                        response="",
                        model=use_model,
                        response_time_ms=elapsed_ms,
                        error_type="AUTH_ERROR",
                        error_message="Invalid Anthropic API Key (401)"
                    )
                else:
                    return ProviderResponse(
                        success=False,
                        provider=self.provider_id,
                        response="",
                        model=use_model,
                        response_time_ms=elapsed_ms,
                        error_type="SERVER_ERROR",
                        error_message=f"Claude API Error: HTTP {res.status_code}"
                    )
        except httpx.TimeoutException:
            elapsed_ms = round((time.time() - start) * 1000, 1)
            return ProviderResponse(
                success=False,
                provider=self.provider_id,
                response="",
                model=use_model,
                response_time_ms=elapsed_ms,
                error_type="TIMEOUT",
                error_message=f"Claude request timed out after {self.timeout_seconds}s"
            )
        except Exception as e:
            elapsed_ms = round((time.time() - start) * 1000, 1)
            return ProviderResponse(
                success=False,
                provider=self.provider_id,
                response="",
                model=use_model,
                response_time_ms=elapsed_ms,
                error_type="NETWORK_ERROR",
                error_message=str(e)[:120]
            )

# =====================================================================
# 3. OpenAI Provider (2nd Fallback)
# =====================================================================
class OpenAIProvider(BaseAIProvider):
    def __init__(self):
        super().__init__("openai", "OpenAI", 3, "gpt-4o-mini")
        self.endpoint = "https://api.openai.com/v1/chat/completions"

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
                response="",
                model=model or self.default_model,
                response_time_ms=0,
                error_type="NOT_CONFIGURED",
                error_message="OpenAI API key is not configured"
            )

        use_model = model or self.default_model

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        for turn in conversation_history[-4:]:
            role = "user" if turn.get("role") == "user" else "assistant"
            content = turn.get("content", "").strip()
            if content:
                messages.append({"role": role, "content": content})

        if image_base64:
            b64_url = image_base64 if image_base64.startswith("data:") else f"data:image/jpeg;base64,{image_base64}"
            user_content = [
                {"type": "text", "text": prompt or "Analyze this image."},
                {"type": "image_url", "image_url": {"url": b64_url}}
            ]
        else:
            user_content = prompt or "Hello"

        messages.append({"role": "user", "content": user_content})

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": use_model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.7
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as http_client:
                res = await http_client.post(self.endpoint, headers=headers, json=payload)
                elapsed_ms = round((time.time() - start) * 1000, 1)

                if res.status_code == 200:
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices:
                        reply = choices[0].get("message", {}).get("content", "").strip()
                        return ProviderResponse(
                            success=True,
                            provider=self.provider_id,
                            response=reply,
                            model=use_model,
                            response_time_ms=elapsed_ms
                        )
                elif res.status_code == 429:
                    return ProviderResponse(
                        success=False,
                        provider=self.provider_id,
                        response="",
                        model=use_model,
                        response_time_ms=elapsed_ms,
                        error_type="QUOTA_EXCEEDED",
                        error_message="OpenAI quota/rate limit exceeded (429)"
                    )
                elif res.status_code == 401:
                    return ProviderResponse(
                        success=False,
                        provider=self.provider_id,
                        response="",
                        model=use_model,
                        response_time_ms=elapsed_ms,
                        error_type="AUTH_ERROR",
                        error_message="Invalid OpenAI API Key (401)"
                    )
                else:
                    return ProviderResponse(
                        success=False,
                        provider=self.provider_id,
                        response="",
                        model=use_model,
                        response_time_ms=elapsed_ms,
                        error_type="SERVER_ERROR",
                        error_message=f"OpenAI API Error: HTTP {res.status_code}"
                    )
        except httpx.TimeoutException:
            elapsed_ms = round((time.time() - start) * 1000, 1)
            return ProviderResponse(
                success=False,
                provider=self.provider_id,
                response="",
                model=use_model,
                response_time_ms=elapsed_ms,
                error_type="TIMEOUT",
                error_message=f"OpenAI request timed out after {self.timeout_seconds}s"
            )
        except Exception as e:
            elapsed_ms = round((time.time() - start) * 1000, 1)
            return ProviderResponse(
                success=False,
                provider=self.provider_id,
                response="",
                model=use_model,
                response_time_ms=elapsed_ms,
                error_type="NETWORK_ERROR",
                error_message=str(e)[:120]
            )

# =====================================================================
# 4. Groq Provider (Ultra-Fast Cloud Inference)
# =====================================================================
class GroqProvider(BaseAIProvider):
    def __init__(self):
        super().__init__("groq", "Groq Cloud", 4, "openai/gpt-oss-120b")
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.fallback_models = [
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b",
            "groq/compound-mini"
        ]

    def get_api_key(self, custom_key: Optional[str] = None) -> Optional[str]:
        if custom_key and custom_key.strip():
            return custom_key.strip()
        return os.environ.get("GROQ_API_KEY", "").strip() or None

    def is_configured(self, custom_key: Optional[str] = None) -> bool:
        return bool(self.get_api_key(custom_key))

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
                response="",
                model=model or self.default_model,
                response_time_ms=0,
                error_type="NOT_CONFIGURED",
                error_message="Groq API key is not configured"
            )

        target_models = list(self.fallback_models)
        if model and model in target_models:
            target_models.remove(model)
            target_models.insert(0, model)
        elif model:
            target_models.insert(0, model)

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        for turn in conversation_history[-4:]:
            role = "user" if turn.get("role") == "user" else "assistant"
            content = turn.get("content", "").strip()
            if content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": prompt or "Hello"})
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        last_error = None
        for try_model in target_models:
            payload = {
                "model": try_model,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.7
            }

            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as http_client:
                    res = await http_client.post(self.endpoint, headers=headers, json=payload)
                    elapsed_ms = round((time.time() - start) * 1000, 1)

                    if res.status_code == 200:
                        data = res.json()
                        choices = data.get("choices", [])
                        if choices:
                            reply = choices[0].get("message", {}).get("content", "").strip()
                            return ProviderResponse(
                                success=True,
                                provider=self.provider_id,
                                response=reply,
                                model=try_model,
                                response_time_ms=elapsed_ms
                            )
                    elif res.status_code == 429:
                        last_error = "Groq rate limit exceeded (429)"
                        continue
                    elif res.status_code == 401:
                        return ProviderResponse(
                            success=False,
                            provider=self.provider_id,
                            response="",
                            model=try_model,
                            response_time_ms=elapsed_ms,
                            error_type="AUTH_ERROR",
                            error_message="Invalid Groq API Key (401)"
                        )
                    elif res.status_code == 404:
                        last_error = f"Model {try_model} not found on Groq"
                        continue
                    else:
                        last_error = f"Groq API Error: HTTP {res.status_code}"
                        continue
            except httpx.TimeoutException:
                last_error = f"Groq request timed out on {try_model}"
                continue
            except Exception as e:
                last_error = str(e)[:120]
                continue

        elapsed_ms = round((time.time() - start) * 1000, 1)
        return ProviderResponse(
            success=False,
            provider=self.provider_id,
            response="",
            model=target_models[0],
            response_time_ms=elapsed_ms,
            error_type="API_ERROR",
            error_message=last_error or "All Groq models failed"
        )

# =====================================================================
# 5. Local Ollama Provider (Final AI Fallback)
# =====================================================================
class OllamaProvider(BaseAIProvider):
    def __init__(self):
        super().__init__("ollama", "Local Ollama", 5, "llama3.2")
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    def is_configured(self, custom_key: Optional[str] = None) -> bool:
        return True

    async def generate(
        self,
        prompt: str,
        conversation_history: List[Dict[str, str]],
        image_base64: Optional[str] = None,
        system_instruction: Optional[str] = None,
        custom_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> ProviderResponse:
        start = time.time()
        use_model = model or os.environ.get("OLLAMA_MODEL", self.default_model)
        target_url = (base_url or self.base_url).rstrip("/") + "/api/chat"

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        for turn in conversation_history[-4:]:
            role = "user" if turn.get("role") == "user" else "assistant"
            content = turn.get("content", "").strip()
            if content:
                messages.append({"role": role, "content": content})

        user_msg_obj: Dict[str, Any] = {"role": "user", "content": prompt or "Hello"}
        if image_base64:
            b64_clean = image_base64.split(",")[-1]
            user_msg_obj["images"] = [b64_clean]
        messages.append(user_msg_obj)

        payload = {
            "model": use_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.7}
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as http_client:
                res = await http_client.post(target_url, json=payload)
                elapsed_ms = round((time.time() - start) * 1000, 1)

                if res.status_code == 200:
                    data = res.json()
                    msg = data.get("message", {})
                    reply = msg.get("content", "").strip()
                    return ProviderResponse(
                        success=True,
                        provider=self.provider_id,
                        response=reply,
                        model=use_model,
                        response_time_ms=elapsed_ms
                    )
                else:
                    return ProviderResponse(
                        success=False,
                        provider=self.provider_id,
                        response="",
                        model=use_model,
                        response_time_ms=elapsed_ms,
                        error_type="SERVER_ERROR",
                        error_message=f"Ollama HTTP {res.status_code}"
                    )
        except (httpx.ConnectError, httpx.ConnectTimeout):
            elapsed_ms = round((time.time() - start) * 1000, 1)
            return ProviderResponse(
                success=False,
                provider=self.provider_id,
                response="",
                model=use_model,
                response_time_ms=elapsed_ms,
                error_type="UNAVAILABLE",
                error_message="Ollama daemon is not running locally"
            )
        except httpx.TimeoutException:
            elapsed_ms = round((time.time() - start) * 1000, 1)
            return ProviderResponse(
                success=False,
                provider=self.provider_id,
                response="",
                model=use_model,
                response_time_ms=elapsed_ms,
                error_type="TIMEOUT",
                error_message="Ollama inference timed out"
            )
        except Exception as e:
            elapsed_ms = round((time.time() - start) * 1000, 1)
            return ProviderResponse(
                success=False,
                provider=self.provider_id,
                response="",
                model=use_model,
                response_time_ms=elapsed_ms,
                error_type="NETWORK_ERROR",
                error_message=str(e)[:120]
            )

# =====================================================================
# 5. Offline Knowledge Engine Provider (Final Fallback)
# =====================================================================
class OfflineProvider(BaseAIProvider):
    def __init__(self):
        super().__init__("offline", "Offline Knowledge Engine", 5, "vrixa-offline-knowledge-v2")

    def is_configured(self, custom_key: Optional[str] = None) -> bool:
        return True

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
        import random

        prompt_clean = (prompt or "").strip()
        prompt_lower = normalize_prompt(prompt_clean)

        # Keep common conversation useful even when every network provider is
        # unavailable. This path must never turn into an implicit web lookup.
        if is_conversational_request(prompt_clean):
            if any(phrase in prompt_lower for phrase in (
                "tell me about yourself", "tell me about you", "introduce yourself",
                "who are you", "what are you", "your name", "who are u", "what are u"
            )):
                response = "I’m Vrixa, your conversational AI assistant. I can answer questions, help with tasks, and create images when you ask."
            elif any(word in prompt_lower for word in ("how are you", "kaise ho", "kya haal")):
                response = "I’m doing well and ready to help. What would you like to work on?"
            else:
                response = random.choice([
                    "Hello! I’m Vrixa. How can I help you today?",
                    "Hey! I’m here and ready to help. What do you need?",
                    "Hello! Tell me what you would like to do."
                ])
            greetings = [
                response
            ]
            elapsed_ms = round((time.time() - start) * 1000, 1)
            return ProviderResponse(
                success=True,
                provider=self.provider_id,
                response=random.choice(greetings),
                model=self.default_model,
                response_time_ms=elapsed_ms
            )

        # Explicit Wikipedia requests remain available for direct callers.
        if not is_explicit_wikipedia_request(prompt_clean):
            elapsed_ms = round((time.time() - start) * 1000, 1)
            return ProviderResponse(
                success=False,
                provider=self.provider_id,
                response="",
                model=self.default_model,
                response_time_ms=elapsed_ms,
                error_type="OFFLINE_FALLBACK",
                error_message="No live provider returned a response"
            )

        import wikipedia

        query = extract_wikipedia_query(prompt_clean)
        wiki_summary = None
        if len(query) >= 4:
            try:
                wiki_summary = wikipedia.summary(query, sentences=2, auto_suggest=False)
            except Exception:
                try:
                    results = wikipedia.search(query)
                    if results:
                        wiki_summary = wikipedia.summary(results[0], sentences=2, auto_suggest=False)
                except Exception:
                    pass

        elapsed_ms = round((time.time() - start) * 1000, 1)
        if wiki_summary and len(wiki_summary.strip()) > 20:
            reply = f"📚 **According to Wikipedia**:\n{wiki_summary}"
            return ProviderResponse(
                success=True,
                provider=self.provider_id,
                response=reply,
                model=self.default_model,
                response_time_ms=elapsed_ms
            )
        else:
            reply = "I’m ready to help. Please ask a direct question or provide a live API key if you want high-quality cloud answers."
            return ProviderResponse(
                success=True,
                provider=self.provider_id,
                response=reply,
                model=self.default_model,
                response_time_ms=elapsed_ms
            )

# =====================================================================
# Central Multi-AI Orchestrator Engine
# =====================================================================
class MultiAIOrchestrator:
    def __init__(self):
        self.providers: Dict[str, BaseAIProvider] = {
            "gemini": GeminiProvider(),
            "groq": GroqProvider(),
            "claude": ClaudeProvider(),
            "openai": OpenAIProvider(),
            "ollama": OllamaProvider(),
            "offline": OfflineProvider()
        }
        self.priority_order = ["gemini", "groq", "claude", "openai", "ollama", "offline"]

    def get_provider(self, provider_id: str) -> Optional[BaseAIProvider]:
        return self.providers.get(provider_id.lower())

    def get_providers_status(self, custom_keys: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        custom_keys = custom_keys or {}
        status_list = []
        for pid in self.priority_order:
            if pid == "offline":
                continue
            prov = self.providers[pid]
            ckey = custom_keys.get(pid, "")
            status_list.append({
                "id": prov.provider_id,
                "name": prov.display_name,
                "priority": prov.priority,
                "enabled": prov.is_enabled,
                "configured": prov.is_configured(ckey),
                "model": prov.default_model,
                "masked_key": prov.get_masked_key(ckey)
            })
        return status_list

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

        fallback_log = []
        configured_cloud_provider = False

        for pid in self.priority_order:
            provider = self.providers.get(pid)
            if not provider:
                continue

            p_conf = provider_configs.get(pid, {})
            is_enabled = p_conf.get("enabled", provider.is_enabled)
            if not is_enabled:
                logger.info(f"[AI-Router] Skipping {provider.display_name} (Disabled by configuration)")
                continue

            target_model = p_conf.get("model", provider.default_model)
            custom_key = custom_keys.get(pid, "")
            base_url = p_conf.get("base_url", getattr(provider, "base_url", None))

            if pid in ["gemini", "claude", "openai"] and not provider.is_configured(custom_key):
                logger.info(f"[AI-Router] Skipping {provider.display_name} (No API key configured)")
                fallback_log.append(f"{provider.display_name}: No Key")
                continue

            if pid in ["gemini", "groq", "claude", "openai"] and provider.is_configured(custom_key):
                configured_cloud_provider = True

            logger.info(f"[AI-Router] Attempting Provider #{provider.priority}: {provider.display_name} (Model: {target_model})...")

            if pid == "ollama":
                resp = await provider.generate(
                    prompt=user_message,
                    conversation_history=conversation_history,
                    image_base64=image_base64,
                    system_instruction=system_instruction,
                    custom_key=custom_key,
                    model=target_model,
                    base_url=base_url
                )
            else:
                resp = await provider.generate(
                    prompt=user_message,
                    conversation_history=conversation_history,
                    image_base64=image_base64,
                    system_instruction=system_instruction,
                    custom_key=custom_key,
                    model=target_model
                )

            if resp.success and resp.response and resp.response.strip():
                logger.info(f"[AI-Router] SUCCESS! {provider.display_name} responded in {resp.response_time_ms}ms (Model: {resp.model})")
                return {
                    "success": True,
                    "provider": resp.provider,
                    "response": resp.response.strip(),
                    "model": resp.model,
                    "response_time_ms": resp.response_time_ms,
                    "fallback_log": fallback_log
                }
            else:
                err_summary = f"{resp.error_type or 'FAILED'}: {resp.error_message or 'No output'}"
                logger.warning(f"[AI-Router] {provider.display_name} Failed ({err_summary}). Falling back to next provider...")
                fallback_log.append(f"{provider.display_name} ({resp.error_type})")

        if not configured_cloud_provider:
            return {
                "success": False,
                "provider": "offline",
                "response": "No AI provider is configured. Add GEMINI_API_KEY to your local environment or enter a provider key in Settings.",
                "model": "configuration-required",
                "response_time_ms": 0,
                "fallback_log": fallback_log
            }

        return {
            "success": False,
            "provider": "offline",
            "response": "I’m having trouble reaching my AI services right now. Please try again in a moment.",
            "model": "fallback-offline",
            "response_time_ms": 0,
            "fallback_log": fallback_log
        }

# Global Orchestrator instance
ai_orchestrator = MultiAIOrchestrator()

# Helper export function matching the user spec
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
