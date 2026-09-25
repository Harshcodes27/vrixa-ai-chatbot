"""Routing middleware for the deployed Vrixa FastAPI app.

This module keeps the existing app and UI intact while placing intent routing
before the legacy /api/chat handler. It is loaded by Render's start command.
"""
import logging
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict

from fastapi import Request

import app as legacy_app
from ai_providers import generateAIResponse

logger = logging.getLogger("VRIXA_REQUEST_ROUTER")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def conversational(text: str) -> bool:
    value = normalize(text)
    if not value:
        return True
    # Explicitly protect self-reference. This must be checked before generic
    # "tell me about" or "what is" rules.
    if re.search(r"\b(?:tell|say|talk)\s+(?:me\s+)?about\s+(?:yourself|you)\b", value):
        return True
    if re.search(r"\b(?:can|could|would|please)\s+you\s+(?:tell|say|talk)\s+(?:me\s+)?about\s+(?:yourself|you)\b", value):
        return True
    phrases = (
        "introduce yourself", "who are you", "what are you", "what is your name",
        "what's your name", "how are you", "kaise ho", "kya haal", "tum kaun ho",
        "tum kon ho", "who made you", "who created you"
    )
    if any(p in value for p in phrases):
        return True
    tokens = set(re.findall(r"\w+", value))
    greetings = {"hi", "hello", "hey", "hii", "namaste", "hola", "sup", "yo", "vrixa", "ok"}
    return bool(tokens) and len(tokens) <= 3 and tokens <= greetings


def explicit_wikipedia(text: str) -> bool:
    value = normalize(text)
    return bool(re.search(r"\b(?:wikipedia|wiki)\b", value))


def current_information(text: str) -> bool:
    value = normalize(text)
    current_words = ("current", "currently", "latest", "today", "now", "recent", "as of")
    time_sensitive_topics = ("prime minister", "president", "news", "price", "stock", "score", "election", "weather")
    return any(w in value for w in current_words) and any(w in value for w in time_sensitive_topics)


def wikipedia_query(text: str) -> str:
    value = normalize(text)
    value = re.sub(r"^.*?according to wikipedia[, ]*", "", value)
    value = re.sub(r"^(?:search|look up)\s+(?:on\s+)?(?:wikipedia|wiki)\s*(?:for|about)?\s*", "", value)
    value = re.sub(r"\b(?:wikipedia|wiki)\b", "", value)
    value = re.sub(r"^(?:tell me about|what is|who is|who was|what was|describe|explain)\s+", "", value)
    return re.sub(r"\s+", " ", value).strip(" ?!.,")


def web_lookup(text: str) -> str:
    """Small current-information provider with no result cache fallback."""
    query = urllib.parse.quote(text)
    url = f"https://www.google.com/search?q={query}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=6) as response:
            body = response.read().decode("utf-8", errors="ignore")
        # Keep this deliberately conservative: never invent an answer from a
        # stale local result. The main AI receives the current search context.
        snippets = re.findall(r"<div[^>]*>([^<>]{40,300})</div>", body)
        clean = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s)).strip() for s in snippets]
        clean = [s for s in clean if text.lower() not in s.lower()][:3]
        if clean:
            return "🌐 Current web information:\n" + "\n\n".join(f"• {s}" for s in clean)
    except Exception as exc:
        logger.warning("current-information lookup failed: %s", exc)
    return "I couldn’t retrieve current web information right now. Please try again shortly."


def public_payload(result: Dict[str, Any], session: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
    # Provider/model/fallback diagnostics stay in logs, never in the API reply.
    return {
        "reply": result.get("reply", "I’m ready to help."),
        "image_url": result.get("image_url"),
        "action_url": result.get("action_url"),
        "api_status": result.get("api_status", "online"),
        "timestamp": timestamp,
        "session": session,
    }


async def routed_chat(req: Any) -> Dict[str, Any]:
    sid = req.session_id
    now = legacy_app.get_ist_now()
    timestamp = now.strftime("%I:%M %p")
    session = legacy_app.sessions_db.setdefault(sid, {
        "id": sid, "title": "New Chat", "messages": [
            {"sender": "bot", "text": legacy_app.INITIAL_GREETING, "timestamp": timestamp},
            {"sender": "bot", "text": legacy_app.VRIXA_INTRODUCTION, "timestamp": timestamp}
        ],
        "awaiting_name": True, "context": []
    })
    user_msg = (req.message or "").strip()
    text = normalize(user_msg)
    legacy_process_chat = legacy_app.process_chat

    if explicit_wikipedia(user_msg):
        query = wikipedia_query(user_msg)
        if not query:
            reply = "Please tell me which topic you want to look up on Wikipedia."
        else:
            try:
                import wikipedia
                reply = f"📖 **According to Wikipedia**:\n{wikipedia.summary(query, sentences=2, auto_suggest=False)}"
            except Exception:
                reply = "I couldn’t find a matching Wikipedia article for that request."
        engine = "wikipedia"
        status = "online"
    elif current_information(user_msg):
        reply = web_lookup(user_msg)
        engine = "web"
        status = "online"
    elif conversational(user_msg):
        session["messages"].append({"sender": "user", "text": user_msg, "timestamp": timestamp})
        if session.get("title") == "New Chat":
            session["title"] = user_msg[:22]
        reply_result = await generateAIResponse(
            userMessage=user_msg,
            conversationHistory=session["context"],
            imageBase64=req.image_base64,
            systemInstruction=(
                "You are Vrixa, a warm conversational AI assistant created by Harsh. "
                "Answer self-introduction questions about Vrixa, never redirect them to Wikipedia. "
                "Do not mention providers, quotas, API errors, routing, or debugging. "
                "Do not assume the user's name; ask naturally when appropriate."
            ),
            customKeys={**(req.custom_keys or {}), **({"gemini": req.api_key} if req.api_key else {})},
            providerConfigs=req.provider_configs,
        )
        reply = reply_result.get("response") or "I’m having trouble reaching my AI services right now. Please try again in a moment."
        engine = reply_result.get("provider", "main-ai")
        status = "online" if reply_result.get("success") else "offline"
        logger.info("request engine=%s intent=conversation", engine)
    else:
        # Preserve every existing local tool/action/UI behavior for requests we
        # do not explicitly reroute.
        result = await legacy_process_chat(req)
        result.pop("provider", None)
        result.pop("model", None)
        result.pop("fallback_log", None)
        return result

    logger.info("request engine=%s query=%r", engine, text[:120])
    session["context"].extend([{"role": "user", "content": user_msg}, {"role": "assistant", "content": reply}])
    session["messages"].append({"sender": "bot", "text": reply, "timestamp": timestamp})
    return public_payload({"reply": reply, "api_status": status}, session, timestamp)


# Replace only the legacy chat route. All other existing endpoints/UI remain.
for route in list(legacy_app.app.router.routes):
    if getattr(route, "path", None) == "/api/chat":
        legacy_app.app.router.routes.remove(route)

@legacy_app.app.post("/api/chat")
async def process_chat(req: legacy_app.ChatRequest):
    try:
        return await routed_chat(req)
    except Exception:
        logger.exception("request routing failed")
        return {"reply": "I’m sorry, I couldn’t process that request right now. Please try again.", "api_status": "offline"}

app = legacy_app.app
