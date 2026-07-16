"""
whatsapp_bot.py
Official WhatsApp Business API (Meta Cloud API) bot for Rainbow Paint.

WHAT IT DOES
- Receives customer WhatsApp messages via Meta webhook
- Uses OpenRouter LLM (your key) + products.json to answer
- Follows the trained sales rules (white-default, no bases, full ladder,
  clarify, never invent -> hand off to human)

SETUP (one-time, by store owner)
1. Create a Meta Business + WhatsApp Business Account (free).
2. Get: WHATSAPP_TOKEN, PHONE_NUMBER_ID, VERIFY_TOKEN (your choice).
3. Host this file on a server with a public HTTPS URL.
4. Set webhook URL = https://<your-host>/webhook  (GET verify, POST receive)
5. Set env vars below, pip install flask openai, run: python whatsapp_bot.py

ENV VARS (put in .env or set in your host):
  WHATSAPP_TOKEN      = Meta Graph API permanent/system token
  PHONE_NUMBER_ID     = from Meta WhatsApp > API Setup
  VERIFY_TOKEN        = any string you choose, also set in Meta dashboard
  OPENROUTER_API_KEY  = your OpenRouter key (already configured for Hermes)
  PUBLIC_BASE_URL     = https://<your-host>   (for media links)
"""

import os
import json
import time
import requests
from flask import Flask, request, jsonify
from threading import Thread

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Import the product search we built
from product_search import search

app = Flask(__name__)

# Helper: read an env var under several common spellings (Render lowercases/
# kebab-cases keys). Case/hyphen-insensitive lookup.
def env(*names, default=""):
    for n in names:
        if n in os.environ and os.environ[n] != "":
            return os.environ[n]
    return default

WHATSAPP_TOKEN = env("WHATSAPP_TOKEN", "whatsapp-access-token", "whatsapp_access_token")
PHONE_NUMBER_ID = env("PHONE_NUMBER_ID", "phone-number-id", "phone_number_id")
VERIFY_TOKEN = env("VERIFY_TOKEN", "verify-token", "verify_token", default="rainbowpaint_verify")
OPENROUTER_API_KEY = env("OPENROUTER_API_KEY", "openrouter_api_key")
MODEL = env("BOT_MODEL", "bot_model", "bot-model", default="google/gemini-3.1-pro-preview")

GRAPH_URL = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"

# ---------------------------------------------------------------------------
# SALES BRAIN PROMPT  (this is the trained knowledge, condensed)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the WhatsApp sales assistant for Rainbow Paints & Hardwares, Coimbatore (rainbowpaint.in). You help customers buy paint from Asian Paints, Berger and MRF.

HARD RULES:
1. NEVER quote tinting bases (names with Base/BS/CS/P0/W1/N1 etc.) unless the customer explicitly asks for a base.
2. ALWAYS offer from ALL THREE brands (Asian Paints, Berger, MRF) when relevant.
3. DEFAULT starting price = WHITE / standard (non-base). After quoting white, ask "White, or a specific shade?" — never assume, but white is the default anchor.
4. Show ₹/L and total when a quantity is given. GST is EXTRA on all dealer prices — state "GST extra".
5. Present the FULL quality ladder (Basic -> Mid -> Premium -> Top/Luxury), never filter to one tier. Anchor high so customers can trade up.
6. "PU paint" from a layman = 2K PU (two-pack), NOT single-pack enamel. If unsure, ASK: surface? (metal/wood/floor) purpose? 1K or 2K?
7. "Epoxy paint" = coloured epoxy TOPCOAT, not primer. "Epoxy primer" is a different product.
8. Customer slang translation: "epoxy primer" usually = Berger 610 Coating Grey (2K epoxy); "PU" = 2K PU (MRF Metalcoat / Berger Bergthane / AP 2K PU); "royale"=Royale; "apex"=Apex; "walmasta"=Walmasta; "campus"=Campus; "weathercoat"=WeatherCoat.
9. NEVER INVENT a price, stock level, or product spec. If the customer NAMES a specific product and it's not in the provided data, send a SHORT holding reply like: "Sure, let me check our latest stock & price and get back to you shortly 😊 You can also reach us on wa.me/918072442930" — then a human takes over. Do NOT auto-follow with fabricated numbers. (A plain greeting like "hi" is NOT a product query — welcome the customer and ask what they need; do not use the holding reply for greetings.)
10. Keep replies SHORT, friendly, WhatsApp-style (use line breaks, ₹, emojis sparingly). End with a clarifying question to move the sale forward.
11. Refer customers to browse shades/products: rainbowpaint.in/buy-paint-online , shades at rainbowpaint.in/color/<shade> , visualizer at rainbowpaint.in/visualizer.
12. Store contact: WhatsApp wa.me/918072442930 , email rainbow_paint@hotmail.com , Coimbatore, open 9am-8pm.

Always answer in the customer's language if detectable; default English.
"""

HUMAN_HANDOFF_LINES = ("let me check", "get back to you", "confirm", "check with our team", "revert")


import re as _re

GREETING_RE = _re.compile(r"^(hi|hello|hey|hii|hlo|good\s*(morning|afternoon|evening)|namaste|vanakkam|sup|yo)[\s!\.]*$", _re.IGNORECASE)

def llm_reply(customer_text):
    """Call OpenRouter LLM with the product context and return a reply."""
    # Greeting-only messages: welcome, don't hand off to human.
    if GREETING_RE.match(customer_text.strip()) or customer_text.strip().lower() in ("hi", "hello", "hey", "hii", "hlo", "namaste", "vanakkam"):
        return ("👋 Hi! Welcome to *Rainbow Paints & Hardwares*, Coimbatore 🎨\n"
                "Tell me what you're looking for — interior/exterior paint, primer, PU, epoxy, "
                "wood/metal finishes — and I'll quote the price (GST extra) from Asian Paints, Berger & MRF.\n"
                "Or browse 4000+ shades: rainbowpaint.in/color/")
    if not OPENROUTER_API_KEY:
        return ("Sure, let me check our latest price & stock and get back to you shortly 😊 "
                "You can also reach us on wa.me/918072442930")
    if OpenAI is None:
        return ("Our assistant is warming up — please message us on wa.me/918072442930 "
                "and our team will help right away.")
    try:
        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
        # Pull relevant products to ground the answer (no invention)
        ctx = search(customer_text, max_results=20)
        user_msg = (
            f"CUSTOMER MESSAGE: {customer_text}\n\n"
            f"RELEVANT PRODUCTS FROM OUR PRICE LIST (dealer price, GST extra):\n{ctx}\n\n"
            f"(If the product is not in the list above, follow rule 9: short holding reply, no invented price.)"
        )
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=400,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # Fail safe: hand off to human, never invent
        return ("Sure, let me check our latest stock & price and get back to you shortly 😊 "
                "You can also reach us on wa.me/918072442930")


def send_whatsapp(recipient_wa_id, text):
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        print("[send_whatsapp] Token/Phone ID not set; skipping send.")
        return False
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_wa_id,
        "type": "text",
        "text": {"preview_url": True, "body": text},
    }
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    try:
        r = requests.post(GRAPH_URL, json=payload, headers=headers, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print("[send_whatsapp] error:", e)
        return False


@app.route("/webhook", methods=["GET"])
def verify():
    # Meta webhook verification
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "Verification failed", 403


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True) or {}
    # Return 200 immediately so Meta doesn't retry, then process async.
    Thread(target=process_messages, args=(data,), daemon=True).start()
    return jsonify({"status": "ok"}), 200


def process_messages(data):
    try:
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                if change.get("field") != "messages":
                    continue
                vals = change.get("value", {})
                for msg in vals.get("messages", []):
                    if msg.get("type") != "text":
                        continue
                    wa_id = msg["from"]
                    body = msg["text"]["body"]
                    print(f"[msg] {wa_id}: {body}")
                    reply = llm_reply(body)
                    print(f"[reply] {reply[:120]}...")
                    send_whatsapp(wa_id, reply)
    except Exception as e:
        print("[process_messages] error:", e)


# Local test endpoint (no WhatsApp needed) — send {"text":"..."}
@app.route("/test", methods=["POST"])
def test():
    txt = (request.get_json(silent=True) or {}).get("text", "")
    return jsonify({"reply": llm_reply(txt)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Rainbow Paint WhatsApp bot starting on :{port}")
    print(f"OpenRouter key set: {bool(OPENROUTER_API_KEY)} | WhatsApp token set: {bool(WHATSAPP_TOKEN)}")
    app.run(host="0.0.0.0", port=port, debug=False)
