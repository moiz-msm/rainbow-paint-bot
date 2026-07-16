"""
whatsapp_bot.py
Official WhatsApp Business API (Meta Cloud API) bot for Rainbow Paint.

WHAT IT DOES
- Receives customer WhatsApp messages via Meta webhook
- Uses OpenRouter LLM (your key) + products.json to answer
- Follows trained sales rules (white-default, all 3 brands, full ladder, clarify)
- SELF-LEARNING: remembers the conversation, reflects after each reply into
  learnings.json, and web-verifies unknown products — WITHOUT ever inventing a
  price, stock level, or product.

SETUP
1. Meta Business + WhatsApp Business Account (free).
2. Env vars: WHATSAPP_TOKEN / PHONE_NUMBER_ID / VERIFY_TOKEN / OPENROUTER_API_KEY / BOT_MODEL
   (Render kebab-case names also accepted: whatsapp-access-token, phone-number-id, etc.)
3. Host on a public HTTPS server (Render free tier). Webhook = https://<host>/webhook
4. pip install flask openai requests ; run: python whatsapp_bot.py
"""

import os
import re
import json
import time
import html
import requests
import urllib.parse
from flask import Flask, request, jsonify
from threading import Thread

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from product_search import search

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Env (Render lowercases / kebab-cases keys; look up several spellings)
# ---------------------------------------------------------------------------
def env(*names, default=""):
    for n in names:
        if n in os.environ and os.environ[n] != "":
            return os.environ[n]
    return default

WHATSAPP_TOKEN  = env("WHATSAPP_TOKEN", "whatsapp-access-token", "whatsapp_access_token")
PHONE_NUMBER_ID = env("PHONE_NUMBER_ID", "phone-number-id", "phone_number_id")
VERIFY_TOKEN    = env("VERIFY_TOKEN", "verify-token", "verify_token", default="rainbowpaint_verify")
OPENROUTER_API_KEY = env("OPENROUTER_API_KEY", "openrouter_api_key")
MODEL = env("BOT_MODEL", "bot_model", "bot-model", default="meta-llama/llama-3.3-70b-instruct:free")

# Fallback chain. Free models share ONE global rate-limit pool on OpenRouter, so
# during peak hours ALL free models can 429 at once. The LAST entry is a cheap
# PAID model that is always available — a guaranteed floor so the customer never
# gets the "hiccup" message. Primary = whatever you set in BOT_MODEL.
# Free models verified live 2026-07-16; the paid floor (mistral-nemo) is ~$0.02/1M
# in / $0.04/1M out (~$0.00001 per reply). Requires OpenRouter account credit.
# If a free model is retired by OpenRouter (404), just remove it from the list.
MODEL_CHAIN = [MODEL] + [m for m in (
    "meta-llama/llama-3.2-3b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "mistralai/mistral-nemo",          # cheap paid floor — always available
) if m != MODEL]

GRAPH_URL = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"

# ---------------------------------------------------------------------------
# GitHub backup of learning state (survives Render deploys / disk resets)
# Set: GITHUB_TOKEN (a fine-grained PAT with repo write), GITHUB_REPO = owner/name
# Optional: GITHUB_BRANCH (default main)
# ---------------------------------------------------------------------------
GITHUB_TOKEN  = env("GITHUB_TOKEN", "github-token", "github_token")
GITHUB_REPO   = env("GITHUB_REPO", "github-repo", "github_repo", default="moiz-msm/rainbow-paint-bot")
GITHUB_BRANCH = env("GITHUB_BRANCH", "github-branch", "github_branch", default="main")
GITHUB_STATE_FILES = ("learnings.json", "chat_memory.json")
_SYNC_LOCK = __import__("threading").Lock()
_last_push = {"t": 0.0}

# ---------------------------------------------------------------------------
# Persistent files (self-learning). NOTE: Render free tier resets local disk
# on each NEW DEPLOY — learnings persist across sleeps but not across deploys.
# (Offer: back learnings.json to GitHub for true persistence.)
# ---------------------------------------------------------------------------
MEMORY_FILE   = "chat_memory.json"
LEARNINGS_FILE = "learnings.json"
MEMORY_LIMIT  = 12  # rolling window of messages kept per customer

def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("[save_json] error:", e)

_memory = load_json(MEMORY_FILE, {})

def get_history(wa_id):
    return _memory.get(wa_id, [])

def append_history(wa_id, role, text):
    hist = _memory.setdefault(wa_id, [])
    hist.append({"role": role, "text": text})
    if len(hist) > MEMORY_LIMIT:
        del hist[: len(hist) - MEMORY_LIMIT]
    save_json(MEMORY_FILE, _memory)

def clear_history(wa_id):
    _memory.pop(wa_id, None)
    save_json(MEMORY_FILE, _memory)

# ---------------------------------------------------------------------------
# GitHub sync (pull on boot, push after learn / every 5 min)
# ---------------------------------------------------------------------------
def _gh_headers():
    return {"Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json"}

def _gh_path(fname):
    return f"repos/{GITHUB_REPO}/contents/{fname}"

def github_pull_state():
    """On boot: download learnings.json + chat_memory.json if present in repo."""
    if not GITHUB_TOKEN:
        return
    for fname in GITHUB_STATE_FILES:
        try:
            r = requests.get(f"https://api.github.com/{_gh_path(fname)}?ref={GITHUB_BRANCH}",
                             headers=_gh_headers(), timeout=15)
            if r.status_code == 200:
                import base64
                content = base64.b64decode(r.json()["content"]).decode("utf-8")
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(content)
                if fname == MEMORY_FILE:
                    global _memory
                    _memory = json.loads(content)
                print(f"[gh] pulled {fname}")
        except Exception as e:
            print(f"[gh] pull {fname} failed:", e)

def github_push_state(force=False):
    """Upload learnings.json + chat_memory.json to the repo (best-effort, throttled)."""
    if not GITHUB_TOKEN:
        return
    now = time.time()
    if not force and now - _last_push["t"] < 60:
        return
    with _SYNC_LOCK:
        _last_push["t"] = now
        headers = _gh_headers()
        for fname in GITHUB_STATE_FILES:
            try:
                if not os.path.exists(fname):
                    continue
                with open(fname, "r", encoding="utf-8") as f:
                    data = f.read()
                gr = requests.get(f"https://api.github.com/{_gh_path(fname)}?ref={GITHUB_BRANCH}",
                                  headers=headers, timeout=15)
                sha = gr.json().get("sha") if gr.status_code == 200 else None
                import base64
                payload = {"message": f"auto: update {fname} ({time.strftime('%Y-%m-%d %H:%M')})",
                           "content": base64.b64encode(data.encode("utf-8")).decode("ascii"),
                           "branch": GITHUB_BRANCH}
                if sha:
                    payload["sha"] = sha
                pr = requests.put(f"https://api.github.com/{_gh_path(fname)}",
                                  headers=headers, json=payload, timeout=15)
                if pr.status_code in (200, 201):
                    print(f"[gh] pushed {fname}")
                else:
                    print(f"[gh] push {fname} failed:", pr.status_code, pr.text[:120])
            except Exception as e:
                print(f"[gh] push {fname} error:", e)

def github_periodic_sync():
    while True:
        time.sleep(300)
        try:
            github_push_state(force=True)
        except Exception as e:
            print("[gh] periodic sync error:", e)

# ---------------------------------------------------------------------------
# Web verification (keyless DuckDuckGo HTML). Used ONLY to confirm a product/
# brand is real — never to fetch or invent a price.
# ---------------------------------------------------------------------------
def web_search(query, max_results=3):
    try:
        q = urllib.parse.quote_plus(query)
        url = "https://html.duckduckgo.com/html/?q=" + q
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(url, headers=headers, timeout=8)
        titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', r.text, re.S)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r.text, re.S)
        def clean(s):
            s = re.sub(r"<[^>]+>", "", s)
            return html.unescape(s).strip()
        out = []
        n = min(max_results, max(len(titles), len(snippets)))
        for i in range(n):
            t = clean(titles[i]) if i < len(titles) else ""
            s = clean(snippets[i]) if i < len(snippets) else ""
            if t or s:
                out.append(f"- {t}: {s}")
        return "\n".join(out) if out else "No web results found."
    except Exception as e:
        return f"(web search unavailable: {e})"

_PAINT_KW = ("paint", "primer", "coating", "emulsion", "enamel", " pu ", "pu paint",
             "epoxy", "putty", "distemper", "colour", "color", "lacquer", "wood",
             "metal", "lt", "ltr", "litre", "liter", "royale", "apex", "berg",
             "asian", "mrf", "nippon", "dulux", "weathercoat", "walmasta", "campus")
_GREET_ONLY = ("hi", "hello", "hey", "namaste", "vanakkam", "good morning",
               "good afternoon", "good evening", "thank", "thanks", "ok", "okay")

def looks_like_product_query(text):
    t = " " + text.lower() + " "
    if any(t.startswith(g) and len(text) <= len(g) + 3 for g in
           ("hi", "hello", "hey", "namaste", "vanakkam", "good morning",
            "good afternoon", "good evening", "thank", "thanks", "ok", "okay")):
        return False
    if any(k in t for k in _PAINT_KW):
        return True
    return bool(re.search(r"\b\d+\s*(lt|ltr|litre|l|kg)\b", t)) or bool(re.search(r"\b\d[pP]\d{3,}\b", text))


def product_grounded(customer_text, ctx):
    """True only if the search results actually match the customer's product
    terms (not just a stray size token like '20L'). Prevents false positives
    that would skip web verification for products we don't carry."""
    if "no matching" in ctx.lower():
        return False
    c = customer_text.lower()
    # extract meaningful tokens: words >=4 chars that are letters, plus brand/SKU tokens
    toks = set(re.findall(r"[a-z]{4,}|\b\d[pP]\d{3,}\b", c))
    # drop generic paint words so they don't count as a 'match'
    generic = {"paint", "primer", "coating", "emulsion", "enamel", "colour", "color",
               "metal", "wood", "white", "wall", "interior", "exterior", "litre", "liter",
               "base", "top", "coat", "finish", "for", "the", "and", "ltr", "with"}
    toks -= generic
    if not toks:
        return True  # nothing specific to match; treat as grounded (e.g. 'paint for wall')
    cl = ctx.lower()
    # require at least one specific token to appear in results
    return any(tok in cl for tok in toks)

# ---------------------------------------------------------------------------
# SALES BRAIN PROMPT
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
9. NEVER invent a price, stock level, or product spec. If our price list doesn't have a product the customer names, you MAY use web_search to check if it's a REAL product/brand. If real but we don't carry/price it, say so honestly and offer to confirm with the store — never state a price you haven't verified from our list. If unclear/unreal, guide to what we do carry.
10. Keep replies SHORT, friendly, WhatsApp-style (line breaks, ₹, emojis sparingly). End with a clarifying question to move the sale forward.
11. Refer customers to browse shades/products: rainbowpaint.in/buy-paint-online , shades at rainbowpaint.in/color/<shade> , visualizer at rainbowpaint.in/visualizer.
12. Store contact: WhatsApp wa.me/918072442930 , email rainbow_paint@hotmail.com , Coimbatore, open 9am-8pm.
13. You improve over time: you receive PAST LEARNINGS (slang, product gaps, better phrasing) and the current conversation history. Apply them naturally — don't repeat them verbatim.
14. Always answer in the customer's language if detectable; default English.

Reason about every message. A plain greeting is just a hello — welcome and ask what they need.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": ("Verify whether a paint product, brand, or term the customer mentioned is REAL "
                            "(e.g. confirm 'Berger Easy Clean' or 'Nippon' is a genuine product/brand). "
                            "Use ONLY to verify facts, NEVER to invent or fetch a price."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "search query, e.g. 'Berger Easy Clean paint 20 litre'"}
                },
                "required": ["query"],
            },
        },
    }
]

def _openai_client():
    return OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

def _is_rate_limit(err):
    """True for transient errors we should retry on: 429 rate-limit, 404
    'no endpoints' (model removed/dead), 503/timeout. NOT auth/400 (bad key)."""
    try:
        msg = str(getattr(err, "message", err))
    except Exception:
        msg = str(err)
    low = msg.lower()
    if any(s in low for s in ("429", "404", "no endpoints", "503", "timeout",
                              "rate", "rate-limited", "temporarily", "unavailable")):
        return True
    code = getattr(err, "status_code", None)
    if code in (429, 404, 503):
        return True
    return False

def _sim(a, b):
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return False
    return len(sa & sb) / min(len(sa), len(sb)) > 0.6

# ---------------------------------------------------------------------------
# Core: LLM reply with memory + tool-based web verification + self-learning
# ---------------------------------------------------------------------------
def llm_reply(wa_id, customer_text):
    """Every reply is generated by the LLM from reasoning. No hardcoded/templated
    text, no keyword gating. Conversation memory + past learnings are injected;
    unknown products are web-verified (never invented)."""
    if not OPENROUTER_API_KEY or OpenAI is None:
        return ("Thanks for messaging Rainbow Paints! Our team will get back to you shortly. "
                "You can also reach us on wa.me/918072442930")

    try:
        history = get_history(wa_id)
        ctx = search(customer_text, max_results=20)
        learnings = load_json(LEARNINGS_FILE, [])
        learnings_ctx = "\n".join(f"- {l['note']}" for l in learnings[-15:]) if learnings else "(none yet)"
        sys_prompt = SYSTEM_PROMPT + "\n\nPAST LEARNINGS (apply automatically, never repeat verbatim):\n" + learnings_ctx

        messages = [{"role": "system", "content": sys_prompt}]
        for m in history:
            messages.append({"role": m["role"], "content": m["text"]})
        user_msg = (
            f"CUSTOMER MESSAGE: {customer_text}\n\n"
            f"OUR PRICE LIST (dealer price, GST extra) — only use these numbers, never invent:\n{ctx}\n\n"
            f"Reason about the message and reply naturally. If the customer names a product/brand not in "
            f"our list, call web_search to verify it's real. If we don't carry it, say so honestly and "
            f"suggest alternatives or offer to check with the store. Never invent a price."
        )
        messages.append({"role": "user", "content": user_msg})

        # ---- Try each model in the fallback chain (primary first) ----
        last_err = None
        for model in MODEL_CHAIN:
            try:
                client = _openai_client()
                reply = _llm_agent_pass(client, model, messages, customer_text, ctx)
                if reply is not None:
                    # Persist conversation memory
                    append_history(wa_id, "user", customer_text)
                    append_history(wa_id, "assistant", reply)
                    # Self-learning (async, doesn't delay the customer)
                    Thread(target=reflect_and_learn, args=(customer_text, reply, ctx, wa_id), daemon=True).start()
                    return reply
            except Exception as e:
                if _is_rate_limit(e):
                    print(f"[llm_reply] {model} rate-limited (429), trying next model…")
                    last_err = e
                    continue
                # Non-rate-limit error: don't burn the chain, surface it
                print("[llm_reply] error:", e)
                last_err = e
                break
        # All models failed (or non-429 error)
        print("[llm_reply] all models failed. last_err:", last_err)
        return ("Thanks for your message! I had a small hiccup fetching that — our team will "
                "confirm shortly. Reach us on wa.me/918072442930")
    except Exception as e:
        print("[llm_reply] fatal:", e)
        return ("Thanks for your message! I had a small hiccup fetching that — our team will "
                "confirm shortly. Reach us on wa.me/918072442930")


def _llm_agent_pass(client, model, base_messages, customer_text, ctx):
    """Run the agentic tool loop for one model. Returns reply text or None."""
    messages = list(base_messages)
    MAX_ROUNDS = 3
    searched = False
    reply = None
    for _ in range(MAX_ROUNDS):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, tool_choice="auto",
            max_tokens=600, temperature=0.6,
        )
        msg = resp.choices[0].message
        if getattr(msg, "tool_calls", None):
            tc_out = []
            for tc in msg.tool_calls:
                tc_out.append({"id": tc.id, "type": "function",
                               "function": {"name": tc.function.name,
                                            "arguments": tc.function.arguments}})
                if tc.function.name == "web_search":
                    args = json.loads(tc.function.arguments or "{}")
                    result = web_search(args.get("query", customer_text))
                    searched = True
                    messages.append({"role": "tool", "content": result, "tool_call_id": tc.id})
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": tc_out})
            continue
        reply = msg.content.strip()
        break

    if reply is None:
        return None

    # Proactive verification fallback (if model didn't call the tool but the
    # product isn't truly grounded in our list, web-verify it before replying)
    if not searched and not product_grounded(customer_text, ctx):
        verdict = web_search("paint " + customer_text)
        messages.append({"role": "user", "content":
            f"WEB VERIFICATION (real product check, NOT our price):\n{verdict}\n\n"
            f"Now reply honestly using this — if it's a real product we may carry, say we'll confirm "
            f"price/stock; never invent a price."})
        resp = client.chat.completions.create(model=model, messages=messages,
                                              max_tokens=600, temperature=0.6)
        reply = resp.choices[0].message.content.strip()
    return reply



def reflect_and_learn(customer_text, reply, ctx, wa_id):
    """After each reply, distill ONE new durable learning (no prices)."""
    try:
        if not OPENROUTER_API_KEY or OpenAI is None:
            return
        client = _openai_client()
        learnings = load_json(LEARNINGS_FILE, [])
        existing = "\n".join(f"- {l['note']}" for l in learnings[-20:]) or "(none)"
        prompt = (
            "You maintain learnings for a paint-store WhatsApp assistant. From this exchange, write at "
            "most ONE concise new learning line worth remembering for future chats: a customer "
            "slang->product mapping, a product/brand GAP (asked for but not in our list), or a better "
            "phrasing. Do NOT include prices or stock numbers. If nothing new, reply with exactly NONE.\n\n"
            f"CUSTOMER: {customer_text}\nREPLY: {reply}\nPRODUCTS MATCHED: {ctx[:500]}\n\n"
            f"EXISTING LEARNINGS:\n{existing}"
        )
        last_err = None
        r = None
        for model in MODEL_CHAIN:
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system",
                               "content": "Record concise, durable learnings for a paint store bot. No prices."},
                              {"role": "user", "content": prompt}],
                    max_tokens=140, temperature=0.3,
                )
                break
            except Exception as e:
                if _is_rate_limit(e):
                    print(f"[reflect_and_learn] {model} rate-limited (429), trying next…")
                    last_err = e
                    continue
                last_err = e
                break
        if r is None:
            print("[reflect_and_learn] all models failed:", last_err)
            return
        line = r.choices[0].message.content.strip()
        if line and line.upper() != "NONE":
            if not any(_sim(line, l["note"]) for l in learnings[-8:]):
                learnings.append({"date": time.strftime("%Y-%m-%d"), "note": line})
                if len(learnings) > 400:
                    learnings = learnings[-400:]
                save_json(LEARNINGS_FILE, learnings)
                print("[learn] +", line)
                github_push_state()  # persist new learning to GitHub
    except Exception as e:
        print("[reflect_and_learn] error:", e)

# ---------------------------------------------------------------------------
# WhatsApp send
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------
@app.route("/webhook", methods=["GET"])
def verify():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "Verification failed", 403

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True) or {}
    Thread(target=process_messages, args=(data,), daemon=True).start()
    return jsonify({"status": "ok"}), 200

_THANKS = ("thank", "thanks", "tnx", "ok bye", "that's all", "thats all", "done")

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
                    reply = llm_reply(wa_id, body)
                    print(f"[reply] {reply[:120]}...")
                    send_whatsapp(wa_id, reply)
                    if body.strip().lower() in _THANKS:
                        clear_history(wa_id)
    except Exception as e:
        print("[process_messages] error:", e)

# Local test endpoint (no WhatsApp needed) — {"text":"...","wa_id":"..."}
@app.route("/test", methods=["POST"])
def test():
    payload = request.get_json(silent=True) or {}
    txt = payload.get("text", "")
    wid = payload.get("wa_id", "test")
    return jsonify({"reply": llm_reply(wid, txt)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Rainbow Paint WhatsApp bot starting on :{port}")
    print(f"OpenRouter key set: {bool(OPENROUTER_API_KEY)} | WhatsApp token set: {bool(WHATSAPP_TOKEN)}")
    # Restore learning state from GitHub so it survives deploys/disk resets
    github_pull_state()
    if GITHUB_TOKEN:
        Thread(target=github_periodic_sync, daemon=True).start()
    app.run(host="0.0.0.0", port=port, debug=False)
