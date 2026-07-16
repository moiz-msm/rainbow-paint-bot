"""
product_search.py
Searches products.json for relevant items and returns a compact
text block the LLM can use to answer a customer query.

Keeps prompts small (we never stuff all 4,030 entries into the LLM).
"""
import json
import re
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRODUCTS_PATH = os.path.join(BASE_DIR, "products.json")

# Tinting-base patterns to EXCLUDE unless explicitly asked
BASE_PAT = re.compile(r'BASE|\bBS\s*\d|\bCS\d|BS\d', re.IGNORECASE)

# Brand keyword hints (colloquial -> brand)
BRAND_HINTS = {
    "asian": "Asian Paints", "ap": "Asian Paints", "royale": "Asian Paints",
    "apex": "Asian Paints", "tractor": "Asian Paints", "ace": "Asian Paints",
    "apcolite": "Asian Paints", "woodtech": "Asian Paints", "nilaya": "Asian Paints",
    "berger": "Berger", "weathercoat": "Berger", "walmasta": "Berger",
    "bison": "Berger", "luxol": "Berger", "silk": "Berger", "imperia": "Berger",
    "bergthane": "Berger", "610": "Berger",
    "mrf": "MRF", "campus": "MRF", "altura": "MRF", "specta": "MRF",
    "ruca": "MRF", "metalcoat": "MRF", "zameen": "MRF", "aquafresh": "MRF",
    "epoxy": "ALL", "pu": "ALL", "enamel": "ALL", "primer": "ALL",
    "distemper": "ALL", "putty": "ALL", "emulsion": "ALL",
}


def load_products():
    with open(PRODUCTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["products"]


def is_base(name):
    return bool(BASE_PAT.search(name))


def detect_brands(text):
    t = text.lower()
    found = set()
    for kw, brand in BRAND_HINTS.items():
        if kw in t:
            if brand == "ALL":
                return None  # signal: any brand
            found.add(brand)
    return found if found else None


def search(query, max_results=25, allow_bases=False):
    """
    Returns a text block of matching products (cheapest size first per product)
    so the LLM can quote accurately. Excludes tinting bases by default.
    """
    products = load_products()
    q = query.lower()
    brands = detect_brands(q)

    # Extract a size hint like 20l, 40 ltr, 1l, 4 litre
    size_hint = None
    m = re.search(r'(\d+)\s*(l|lt|ltr|litre|liter|kg|ml)', q)
    if m:
        size_hint = m.group(1) + ("L" if m.group(2).startswith(("l", "lt")) else m.group(2).upper())

    # keyword tokens (drop common words). Keep short technical tokens like pu/2k.
    stop = {"paint", "for", "want", "need", "i", "a", "an", "the", "and", "or",
            "price", "stock", "with", "my", "home", "exterior", "interior", "white",
            "ltr", "lt", "l", "kg", "ml", "get", "need", "have", "do", "you", "is"}
    # Always-keep short technical tokens even if in stop or short
    keep_short = {"pu", "2k", "epoxy", "bs", "cs"}
    tokens = []
    for w in re.findall(r'[a-z0-9]+', q):
        if w in keep_short:
            tokens.append(w)
        elif w not in stop and len(w) > 2:
            tokens.append(w)

    scored = []
    for p in products:
        name = p["name"].upper()
        brand = p["brand"]
        if brands is not None and brand not in brands:
            continue
        if not allow_bases and is_base(name):
            continue
        # score by keyword hits in name
        score = 0
        for tok in tokens:
            if tok in name.lower():
                score += 2
            if tok in brand.lower():
                score += 1
        # boost if size matches
        if size_hint and size_hint.upper() in p["size"].upper():
            score += 3
        if score > 0:
            scored.append((score, p))

    # Sort by score desc, then price asc
    scored.sort(key=lambda x: (-x[0], x[1]["price"]))

    # De-duplicate by (brand, name) keeping cheapest size, cap results
    seen = set()
    lines = []
    for score, p in scored:
        key = (p["brand"], p["name"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"{p['brand']} | {p['name']} | {p['size']} | ₹{p['price']}"
        )
        if len(lines) >= max_results:
            break

    if not lines:
        return "(no matching products found in price list)"

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "berger epoxy primer 40 ltr"
    print("QUERY:", q)
    print("-" * 40)
    print(search(q))
