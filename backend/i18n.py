"""
Multilingual query normalization.

Why this exists: the embedding model (paraphrase-multilingual-mpnet-base-v2)
is trained to place semantically similar sentences from DIFFERENT languages
close together in vector space, so in principle a Hindi query should match
an English standard title without translation. In practice, quality varies
a lot by language and by how technical the vocabulary is (IS titles use
fairly formal/technical English, which multilingual models handle less
reliably than everyday sentences).

So this module does two things instead of relying on the embedding model's
cross-lingual ability alone:
  1. Detects the query's language.
  2. If it isn't English, translates it to English before embedding, so the
     query is compared against the corpus in the SAME language the corpus
     is actually written in. This is more reliable than pure cross-lingual
     embedding matching for technical/legal text.

Translation uses `deep-translator` (free, hits Google Translate's public
endpoint) — this needs internet access, which this sandbox does not have,
so I could not run the self-test below myself. Run `python i18n.py` on
your machine before your demo to confirm real results, and swap in
IndicTrans2 (run locally, no external API dependency, better for
Indian-language technical text) for a production/offline deployment.
"""

import os
import time

import requests
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator, MyMemoryTranslator

load_dotenv()
DetectorFactory.seed = 0  # deterministic langdetect output

GOOGLE_TRANSLATE_API_KEY = os.environ.get("GOOGLE_TRANSLATE_API_KEY", "")


def _translate_official_api(text: str) -> str:
    """Google Cloud Translation API v2 with a real API key — simple REST
    call, no SDK needed. Has a documented, generous free tier (500,000
    characters/month) and a real rate limit, instead of deep-translator's
    undocumented scraping of the free consumer website (which is what
    caused the earlier 'too many requests' errors)."""
    resp = requests.post(
        "https://translation.googleapis.com/language/translate/v2",
        params={"key": GOOGLE_TRANSLATE_API_KEY},
        json={"q": text, "target": "en", "format": "text"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["data"]["translations"][0]["translatedText"]


def _translate_with_retry(text: str, retries: int = 2, delay_seconds: float = 1.5) -> str:
    """Prefer the official, keyed API when configured. Otherwise fall back
    to the free scraping-based providers, with retry + backoff, so the
    feature still works without any setup — just less reliably under load."""
    if GOOGLE_TRANSLATE_API_KEY:
        try:
            return _translate_official_api(text)
        except Exception as e:
            print(f"Official Google Translate API call failed, falling back: {e}")

    last_error = None
    for attempt in range(retries):
        try:
            return GoogleTranslator(source="auto", target="en").translate(text)
        except Exception as e:
            last_error = e
            time.sleep(delay_seconds)

    try:
        return MyMemoryTranslator(source="autodetect", target="en-GB").translate(text)
    except Exception:
        raise last_error


def normalize_query(text: str) -> dict:
    """Returns {"original": ..., "detected_lang": ..., "english": ...}"""
    try:
        lang = detect(text)
    except Exception:
        lang = "en"

    if lang == "en":
        return {"original": text, "detected_lang": "en", "english": text}

    try:
        translated = _translate_with_retry(text)
    except Exception as e:
        # Fail open: fall back to the original text and let the multilingual
        # embedding model take its best shot, rather than erroring out.
        return {"original": text, "detected_lang": lang, "english": text,
                "translation_error": str(e)}

    return {"original": text, "detected_lang": lang, "english": translated}


# ---------------------------------------------------------------------------
# Self-test: run this file directly on a machine with internet to verify
# multilingual handling actually works before you demo it to judges.
# ---------------------------------------------------------------------------
SAMPLE_QUERIES = [
    ("hi", "आरसीसी निर्माण कार्य के लिए 53 ग्रेड सीमेंट", "53 grade cement for RCC construction"),
    ("hi", "घरेलू एलपीजी गैस स्टोव खाना पकाने के लिए", "domestic LPG gas stove for cooking"),
    ("ta", "இரு சக்கர வாகன ஓட்டிகளுக்கான பாதுகாப்பு தலைக்கவசம்", "protective helmet for two-wheeler riders"),
    ("te", "త్రాగునీటి కోసం ప్యాకేజ్డ్ మంచినీటి స్పెసిఫికేషన్", "packaged drinking water specification"),
]

if __name__ == "__main__":
    print("Testing multilingual query normalization...\n")
    for expected_lang, query, expected_meaning in SAMPLE_QUERIES:
        result = normalize_query(query)
        print(f"Input ({expected_lang}): {query}")
        print(f"  Detected lang : {result['detected_lang']}")
        print(f"  Translated to : {result['english']}")
        print(f"  Expected gist : {expected_meaning}")
        if "translation_error" in result:
            print(f"  ⚠️ Translation error: {result['translation_error']}")
        print()
        time.sleep(2)  # stay under the free endpoint's rate limit between test queries
