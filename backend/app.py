import os
import re
import json
import base64
import io
import requests
import feedparser
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from gtts import gTTS
import anthropic


load_dotenv()

app = Flask(__name__)
CORS(app, origins="*")

#API KEYS
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY")
NEWS_API_KEY        = os.getenv("NEWS_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")


if not ANTHROPIC_API_KEY:
    print("⚠️  WARNING: ANTHROPIC_API_KEY is not set. Most features will not work.")
    print("   Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-...")

# ── Hugging Face sentiment model ──
HF_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
HF_BASE = "https://api-inference.huggingface.co/models"


GTTS_SUPPORTED = {"english": "en", "hindi": "hi"}


# Indian RSS Feeds
INDIAN_RSS_FEEDS = [
    {"name": "NDTV News",            "url": "https://feeds.feedburner.com/ndtvnews-top-stories"},
    {"name": "NDTV India",           "url": "https://feeds.feedburner.com/ndtvnews-india-news"},
    {"name": "NDTV Sports",          "url": "https://feeds.feedburner.com/ndtvnews-sports"},
    {"name": "Times of India",       "url": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"},
    {"name": "Times of India Sport", "url": "https://timesofindia.indiatimes.com/rssfeeds/4719148.cms"},
    {"name": "Hindustan Times",      "url": "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml"},
    {"name": "India Today",          "url": "https://www.indiatoday.in/rss/home"},
    {"name": "India Today Sport",    "url": "https://www.indiatoday.in/rss/1206578"},
    {"name": "The Hindu",            "url": "https://www.thehindu.com/news/feeder/default.rss"},
    {"name": "The Hindu Sport",      "url": "https://www.thehindu.com/sport/feeder/default.rss"},
    {"name": "Indian Express",       "url": "https://indianexpress.com/feed/"},
    {"name": "Indian Express Sport", "url": "https://indianexpress.com/section/sports/feed/"},
    {"name": "Zee News",             "url": "https://zeenews.india.com/rss/india-national-news.xml"},
    {"name": "Zee News Sport",       "url": "https://zeenews.india.com/rss/sports-news.xml"},
    {"name": "ABP Live",             "url": "https://news.abplive.com/news/india/feed"},
    {"name": "News18 India",         "url": "https://www.news18.com/rss/india.xml"},
    {"name": "News18 Cricket",       "url": "https://www.news18.com/rss/cricket.xml"},
    {"name": "Firstpost",            "url": "https://www.firstpost.com/rss/india.xml"},
    {"name": "Firstpost Sport",      "url": "https://www.firstpost.com/rss/sports.xml"},
    {"name": "Business Standard",    "url": "https://www.business-standard.com/rss/latest.rss"},
    {"name": "LiveMint",             "url": "https://www.livemint.com/rss/news"},
    {"name": "Economic Times",       "url": "https://economictimes.indiatimes.com/rssfeedstopstories.cms"},
]



def fetch_wikipedia(name: str) -> dict:
    headers = {"User-Agent": "NexAI/1.0 (nexai@example.com)"}

    encoded_name = name.strip().title().replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_name}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200 and "application/json" in resp.headers.get("Content-Type", ""):
            data = resp.json()
            if data.get("type") != "disambiguation":
                return {
                    "title":     data.get("title", name),
                    "summary":   data.get("extract", ""),
                    "thumbnail": data.get("thumbnail", {}).get("source", ""),
                    "url":       data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                }
    except Exception as e:
        print(f"Wikipedia direct lookup error: {e}")

    print(f"  Direct lookup failed, trying Wikipedia search for '{name}'...")
    try:
        search_resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": name,
                "format": "json",
                "srlimit": 1,
            },
            headers=headers,
            timeout=10
        )
        search_data = search_resp.json()
        results = search_data.get("query", {}).get("search", [])
        if not results:
            return {"error": f"Wikipedia could not find '{name}'. Check spelling or try a more specific name."}

        top_title = results[0]["title"].replace(" ", "_")
        fallback_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{top_title}"
        fallback_resp = requests.get(fallback_url, headers=headers, timeout=10)

        if fallback_resp.status_code == 200:
            data = fallback_resp.json()
            return {
                "title":     data.get("title", name),
                "summary":   data.get("extract", ""),
                "thumbnail": data.get("thumbnail", {}).get("source", ""),
                "url":       data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            }
    except Exception as e:
        print(f"Wikipedia search fallback error: {e}")

    return {"error": f"Wikipedia could not find '{name}'. Check spelling or try a more specific name."}
def _parse_date(entry) -> str:
    for attr in ("published", "updated", "created"):
        t = entry.get(f"{attr}_parsed")
        if t:
            try:
                return datetime(*t[:3]).strftime("%Y-%m-%d")
            except Exception:
                pass
        raw = entry.get(attr, "")
        if raw and len(raw) >= 10:
            return raw[:10]
    return ""


def _score_relevance(entry: dict, name: str) -> int:
    name_lower = name.lower()
    title      = entry.get("title", "").lower()
    summary    = entry.get("summary", "").lower()
    words      = [w for w in name_lower.split() if len(w) >= 4]
    score      = title.count(name_lower) * 3 + summary.count(name_lower)
    for w in words:
        score += title.count(w) * 2 + summary.count(w)
    return score


def _fetch_single_feed(feed_info: dict, name: str) -> list:
    """Fetch one RSS feed and return matching articles."""
    name_lower = name.lower()
    name_words = [w for w in name_lower.split() if len(w) >= 4]
    results    = []
    try:
        feed = feedparser.parse(feed_info["url"])
        for entry in feed.entries:
            title   = entry.get("title", "")
            summary = entry.get("summary", "")
            link    = entry.get("link", "")
            text    = (title + " " + summary).lower()
            matched = name_lower in text or any(w in text for w in name_words)
            if not matched or not link:
                continue
            results.append({
                "title":       title,
                "source":      feed_info["name"],
                "url":         link,
                "publishedAt": _parse_date(entry),
                "region":      "India",
                "_score":      _score_relevance(entry, name),
            })
    except Exception as e:
        print(f"  RSS error ({feed_info['name']}): {e}")
    return results


def fetch_news_from_rss(name: str) -> list:
    """Search Indian RSS feeds concurrently with per-feed timeout."""
    results   = []
    seen_urls = set()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_fetch_single_feed, feed_info, name): feed_info
            for feed_info in INDIAN_RSS_FEEDS
        }
        for future in as_completed(futures, timeout=12):
            try:
                for article in future.result():
                    if article["url"] not in seen_urls:
                        seen_urls.add(article["url"])
                        results.append(article)
            except TimeoutError:
                print(f"  Feed timed out: {futures[future]['name']}")
            except Exception as e:
                print(f"  Feed error: {e}")

    results.sort(key=lambda x: (x["_score"], x["publishedAt"]), reverse=True)
    for r in results:
        r.pop("_score", None)

    print(f"  RSS articles matched: {len(results)}")
    return results[:10]


def fetch_news_from_newsapi(name: str) -> list:
    """Fallback: NewsAPI (requires NEWS_API_KEY)."""
    if not NEWS_API_KEY:
        return []
    results = []
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": name, "sortBy": "publishedAt", "pageSize": 8, "apiKey": NEWS_API_KEY},
            timeout=10
        )
        for a in resp.json().get("articles", []):
            results.append({
                "title":       a.get("title", ""),
                "source":      a.get("source", {}).get("name", ""),
                "url":         a.get("url", ""),
                "publishedAt": a.get("publishedAt", "")[:10],
                "region":      "Global",
            })
    except Exception as e:
        print(f"  NewsAPI error: {e}")
    return results


def fetch_news(name: str) -> list:
    print("  Fetching Indian RSS feeds (free, no key)...")
    rss_results = fetch_news_from_rss(name)
    combined    = list(rss_results)
    seen_urls   = {r["url"] for r in combined}

    if len(combined) < 3:
        print("  RSS <3 results — trying NewsAPI fallback...")
        for a in fetch_news_from_newsapi(name):
            if a["url"] not in seen_urls:
                seen_urls.add(a["url"])
                combined.append(a)

    if not combined:
        return [{"error": "No recent news found for this person"}]

    print(f"  Total news articles: {len(combined)}")
    return combined[:8]


def _parse_claude_json(raw: str) -> dict:
    """Robustly extract JSON from Claude's response, handling markdown fences."""
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*",     "", raw)
    raw = re.sub(r"\s*```$",     "", raw)
    raw = raw.strip()
    return json.loads(raw)


def fetch_ai_content(name: str, wiki_summary: str) -> dict:
    if not ANTHROPIC_API_KEY:
        return {}
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": f"""You are a professional journalist AI.

Return ONLY valid JSON with these keys:
- one_line_bio (string)
- interview_questions (list of 7)
- podcast_script (2-3 sentences)
- controversy_section (string)
- fun_facts (list of 5)
- career_highlights (list of 6)

RULES:
- controversy_section MUST describe a REAL controversy or criticism
- Always include at least one known controversy if the person is widely known
- Be factual and neutral (journalistic tone)
- Never say 'No data' or similar phrases
- Do NOT include news headlines
- Return ONLY the JSON object, no markdown fences

Context:
{wiki_summary[:500]}
"""
            }]
        )
        raw = message.content[0].text.strip()
        return _parse_claude_json(raw)
    except Exception as e:
        print("Claude ERROR:", e)
        return {
            "one_line_bio": f"{name} is a public figure.",
            "interview_questions": ["What inspired your journey?"],
            "podcast_script": f"Today we talk about {name}.",
            "controversy_section": "No data available.",
            "fun_facts": ["No data"],
            "career_highlights": ["No data"]
        }

def fetch_sentiment(text: str) -> dict | None:
    if not HUGGINGFACE_API_KEY:
        return None
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    try:
        resp   = requests.post(
            f"{HF_BASE}/{HF_SENTIMENT_MODEL}",
            headers=headers,
            json={"inputs": text[:512]},
            timeout=20
        )
        result = resp.json()
        if isinstance(result, list) and result:
            scores     = result[0] if isinstance(result[0], list) else result
            best       = max(scores, key=lambda x: x.get("score", 0))
            label      = best.get("label", "neutral").lower()
            confidence = round(best.get("score", 0) * 100)
            return {
                "score_tag":  {"positive": "P", "negative": "N", "neutral": "NEU"}.get(label, "NEU"),
                "label":      {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}.get(label, "Neutral"),
                "confidence": confidence,
            }
    except Exception as e:
        print(f"HF Sentiment error: {e}")
    return None

def translate_with_claude(text: str, language: str) -> str:
    if not ANTHROPIC_API_KEY or not text.strip():
        return text
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            messages=[{"role": "user", "content": (
                f"Translate into {language} using native script "
                f"(Hindi → Devanagari, Punjabi → Gurmukhi). "
                f"Keep meaning natural and fluent:\n\n{text}"
            )}]
        )
        return message.content[0].text.strip()
    except Exception as e:
        print("Translation Error:", e)
        return text



def generate_audio(text: str, language: str = "english") -> str:
    lang_code = GTTS_SUPPORTED.get(language)
    if not lang_code:
        print(f"  gTTS: language '{language}' not supported, skipping audio.")
        return ""
    try:
        tts = gTTS(text=text[:1000], lang=lang_code, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print("gTTS ERROR:", e)
        return ""

def generate_fun_facts_fallback(name, wiki_summary):
    facts = [s.strip() for s in wiki_summary.split(". ") if len(s.strip()) > 20][:5]
    if len(facts) < 3:
        facts += [
            f"{name} is widely recognized in their field.",
            f"{name} has influenced many people through their work.",
        ]
    return facts[:5]


def generate_career_highlights_fallback(name, wiki_summary):
    keywords = ["won", "award", "record", "famous", "known", "first", "led", "achieved"]
    highlights = [
        s.strip() for s in wiki_summary.split(". ")
        if any(w in s.lower() for w in keywords) and len(s.strip()) > 20
    ]
    if len(highlights) < 3:
        highlights += [
            f"{name} has made significant contributions in their field.",
            f"{name} is widely recognized for their achievements.",
            f"{name} continues to influence many people.",
        ]
    return highlights[:6]


def generate_interview_questions_fallback(name):
    return [
        f"What inspired you to start your journey, {name}?",
        "What challenges did you face early in your career?",
        "What was a turning point in your success story?",
        "How do you handle criticism and pressure?",
        "What motivates you to keep going?",
        "What advice would you give to beginners?",
        "What are your future goals?",
    ]


def extract_and_summarize_controversy(name, news):
    keywords = [
        "controversy", "backlash", "criticized", "accused", "scandal", "debate",
        "lawsuit", "investigation", "ban", "fired", "sued", "court", "regulator",
        "complaint", "violation",
    ]
    filtered = [
        a for a in news
        if any(k in a.get("title", "").lower() for k in keywords)
    ][:3]

    if filtered and ANTHROPIC_API_KEY:
        combined = "\n".join(f"- {a['title']}" for a in filtered)
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                messages=[{"role": "user", "content":
                    f"Summarize in 2-3 neutral sentences based ONLY on:\n{combined}\nDo NOT add new facts."}]
            )
            return msg.content[0].text.strip()
        except Exception as e:
            print("Controversy AI error:", e)

    try:
        resp  = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "format": "json", "prop": "extracts",
                "titles": name, "explaintext": True,
            },
            timeout=10
        )
        pages = resp.json().get("query", {}).get("pages", {})
        page  = next(iter(pages.values()))
        ck    = ["controversy", "criticism", "allegation", "scandal", "accused", "backlash"]
        rel   = [
            s.strip() for s in page.get("extract", "").split(". ")
            if any(k in s.lower() for k in ck) and len(s) > 40
        ]
        if rel:
            return "Historically, " + " ".join(rel[:2])
    except Exception:
        pass

    return f"{name} has been involved in public debates and discussions in media and politics."


@app.route("/api/generate", methods=["POST"])
def generate():
    body     = request.get_json()
    name     = body.get("name", "").strip()
    language = body.get("language", "english").lower()

    if not name:
        return jsonify({"error": "Name is required"}), 400

    print(f"\n🔍 Researching: {name} | Language: {language}")

    print("  [1/6] Wikipedia...")
    wikipedia = fetch_wikipedia(name)
    if "error" in wikipedia:
        return jsonify({"error": wikipedia["error"]}), 404
    wiki_summary = wikipedia.get("summary", "")

    print("  [2/6] News (Indian RSS → NewsAPI fallback)...")
    news = fetch_news(name)

    print("  [3/6] Claude AI content...")
    ai_content = fetch_ai_content(name, wiki_summary) or {}

    if not ai_content.get("interview_questions") or len(ai_content["interview_questions"]) < 5:
        ai_content["interview_questions"] = generate_interview_questions_fallback(name)
    if not ai_content.get("career_highlights") or ai_content["career_highlights"] == ["No data"]:
        ai_content["career_highlights"] = generate_career_highlights_fallback(name, wiki_summary)
    if not ai_content.get("fun_facts") or ai_content["fun_facts"] == ["No data"]:
        ai_content["fun_facts"] = generate_fun_facts_fallback(name, wiki_summary)

    existing = ai_content.get("controversy_section", "").lower()
    if not existing or any(x in existing for x in ["no ", "not found", "no data"]):
        valid_news = [a for a in news if not a.get("error")]
        ai_content["controversy_section"] = (
            extract_and_summarize_controversy(name, valid_news) if valid_news
            else f"{name} has faced public criticism and controversies over time."
        )

    if not ai_content.get("podcast_script") or ai_content["podcast_script"] == "No data.":
        ai_content["podcast_script"] = (
            f"Today on NexAI, we explore the life and career of {name}. "
            f"{wiki_summary[:300]}. Stay tuned for more insights."
        )

    print("  [4/6] HuggingFace Sentiment...")
    sentiment = fetch_sentiment(wiki_summary)

    questions = ai_content.get("interview_questions", [])
    podcast   = ai_content.get("podcast_script", "")
    translated_questions = []
    translated_script    = ""

    print(f"  [5/6] Translation → {language}...")
    if language == "english":
        translated_questions = questions
        translated_script    = podcast
    elif language in ["hindi", "punjabi"]:
        translated_all       = translate_with_claude("\n".join(questions), language)
        translated_questions = [q.strip() for q in translated_all.split("\n") if q.strip()]
        translated_script    = translate_with_claude(podcast, language)
        ai_content["one_line_bio"]        = translate_with_claude(ai_content.get("one_line_bio", ""), language)
        ai_content["controversy_section"] = translate_with_claude(ai_content.get("controversy_section", ""), language)
        ai_content["fun_facts"]           = [translate_with_claude(f, language) for f in ai_content.get("fun_facts", [])]
        ai_content["career_highlights"]   = [translate_with_claude(h, language) for h in ai_content.get("career_highlights", [])]
        translated_summary = translate_with_claude(wiki_summary, language)
        if translated_summary:
            wikipedia["summary"] = translated_summary
    else:
        translated_questions = questions
        translated_script    = podcast

    print("  [6/6] gTTS Audio Generation...")
    audio_base64 = generate_audio(translated_script or podcast, language)

    return jsonify({
        "name":       name,
        "wikipedia":  wikipedia,
        "news":       news,
        "ai_content": ai_content,
        "sentiment":  sentiment,
        "translation": {
            "language":                       language,
            "interview_questions_translated": translated_questions,
            "podcast_script_translated":      translated_script,
            "audio_base64":                   audio_base64,
        }
    })

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "apis_configured": {
            "anthropic":   bool(ANTHROPIC_API_KEY),
            "newsapi":     bool(NEWS_API_KEY),
            "huggingface": bool(HUGGINGFACE_API_KEY),
            "gtts":        True,
            "indian_rss":  True,
        }
    })


if __name__ == "__main__":
    print("🚀 NexAI Backend starting on http://localhost:5000")
    print("📋 Health check: http://localhost:5000/api/health")
    app.run(debug=True, port=5000)