import os
import io
import json
import time
import re
import logging
import html
import hashlib
import math
import urllib.parse
from datetime import date, datetime
from pathlib import Path
from collections import defaultdict

import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle

load_dotenv()

# First try local .env
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.6-flash")

# If deployed on Streamlit Cloud, use Streamlit Secrets
if not API_KEY:
    try:
        API_KEY = st.secrets["GOOGLE_API_KEY"]
        MODEL = st.secrets.get("GOOGLE_MODEL", "gemini-3.6-flash")
    except Exception:
        pass

st.set_page_config(page_title="Voyage | AI Travel Assistant", page_icon="✈️", layout="wide")

if not API_KEY:
    st.info("Gemini is not configured yet. You can preview the interface and use non-AI tools; add GOOGLE_API_KEY in Streamlit Secrets when ready.")
# CSS uses Streamlit theme variables, so both light and dark mode stay readable.
st.markdown("""
<style>
:root { --voyage-accent: #3987e8; }
.block-container { padding-top: 1.3rem; padding-bottom: 3.5rem; max-width: 1200px; }
.hero { padding: 1.6rem 1.8rem; border-radius: 22px;
  background: linear-gradient(120deg, color-mix(in srgb, var(--primary-color, #3987e8) 13%, var(--secondary-background-color, #f5f7fb)), var(--secondary-background-color, #f5f7fb));
  border: 1px solid color-mix(in srgb, var(--text-color, #222) 12%, transparent);
  color: var(--text-color, #222); margin-bottom: 1rem; }
.hero h1 { margin: 0 0 .25rem 0; color: inherit; }
.hero p { margin: 0; opacity: .85; color: inherit; }
.voyage-card { background: var(--secondary-background-color, #f5f7fb);
  border: 1px solid color-mix(in srgb, var(--text-color, #222) 12%, transparent);
  border-radius: 16px; padding: 1rem 1.2rem; margin: .55rem 0; color: var(--text-color, #222); }
.voyage-card h3 { margin-top: 0; color: inherit; }
.pill { display: inline-block; padding: .25rem .65rem; margin: .12rem;
  border-radius: 999px; border: 1px solid color-mix(in srgb, var(--text-color, #222) 22%, transparent);
  color: var(--text-color, #222); font-size: .86rem; }
[data-testid="stSidebar"] { border-right: 1px solid color-mix(in srgb, var(--text-color, #222) 9%, transparent); }
@media(max-width: 640px) { .block-container { padding: 1rem .8rem 2rem; }
  .hero { padding: 1.15rem; border-radius: 15px; } .hero h1 { font-size: 1.75rem; } }
</style>
""", unsafe_allow_html=True)

LOGGER = logging.getLogger("voyage")

SYSTEM_PROMPT = """
You are Voyage, an AI travel-planning assistant.

Preserve all supplied trip details and conversation context. Never ask again for details
already present. If the user changes one preference, revise the existing trip rather than
starting over. Use past-trip preferences when supplied.

When enough information is known, provide:
1. Trip title and short trip mood.
2. Day-by-day itinerary with morning / afternoon / evening.
3. Suggested areas to stay and 2-4 example hotels/hostels/homestays per base. State clearly
   that live availability/prices must be checked.
4. Practical transport advice: airport transfer, metro/tram/bus cards, cab/ride-hailing apps,
   walking practicality, and intercity travel.
5. Food/local experience suggestions.
6. Estimated budget allocation within the user's stated budget. If unrealistic, say so and
   suggest the least disruptive changes.
7. Packing notes.
8. Important local things to know: customs, etiquette, permits, transit rules, photography
   restrictions, closures, or unusual practical rules. Flag time-sensitive facts for verification.
9. Responsible travel suggestions and genuine SDG links.
10. Places whose photos would be useful.

Never claim live hotel availability, exact live airfare, exact current prices, current closures,
current laws, or future weather unless supplied by a live data source. Distinguish estimates
from verified/current data.

Do not recommend alcohol, casinos/gambling, age-restricted nightlife, or unsafe activities.
Offer age-appropriate alternatives such as night markets, cultural shows, cafés, viewpoints,
or evening walks in busy public areas.
BUDGET FEASIBILITY:

Before generating any complete itinerary, perform a realistic feasibility check.

The user's stated total budget must be treated as a hard constraint unless the
user explicitly says otherwise.

Use Google Search grounding when current travel costs materially affect feasibility.

Consider:
- international or domestic transport from the user's starting city
- accommodation
- local transport
- food
- activities / entry fees
- taxes or common mandatory charges where relevant
- a reasonable emergency / contingency buffer

Clearly state whether the trip is:

✅ REALISTIC WITHIN BUDGET
⚠️ POSSIBLE ONLY WITH MAJOR COMPROMISES
❌ NOT REALISTIC WITHIN BUDGET

Do not force an itinerary to fit an unrealistic budget.

If the trip is not realistic:
1. Explain which categories make the budget unrealistic.
2. Give a realistic approximate budget range based on grounded/current evidence.
3. Suggest the smallest useful changes, such as:
   - changing destination
   - shortening the trip
   - changing travel dates
   - using cheaper accommodation
   - increasing the budget
4. Ask the user which alternative they prefer before finalizing the itinerary.

Never claim that an international trip fits the user's budget merely because
individual daily expenses look affordable. Include the cost of reaching the destination
when an origin city has been supplied.

Use clear Markdown headings and concise detail.
"""

def new_client():
    return genai.Client(api_key=API_KEY)

def build_context():
    profile = st.session_state.get("trip_profile", {})
    history = st.session_state.get("messages", [])
    past = st.session_state.get("past_trips", [])
    text = SYSTEM_PROMPT + "\n\n"
    text += "CURRENT TRIP PROFILE:\n" + json.dumps(profile, ensure_ascii=False, indent=2) + "\n\n"
    if past:
        text += "PAST TRIP MEMORY:\n" + json.dumps(past[-5:], ensure_ascii=False, indent=2) + "\n\n"
    text += "CONVERSATION:\n"
    for m in history:
        text += f'{m["role"].upper()}: {m["content"]}\n\n'
    return text
def add_grounding_citations(response):
    """
    Adds clickable citations from Google Search grounding
    to the Gemini response text.
    """
    text = response.text

    try:
        metadata = response.candidates[0].grounding_metadata

        if not metadata:
            return text

        supports = metadata.grounding_supports or []
        chunks = metadata.grounding_chunks or []

        # Insert citations from the end backwards so indexes don't shift
        supports = sorted(
            supports,
            key=lambda s: s.segment.end_index if s.segment else 0,
            reverse=True
        )

        for support in supports:

            if not support.segment:
                continue

            end_index = support.segment.end_index

            links = []

            if support.grounding_chunk_indices:

                for i in support.grounding_chunk_indices:

                    if i < len(chunks):

                        chunk = chunks[i]

                        if chunk.web and chunk.web.uri:

                            title = chunk.web.title or f"Source {i + 1}"

                            links.append(
                                f"[{i + 1}]({chunk.web.uri})"
                            )

            if links:

                citation = " " + " ".join(links)

                text = (
                    text[:end_index]
                    + citation
                    + text[end_index:]
                )

    except Exception:
        # If citation parsing ever fails, still show the AI answer
        pass

    return text



def call_gemini(extra_instruction=None, retries=2):
    """
    Sends the current Voyage conversation to Gemini.

    Features:
    - Temporarily disables Google Search grounding
    - Retries temporary 503 errors
    - Handles quota / 429 errors
    - Logs safe diagnostic information
    - Returns generated itinerary text
    """

    if not API_KEY:
        raise RuntimeError("Gemini API key is not configured")

    prompt = build_context()

    if extra_instruction:
        prompt += "\n\nTASK FOR THIS TURN:\n" + extra_instruction

    for attempt in range(retries):
        try:
            # Create a fresh Gemini client
            client = genai.Client(api_key=API_KEY)

            # Google Search grounding temporarily disabled
            config = types.GenerateContentConfig()

            # Generate the response
            response = client.models.generate_content(
                model=MODEL,
                contents="Reply with exactly: Voyage connected",
                config=config
            )

            # Extract response text
            result = response.text

            if not result or not result.strip():
                raise RuntimeError("Gemini returned an empty response")

            return result.strip()

        except Exception as e:
            msg = str(e).lower()
            error_type = type(e).__name__

            # Log the error category without exposing
            # API keys or sensitive request details
            status = next(
                (
                    code
                    for code in ("429", "503", "401", "403", "404")
                    if code in msg
                ),
                "unknown"
            )

            LOGGER.warning(
                "Gemini request failed: type=%s status=%s attempt=%s",
                error_type,
                status,
                attempt + 1
            )

            # Handle quota / rate-limit errors
            if (
                "429" in msg
                or "resource_exhausted" in msg
                or "quota" in msg
            ):
                raise RuntimeError(
                    "Voyage AI quota/rate limit reached (429). "
                    "Check project quota and model access."
                ) from None

            # Retry temporary server errors
            if (
                "503" in msg
                or "unavailable" in msg
                or "high demand" in msg
            ):
                if attempt < retries - 1:
                    time.sleep(2 + attempt * 2)
                    continue

                raise RuntimeError(
                    "Gemini is temporarily unavailable (503)."
                ) from None

            # Retry closed-client errors
            if "client has been closed" in msg:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue

            # Log safe details for local RuntimeErrors
            if isinstance(e, RuntimeError):
                LOGGER.error(
                    "Gemini diagnostic: %s",
                    str(e)
                )

            # Handle other errors
            raise RuntimeError(
                f"Gemini request failed ({error_type}). "
                "Check private app logs."
            ) from None

    raise RuntimeError(
        "Gemini is temporarily unavailable after retries."
    ) from None

def geocode_city(place):
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": place, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    r.raise_for_status()
    results = r.json().get("results", [])
    return results[0] if results else None

@st.cache_data(ttl=1800)
def live_forecast(place):
    loc = geocode_city(place)
    if not loc:
        return None
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 14,
        },
        timeout=10,
    )
    r.raise_for_status()
    return {"location": loc, "daily": r.json().get("daily", {})}

def weather_code_text(code):
    mapping = {0:"Clear",1:"Mostly clear",2:"Partly cloudy",3:"Overcast",45:"Fog",48:"Rime fog",
               51:"Light drizzle",53:"Drizzle",55:"Heavy drizzle",61:"Light rain",63:"Rain",
               65:"Heavy rain",71:"Light snow",73:"Snow",75:"Heavy snow",80:"Rain showers",
               81:"Rain showers",82:"Heavy showers",95:"Thunderstorms"}
    return mapping.get(code, "Mixed conditions")


@st.cache_data(ttl=86400)
def best_commons_image(place):
    """
    Finds one useful photograph for one specific attraction/place
    from Wikimedia Commons.

    Filters out PDFs, maps, scans, SVGs and document-type results.
    """

    if not place:
        return None

    try:
        search_url = "https://commons.wikimedia.org/w/api.php"

        # Search for the EXACT attraction rather than a whole country
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": f'"{place}"',
            "srnamespace": 6,
            "srlimit": 20,
            "format": "json",
            "origin": "*"
        }

        r = requests.get(
            search_url,
            params=search_params,
            timeout=12,
            headers={
                "User-Agent": "VoyageSchoolAIProject/1.0"
            }
        )

        r.raise_for_status()

        results = r.json().get(
            "query", {}
        ).get(
            "search", []
        )

        # We only want normal image formats
        allowed_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp"
        )

        # Reject obviously useless results
        banned_words = [
            "map",
            "diagram",
            "logo",
            "flag",
            "coat of arms",
            "document",
            "scan",
            "book",
            "pdf",
            "chart",
            "graph"
        ]

        candidates = []

        for result in results:

            title = result.get("title", "")

            title_lower = title.lower()

            # Must be an actual photograph/image
            if not title_lower.endswith(
                allowed_extensions
            ):
                continue

            # Skip maps/documents/etc.
            if any(
                word in title_lower
                for word in banned_words
            ):
                continue

            candidates.append(title)

        if not candidates:
            return None

        # Try the best few candidates
        for title in candidates[:8]:

            info_params = {
                "action": "query",
                "titles": title,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata",
                "iiurlwidth": 1200,
                "format": "json",
                "origin": "*"
            }

            info_response = requests.get(
                search_url,
                params=info_params,
                timeout=12,
                headers={
                    "User-Agent":
                    "VoyageSchoolAIProject/1.0"
                }
            )

            info_response.raise_for_status()

            pages = info_response.json().get(
                "query", {}
            ).get(
                "pages", {}
            )

            for page in pages.values():

                image_info = (
                    page.get("imageinfo")
                    or [{}]
                )[0]

                image_url = (
                    image_info.get("thumburl")
                    or image_info.get("url")
                )

                if image_url:

                    return {
                        "place": place,
                        "url": image_url,
                        "title": title.replace(
                            "File:", ""
                        ),
                        "source":
                            image_info.get(
                                "descriptionurl",
                                ""
                            )
                    }

    except Exception:
        pass

    return None
def extract_photo_places():
    """No additional Gemini request: find named places in the existing plan."""
    plan = st.session_state.get("latest_plan", "")
    matches = re.findall(r"(?:^|\n)\s*(?:[-*]\s*)?(?:Visit|Explore|See|Stop at|Discover)\s+([A-Z][^\n,.;:]{3,55})", plan, re.I)
    matches += re.findall(r"\*\*([^*\n]{5,55})\*\*", plan)
    banned = ("morning", "afternoon", "evening", "budget", "day ", "total", "transport", "hotel", "meal")
    return list(dict.fromkeys(x.strip(" -:0123456789") for x in matches
        if not any(w in x.lower() for w in banned)))[:5]


def extract_itinerary_cities():
    """No additional Gemini request; prefer known destination and manually entered bases."""
    profile = st.session_state.get("trip_profile", {})
    destination = profile.get("destination", "")
    if destination and destination != "not decided":
        return [destination]
    return []

def qp(s):
    return urllib.parse.quote_plus(str(s or ""))

def google_flights_link(origin, destination, depart, ret):
    query = f"flights from {origin} to {destination}"
    if depart: query += f" on {depart}"
    if ret: query += f" returning {ret}"
    return "https://www.google.com/travel/flights?q=" + qp(query)

def skyscanner_link(origin, destination, depart, ret):
    return "https://www.skyscanner.com/transport/flights/?query=" + qp(
        f"{origin} to {destination} flights {depart or ''} {ret or ''}"
    )

def booking_link(destination, checkin, checkout):
    params = {"ss": destination}
    if checkin: params["checkin"] = str(checkin)
    if checkout: params["checkout"] = str(checkout)
    return "https://www.booking.com/searchresults.html?" + urllib.parse.urlencode(params)

def maps_search_link(query):
    return "https://www.google.com/maps/search/?api=1&query=" + qp(query)

def load_past_trips():
    """Session-only memories: never share a server-side JSON file across users."""
    return []


def save_past_trips(trips):
    """No server persistence. Users may download/import their own JSON."""
    return None

def markdown_to_pdf_bytes(title, markdown_text):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenterTitle", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=12))
    story = [Paragraph(title, styles["CenterTitle"]), Spacer(1,8)]
    for raw in markdown_text.splitlines():
        line = raw.strip()
        if not line:
            story.append(Spacer(1,6)); continue
        safe = line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        safe = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", safe)
        safe = re.sub(r"\*(.*?)\*", r"<i>\1</i>", safe)
        if line.startswith("### "):
            story.append(Paragraph(safe[4:], styles["Heading3"]))
        elif line.startswith("## "):
            story.append(Paragraph(safe[3:], styles["Heading2"]))
        elif line.startswith("# "):
            story.append(Paragraph(safe[2:], styles["Heading1"]))
        elif line.startswith("- "):
            story.append(Paragraph("• " + safe[2:], styles["BodyText"]))
        else:
            story.append(Paragraph(safe.replace("|"," &nbsp; "), styles["BodyText"]))
        story.append(Spacer(1,3))
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

def parse_budget(profile):
    raw = str(profile.get("budget", ""))
    currency = raw.split(" ")[0] if " " in raw else "INR"
    try:
        amount = float(re.sub(r"[^0-9.]", "", raw))
    except ValueError:
        amount = 0
    return currency, amount


def budget_allocations(profile):
    """Illustrative allocation only; NOT current quotes or Gemini-derived prices."""
    currency, amount = parse_budget(profile)
    weights = {"Long-distance travel": .30, "Stay": .28, "Food": .16,
               "Local transport": .10, "Activities": .09, "Contingency": .07}
    return currency, {k: round(amount * v) for k, v in weights.items()}


def packing_suggestions(profile):
    season = str(profile.get("season", "") or "").lower()
    interests = " ".join(profile.get("interests", [])).lower()
    items = ["Travel documents / ID", "Tickets and booking confirmations",
             "Phone and charger", "Personal medication and essentials",
             "Comfortable shoes", "Reusable water bottle", "Weather-appropriate clothes"]
    if "winter" in season or "mountain" in interests:
        items += ["Warm layers", "Warm socks"]
    if "monsoon" in season:
        items += ["Rain jacket or umbrella"]
    if "beach" in interests or "summer" in season:
        items += ["Sun protection", "Hat"]
    if "adventure" in interests:
        items += ["Suitable activity footwear"]
    return list(dict.fromkeys(items))


def show_itinerary_cards(text):
    """Render day headings as readable collapsible sections; raw Markdown fallback."""
    parts = re.split(r"(?im)^(?=\s{0,3}#{0,3}\s*\*{0,2}day\s+\d+\b)", text)
    days = [p for p in parts if re.match(r"(?im)^\s{0,3}#{0,3}\s*\*{0,2}day\s+\d+\b", p)]
    if len(days) < 2:
        st.markdown(text)
        return
    intro = parts[0]
    if intro.strip(): st.markdown(intro)
    for part in days:
        heading = part.strip().splitlines()[0].strip(" #*")
        with st.expander(heading, expanded=False):
            st.markdown(part)


def safe_error_display(e):
    msg = str(e)
    if "quota" in msg.lower() or "429" in msg:
        st.warning("Voyage AI is currently rate-limited. Your trip setup is preserved. Check the API project's quota.")
    else:
        st.error("Voyage AI is unavailable right now. Check the app's private logs for the error category.")

if "messages" not in st.session_state: st.session_state.messages = []
if "photo_places" not in st.session_state:
    st.session_state.photo_places = []
if "weather_cities" not in st.session_state:
    st.session_state.weather_cities = []
if "trip_profile" not in st.session_state: st.session_state.trip_profile = {}
if "past_trips" not in st.session_state: st.session_state.past_trips = load_past_trips()
if "latest_plan" not in st.session_state: st.session_state.latest_plan = ""

st.markdown("""
<div class="hero"><h1>✈️ Voyage</h1>
<p>Your AI-powered travel companion — plan, explore, refine, and export your journey.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🧳 Trip setup")
    origin = st.text_input("Starting city", placeholder="e.g. Delhi")
    destination = st.text_input("Destination", placeholder="e.g. Rome, Italy (optional)")
    exact_dates = st.checkbox("I know my exact dates", value=False)

    start_date = end_date = None
    if exact_dates:
        c1, c2 = st.columns(2)
        start_date = c1.date_input("Start", value=date.today())
        end_date = c2.date_input("End", value=date.today())
        if end_date < start_date:
            st.warning("End date should be on or after start date.")
    else:
        season = st.selectbox("Approximate season",
                              ["Summer","Winter","Spring","Autumn / Fall","Monsoon","Flexible"])

    days = st.number_input("Number of days", min_value=1, max_value=45, value=5)
    travellers = st.number_input("Travellers", min_value=1, max_value=20, value=2)
    budget = st.number_input("Total budget", min_value=0, value=50000, step=5000)
    currency = st.selectbox("Currency", ["INR","USD","EUR","GBP","JPY","AUD","CAD"])
    interests = st.multiselect("Preferences",
        ["Adventure","Nature","Beaches","Mountains","Culture & History","Food","Shopping",
         "Architecture / Skyscrapers","Photography","Relaxation","Family activities",
         "Hidden gems","Night markets","Museums","Theme parks"])
    pace = st.select_slider("Pace", ["Relaxed","Balanced","Packed"], value="Balanced")
    accommodation = st.selectbox("Accommodation preference",
                                 ["Budget","Mid-range","Luxury","Hostel / Homestay","Flexible"])
    transport = st.multiselect("Transport preference",
                               ["Walking","Metro / Tram","Public bus","Train",
                                "Cab / Ride-hailing","Rental car","Domestic flight"])
    past_note = st.text_area("Past-trip preferences (optional)",
                             placeholder="Loved quiet mountain towns; disliked rushed museum days.")
    start_btn = st.button("✨ Build my trip", use_container_width=True)
    reset_btn = st.button("🗑️ New trip", use_container_width=True)

if reset_btn:
    st.session_state.messages = []
    st.session_state.trip_profile = {}
    st.session_state.latest_plan = ""
    st.session_state.photo_places = []
    st.session_state.weather_cities = []
    st.rerun()

if start_btn:
    profile = {
        "origin": origin or "not provided",
        "destination": destination or "not decided",
        "exact_dates": exact_dates,
        "start_date": str(start_date) if start_date else None,
        "end_date": str(end_date) if end_date else None,
        "season": None if exact_dates else season,
        "days": int(days),
        "travellers": int(travellers),
        "budget": f"{currency} {budget:,}",
        "interests": interests or ["not specified"],
        "pace": pace,
        "accommodation": accommodation,
        "transport": transport or ["flexible"],
        "past_trip_note": past_note or "none",
    }
    st.session_state.trip_profile = profile
    with st.spinner("Voyage is designing your trip..."):
        try:
            answer = call_gemini(
      """
      First perform a grounded budget-feasibility assessment using the user's
      origin, destination, dates/season, travellers and total budget.

      Search current web information when transport, accommodation,
      entry fees, local passes or other current prices materially affect
      the feasibility assessment.

      If the trip is unrealistic within the budget, DO NOT generate a
      fake budget-compliant itinerary. Explain the realistic problem and
      offer 2-3 practical alternatives.

      If the trip is realistically feasible, create the complete itinerary.

      For factual or time-sensitive claims such as prices, travel rules,
      transport passes, closures, visas, opening information or current
      availability, prefer grounded information and include citations.
      """
      )
        except Exception as e:
            safe_error_display(e)
            answer = None
    if answer:
        st.session_state.messages = [{"role":"assistant","content":answer}]
        st.session_state.latest_plan = answer
        st.session_state.photo_places = []
        st.session_state.weather_cities = []
        st.rerun()

profile = st.session_state.trip_profile

if profile:
    st.subheader("🧭 Trip dashboard")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Days", profile.get("days","—"))
    c2.metric("Travellers", profile.get("travellers","—"))
    c3.metric("Budget", profile.get("budget","—"))
    c4.metric("Pace", profile.get("pace","—"))

    with st.expander("✈️ Flights, hotels & maps", expanded=False):
        st.caption("These open live search sites. Voyage does not claim a fare is the cheapest unless a live airfare API is connected.")
        dest = profile.get("destination","")
        origin_for_link = profile.get("origin","")
        depart = profile.get("start_date")
        ret = profile.get("end_date")
        b1,b2,b3,b4 = st.columns(4)
        b1.link_button("Google Flights", google_flights_link(origin_for_link,dest,depart,ret), use_container_width=True)
        b2.link_button("Skyscanner", skyscanner_link(origin_for_link,dest,depart,ret), use_container_width=True)
        b3.link_button("Search hotels", booking_link(dest,depart,ret), use_container_width=True)
        b4.link_button("Open map", maps_search_link(dest), use_container_width=True)

tab_plan, tab_weather, tab_photos, tab_memory, tab_tools = st.tabs(
    ["💬 Plan & Chat", "🌦️ Weather", "🖼️ Photos", "🧠 Trip Memory", "🗺️ Trip Tools"]
)

with tab_plan:
    if not st.session_state.messages:
        st.info("Set preferences in the sidebar and click **Build my trip**.")
    else:
        card_view = st.toggle("Day-wise itinerary cards", value=False, key="card_view")
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="✈️" if message["role"]=="assistant" else None):
                if card_view and message["role"] == "assistant":
                    show_itinerary_cards(message["content"])
                else:
                    st.markdown(message["content"])

    user_message = st.chat_input("Ask Voyage to refine, compare, explain, or change your trip...")
    if user_message:
        st.session_state.messages.append({"role":"user","content":user_message})
        with st.spinner("Voyage is thinking..."):
            try:
                answer = call_gemini()
            except Exception as e:
                safe_error_display(e)
                answer = None
        if answer:
            st.session_state.messages.append({"role":"assistant","content":answer})
            st.session_state.latest_plan = answer
            st.rerun()

    if st.session_state.latest_plan:
        st.divider()
        st.subheader("📄 Export")
        pdf = markdown_to_pdf_bytes("Voyage — Detailed Travel Itinerary", st.session_state.latest_plan)
        st.download_button("⬇️ Download latest itinerary as PDF", data=pdf,
                           file_name="voyage_itinerary.pdf", mime="application/pdf",
                           use_container_width=True)

with tab_weather:

    st.subheader("🌦️ Weather Along Your Journey")

    st.write(
        "Voyage shows near-term forecasts for your destinations. "
        "For travel beyond the forecast window, use seasonal guidance instead of treating today's forecast as your travel-day weather."
    )

    if not profile:

        st.info(
            "Create an itinerary first."
        )

    else:

        # ----------------------------------------
        # IDENTIFY ITINERARY CITIES
        # ----------------------------------------

        if st.button(
            "✨ Load weather for my itinerary",
            use_container_width=True
        ):

            with st.spinner(
                "Finding the cities in your journey..."
            ):

                cities = extract_itinerary_cities()

                st.session_state.weather_cities = cities

        cities = st.session_state.get(
            "weather_cities",
            []
        )
        city_override = st.text_area("Trip cities / overnight bases (one city per line)",
            value="\n".join(cities), key="weather_city_override",
            help="Enter each city on a separate line (e.g. Rome, Italy); no AI request.")
        if city_override.strip():
            cities = [x.strip() for x in city_override.splitlines() if x.strip()][:6]


        # ----------------------------------------
        # WEATHER FOR EACH CITY
        # ----------------------------------------

        if cities:

            st.caption(
                "Weather shown below is live forecast data "
                "where available. Forecasts cover the next 14 days, NOT distant travel dates."
            )

            for city in cities:

                st.markdown("---")

                st.markdown(
                    f"## 📍 {city}"
                )

                try:

                    forecast = live_forecast(
                        city
                    )

                except Exception as e:

                    forecast = None

                    st.warning(
                        f"Weather service unavailable "
                        f"for {city}."
                    )

                if forecast:

                    loc = forecast[
                        "location"
                    ]

                    daily = forecast[
                        "daily"
                    ]

                    dates = daily.get(
                        "time", []
                    )

                    codes = daily.get(
                        "weather_code", []
                    )

                    max_temps = daily.get(
                        "temperature_2m_max", []
                    )

                    min_temps = daily.get(
                        "temperature_2m_min", []
                    )

                    rain = daily.get(
                        "precipitation_probability_max",
                        []
                    )

                    if dates:

                        # --------------------------------
                        # FIRST-DAY WEATHER SUMMARY
                        # --------------------------------

                        condition = (
                            weather_code_text(
                                codes[0]
                            )
                            if codes
                            else "Unknown"
                        )

                        max_temp = (
                            max_temps[0]
                            if max_temps
                            else "—"
                        )

                        min_temp = (
                            min_temps[0]
                            if min_temps
                            else "—"
                        )

                        rain_chance = (
                            rain[0]
                            if rain
                            else "—"
                        )

                        col1, col2, col3, col4 = (
                            st.columns(4)
                        )

                        col1.metric(
                            "Conditions",
                            condition
                        )

                        col2.metric(
                            "High",
                            f"{max_temp} °C"
                        )

                        col3.metric(
                            "Low",
                            f"{min_temp} °C"
                        )

                        col4.metric(
                            "Rain chance",
                            f"{rain_chance}%"
                        )

                        # --------------------------------
                        # UPCOMING FORECAST
                        # --------------------------------

                        st.markdown(
                            "#### 📅 Upcoming forecast"
                        )

                        rows = []

                        for i, d in enumerate(
                            dates[:7]
                        ):

                            rows.append(
                                {
                                    "Date": d,

                                    "Weather":
                                        weather_code_text(
                                            codes[i]
                                        )
                                        if i < len(codes)
                                        else "—",

                                    "Low °C":
                                        min_temps[i]
                                        if i < len(
                                            min_temps
                                        )
                                        else "—",

                                    "High °C":
                                        max_temps[i]
                                        if i < len(
                                            max_temps
                                        )
                                        else "—",

                                    "Rain %":
                                        rain[i]
                                        if i < len(
                                            rain
                                        )
                                        else "—"
                                }
                            )

                        st.dataframe(
                            rows,
                            use_container_width=True,
                            hide_index=True
                        )

                        # --------------------------------
                        # AI-FREE PACKING ADVICE
                        # --------------------------------

                        tips = []

                        if (
                            isinstance(
                                rain_chance,
                                (int, float)
                            )
                            and rain_chance >= 40
                        ):

                            tips.append(
                                "☔ Carry a compact "
                                "umbrella or rain jacket."
                            )

                        if (
                            isinstance(
                                max_temp,
                                (int, float)
                            )
                            and max_temp >= 30
                        ):

                            tips.append(
                                "☀️ Expect warm conditions — "
                                "carry water, sunscreen and "
                                "light clothing."
                            )

                        if (
                            isinstance(
                                min_temp,
                                (int, float)
                            )
                            and min_temp <= 10
                        ):

                            tips.append(
                                "🧥 Pack a warm layer for "
                                "cool mornings or evenings."
                            )

                        if tips:

                            st.markdown(
                                "#### 🎒 Weather tips"
                            )

                            for tip in tips:

                                st.write(tip)

                else:

                    st.warning(
                        f"No weather data found "
                        f"for {city}."
                    )

        else:

            st.info(
                "Click **Load weather for my itinerary** "
                "to automatically detect your trip's cities."
            )

        # ----------------------------------------
        # MANUAL WEATHER LOOKUP
        # ----------------------------------------

        st.divider()

        with st.expander(
            "🔎 Check another city"
        ):

            manual_city = st.text_input(
                "City name",
                placeholder="e.g. Milan, Italy",
                key="manual_weather_city"
            )

            if manual_city:

                try:

                    manual_forecast = (
                        live_forecast(
                            manual_city
                        )
                    )

                except Exception:

                    manual_forecast = None

                if manual_forecast:

                    daily = manual_forecast[
                        "daily"
                    ]

                    rows = []

                    for i, d in enumerate(
                        daily.get(
                            "time", []
                        )[:7]
                    ):

                        rows.append(
                            {
                                "Date": d,

                                "Weather":
                                    weather_code_text(
                                        daily.get(
                                            "weather_code",
                                            []
                                        )[i]
                                    ),

                                "Low °C":
                                    daily.get(
                                        "temperature_2m_min",
                                        []
                                    )[i],

                                "High °C":
                                    daily.get(
                                        "temperature_2m_max",
                                        []
                                    )[i],

                                "Rain %":
                                    daily.get(
                                        "precipitation_probability_max",
                                        []
                                    )[i]
                            }
                        )

                    st.dataframe(
                        rows,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.warning(
                        "Weather information "
                        "couldn't be found."
                    )

with tab_photos:

    st.subheader(
        "📸 Photos of Recommended Places"
    )

    st.write(
        "Voyage selects the main attractions from "
        "your itinerary and finds a labelled image "
        "for each place."
    )

    if not profile:

        st.info(
            "Create an itinerary first."
        )

    else:

        # ----------------------------------------
        # AUTOMATIC ITINERARY GALLERY
        # ----------------------------------------

        if st.button(
            "✨ Show photos from my itinerary",
            use_container_width=True
        ):

            with st.spinner(
                "Finding the places in your itinerary..."
            ):

                places = extract_photo_places()

                st.session_state.photo_places = places

        places = st.session_state.get(
            "photo_places",
            []
        )
        place_override = st.text_area("Attractions for photos (one place per line)",
            value="\n".join(places), key="photo_place_override",
            help="Enter attractions directly to avoid extra AI requests.")
        if place_override.strip():
            places = [x.strip() for x in place_override.splitlines() if x.strip()][:5]


        if places:

            st.caption(
                "These are the main places Voyage "
                "identified in your itinerary."
            )

            # Display two cards per row
            for i in range(
                0,
                len(places),
                2
            ):

                cols = st.columns(2)

                for j in range(2):

                    index = i + j

                    if index >= len(places):
                        break

                    place = places[index]

                    with cols[j]:

                        st.markdown(
                            f"### 📍 {place}"
                        )

                        with st.spinner(
                            f"Finding photo of {place}..."
                        ):

                            image = (
                                best_commons_image(
                                    place
                                )
                            )

                        if image:

                            st.image(
                                image["url"],
                                caption=place,
                                use_container_width=True
                            )

                            if image["source"]:

                                st.link_button(
                                    "Image source / licence",
                                    image["source"],
                                    use_container_width=True
                                )

                        else:

                            st.info(
                                "No suitable photograph "
                                "was found automatically."
                            )

        # ----------------------------------------
        # MANUAL PLACE SEARCH
        # ----------------------------------------

        st.divider()

        st.subheader(
            "🔎 Search another place"
        )

        manual_place = st.text_input(
            "Specific attraction",
            placeholder=(
                "e.g. Gyeongbokgung Palace"
            )
        )

        if manual_place:

            image = best_commons_image(
                manual_place
            )

            if image:

                st.markdown(
                    f"### 📍 {manual_place}"
                )

                st.image(
                    image["url"],
                    caption=manual_place,
                    use_container_width=True
                )

                if image["source"]:

                    st.link_button(
                        "Image source / licence",
                        image["source"]
                    )

            else:

                st.warning(
                    "I couldn't find a suitable "
                    "photograph for that exact place."
                )
 

with tab_memory:
    st.write("Trip memories stay in your current browser session. Download your private JSON backup to keep them between sessions; they are NOT saved to a shared server file.")
    upload_mem = st.file_uploader("Import your trip memories (JSON)", type=["json"], key="memory_import")
    if upload_mem is not None and st.button("Import memories"):
        try:
            data = json.loads(upload_mem.getvalue().decode("utf-8"))
            if not isinstance(data, list) or len(data) > 100 or not all(isinstance(x, dict) for x in data):
                raise ValueError("Invalid memory format")
            st.session_state.past_trips = [{"trip": str(x.get("trip", ""))[:100],
                "preferences": str(x.get("preferences", ""))[:1000],
                "saved_at": str(x.get("saved_at", ""))[:30]} for x in data]
            st.success("Imported into this session.")
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            st.error("Please upload a valid Voyage memory JSON file.")
    memory_name = st.text_input("Trip name", placeholder="e.g. Manali winter trip")
    memory_likes = st.text_area("What did you like/dislike?",
                                placeholder="Loved mountain views and cafés; disliked rushed sightseeing.")
    if st.button("💾 Save to trip memory"):
        if memory_name and memory_likes:
            item = {"trip":memory_name,"preferences":memory_likes,
                    "saved_at":datetime.now().isoformat(timespec="seconds")}
            st.session_state.past_trips.append(item)
            save_past_trips(st.session_state.past_trips)
            st.success("Saved.")
        else:
            st.warning("Add both fields.")
    if st.session_state.past_trips:
        st.download_button("⬇️ Download my private trip memories", 
            json.dumps(st.session_state.past_trips, ensure_ascii=False, indent=2),
            file_name="my_voyage_memories.json", mime="application/json")
        if st.button("Clear my trip memories"):
            st.session_state.past_trips = []
            st.rerun()
        st.subheader("Saved memories")
        for item in reversed(st.session_state.past_trips[-8:]):
            st.markdown(f"**{item.get('trip','Trip')}** — {item.get('preferences','')}")
    else:
        st.caption("No past-trip memories saved yet.")

st.divider()
st.caption("Voyage is an AI planning prototype. Verify prices, availability, travel rules, closures and long-range weather before booking.")

with tab_tools:
    st.subheader("🗺️ Trip Tools")
    if not profile:
        st.info("Set up a trip first to use the budget, packing, and map tools.")
    else:
        st.markdown("### 💰 Budget allocation")
        st.caption("Illustrative percentage-based planning allocation only — not live prices or a guarantee of affordability. Check real transport and accommodation quotes before booking.")
        currency, allocation = budget_allocations(profile)
        st.bar_chart(pd.DataFrame({"Category": list(allocation), "Amount": list(allocation.values())}).set_index("Category"), horizontal=True)
        for category, value in allocation.items():
            st.write(f"**{category}:** {currency} {value:,}")
        st.divider()
        st.markdown("### 🎒 Packing checklist")
        st.caption("General suggestions based on your trip profile; check local rules and actual travel-date weather.")
        for item in packing_suggestions(profile):
            st.checkbox(item, key="pack_" + hashlib.sha256(item.encode()).hexdigest()[:12])
        st.divider()
        st.markdown("### 📍 Interactive map")
        st.caption("Enter coordinates for trip stops to see them on an interactive OpenStreetMap-backed map. No paid map API or Gemini request required.")
        st.caption("Format: Place name, latitude, longitude — one place per line. Coordinates can be copied from a map listing.")
        map_text = st.text_area("Map stops", placeholder="Eiffel Tower, 48.8584, 2.2945\nLouvre Museum, 48.8606, 2.3376", key="map_stops")
        markers = []
        if map_text.strip():
            for line in map_text.splitlines()[:25]:
                fields = [v.strip() for v in line.rsplit(",", 2)]
                if len(fields) != 3:
                    st.warning(f"Invalid map row: {line[:70]}")
                    continue
                try:
                    lat, lon = float(fields[1]), float(fields[2])
                    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                        raise ValueError()
                    markers.append({"name": fields[0][:80], "lat": lat, "lon": lon})
                except ValueError:
                    st.warning(f"Invalid coordinates: {line[:70]}")
            if markers:
                st.map(markers, latitude="lat", longitude="lon", size=120)
                for m in markers:
                    st.link_button("📍 " + m["name"], maps_search_link(m["name"]))
        else:
            dest = profile.get("destination", "")
            if dest and dest != "not decided":
                st.link_button("Find places in Google Maps", maps_search_link(dest))
