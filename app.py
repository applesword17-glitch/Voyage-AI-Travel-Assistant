import os
import io
import json
import time
import re
import urllib.parse
from datetime import date, datetime
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm

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
    st.error(
        "GOOGLE_API_KEY is missing. "
        "Add it to your local .env file or Streamlit Secrets."
    )
    st.stop()
st.markdown("""
<style>
.block-container {padding-top:1.2rem; padding-bottom:4rem; max-width:1200px;}
.hero {padding:1.4rem 1.6rem;border-radius:22px;background:linear-gradient(135deg,#eef6ff,#f8f3ff);border:1px solid rgba(0,0,0,.06);margin-bottom:1rem;}
.hero h1{margin:0 0 .25rem 0}.hero p{margin:0;opacity:.78}
.pill{display:inline-block;padding:.25rem .65rem;margin:.12rem;border-radius:999px;border:1px solid rgba(0,0,0,.12);font-size:.86rem}
</style>
""", unsafe_allow_html=True)

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


def call_gemini(extra_instruction=None, retries=3):
    """
    Sends the current Voyage conversation to Gemini.

    Features:
    - Uses Google Search grounding
    - Retries temporary 503 errors
    - Handles quota / 429 errors gracefully
    - Returns grounded citations through add_grounding_citations()
    """

    prompt = build_context()

    # Add any special instruction for this particular request
    if extra_instruction:
        prompt += (
            "\n\nTASK FOR THIS TURN:\n"
            + extra_instruction
        )

    last_error = None

    for attempt in range(retries):

        try:
            # Create a fresh Gemini client for this request
            client = genai.Client(
                api_key=API_KEY
            )

            # Enable Google Search grounding
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )

            config = types.GenerateContentConfig(
                tools=[grounding_tool]
            )

            # Send request to Gemini
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=config
            )

            # Add source citations when grounding data is available
            return add_grounding_citations(response)

        except Exception as e:

            last_error = e
            msg = str(e).lower()

            # -----------------------------------------
            # QUOTA / RATE LIMIT ERROR
            # -----------------------------------------
            if (
                "429" in msg
                or "resource_exhausted" in msg
                or "quota" in msg
            ):
                return (
                    "⚠️ **Voyage's AI service has temporarily "
                    "reached its request limit.**\n\n"
                    "Please try again later. Your trip details "
                    "are still saved in Voyage."
                )

            # -----------------------------------------
            # TEMPORARY GEMINI SERVER ERROR
            # -----------------------------------------
            if (
                "503" in msg
                or "unavailable" in msg
                or "high demand" in msg
            ):

                # Wait slightly longer after each failed attempt
                time.sleep(2 + attempt * 2)

                continue

            # -----------------------------------------
            # CLOSED CLIENT ERROR
            # -----------------------------------------
            if "client has been closed" in msg:

                # Try again with a fresh client
                time.sleep(1)

                continue

            # Unknown error - show it instead of hiding it
            raise

    # All retries failed
    raise last_error

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
    """
    Uses the latest itinerary to identify the most important
    specific attractions that deserve photographs.
    """

    if not st.session_state.latest_plan:
        return []

    prompt = f"""
Read this travel itinerary:

{st.session_state.latest_plan}

Return ONLY a JSON list containing the names of the
5 most important specific attractions or places that
the traveller will actually visit.

Rules:
- Specific places only.
- Do NOT return countries.
- Do NOT return generic cities unless absolutely necessary.
- Prefer landmarks, neighbourhoods, parks, palaces,
  beaches, museums, towers, temples, lakes, etc.
- Maximum 5 places.

Example:

[
  "Gyeongbokgung Palace",
  "Bukchon Hanok Village",
  "N Seoul Tower",
  "Myeongdong",
  "Dongdaemun Design Plaza"
]

Return ONLY the JSON list.
"""

    try:

        client = genai.Client(
            api_key=API_KEY
        )

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        text = response.text.strip()

        # Remove Markdown code fences if Gemini adds them
        text = text.replace(
            "```json", ""
        ).replace(
            "```", ""
        ).strip()

        places = json.loads(text)

        if isinstance(places, list):
            return places[:5]

    except Exception:
        pass

    return []
def extract_itinerary_cities():
    """
    Uses the latest itinerary to identify the main cities
    or overnight bases where the traveller will stay.
    """

    if not st.session_state.latest_plan:
        return []

    prompt = f"""
Read this travel itinerary:

{st.session_state.latest_plan}

Return ONLY a JSON list containing the main cities,
towns, or overnight bases where the traveller actually
spends significant time.

Rules:
- Return places in itinerary order.
- Do not include individual attractions.
- Do not include countries.
- Do not include airports.
- Do not repeat cities.
- Maximum 6 places.

Example:
[
  "Rome, Italy",
  "Florence, Italy",
  "Venice, Italy"
]

Return ONLY the JSON list.
"""

    try:
        client = genai.Client(
            api_key=API_KEY
        )

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        text = response.text.strip()

        text = (
            text.replace("```json", "")
            .replace("```", "")
            .strip()
        )

        cities = json.loads(text)

        if isinstance(cities, list):
            return cities[:6]

    except Exception:
        pass

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

MEMORY_FILE = Path("voyage_trips.json")

def load_past_trips():
    try:
        if MEMORY_FILE.exists():
            data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
    except Exception:
        pass
    return []

def save_past_trips(trips):
    try:
        MEMORY_FILE.write_text(json.dumps(trips, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

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
            st.error(f"Gemini could not answer right now: {e}")
            st.stop()
    st.session_state.messages = [{"role":"assistant","content":answer}]
    st.session_state.latest_plan = answer
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

tab_plan, tab_weather, tab_photos, tab_memory = st.tabs(["💬 Plan & Chat","🌦️ Weather","🖼️ Photos","🧠 Trip Memory"])

with tab_plan:
    if not st.session_state.messages:
        st.info("Set preferences in the sidebar and click **Build my trip**.")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="✈️" if message["role"]=="assistant" else None):
                st.markdown(message["content"])

    user_message = st.chat_input("Ask Voyage to refine, compare, explain, or change your trip...")
    if user_message:
        st.session_state.messages.append({"role":"user","content":user_message})
        with st.spinner("Voyage is thinking..."):
            try:
                answer = call_gemini()
            except Exception as e:
                st.error(f"Gemini could not answer right now: {e}")
                st.stop()
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
        "Voyage identifies the main cities in your itinerary "
        "and shows the weather for each stop."
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

        # ----------------------------------------
        # WEATHER FOR EACH CITY
        # ----------------------------------------

        if cities:

            st.caption(
                "Weather shown below is live forecast data "
                "where available."
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
    st.write("Save a short summary of a trip you liked or disliked. Voyage can use it for future recommendations on this computer.")
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
        st.subheader("Saved memories")
        for item in reversed(st.session_state.past_trips[-8:]):
            st.markdown(f"**{item.get('trip','Trip')}** — {item.get('preferences','')}")
    else:
        st.caption("No past-trip memories saved yet.")

st.divider()
st.caption("Voyage is an AI planning prototype. Verify prices, availability, travel rules, closures and long-range weather before booking.")
