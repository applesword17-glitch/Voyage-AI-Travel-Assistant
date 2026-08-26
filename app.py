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
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.6-flash")

st.set_page_config(page_title="Voyage | AI Travel Assistant", page_icon="✈️", layout="wide")

if not API_KEY:
    st.error("GOOGLE_API_KEY is missing. Put it in your .env file.")
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

def call_gemini(extra_instruction=None, retries=3):
    prompt = build_context()
    if extra_instruction:
        prompt += "\nTASK FOR THIS TURN:\n" + extra_instruction
    last_error = None
    for attempt in range(retries):
        try:
            response = new_client().models.generate_content(model=MODEL, contents=prompt)
            return response.text
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            if "503" in msg or "unavailable" in msg or "high demand" in msg:
                time.sleep(2 + attempt * 2)
                continue
            raise
    raise last_error

@st.cache_data(ttl=3600)
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
def commons_images(query, limit=4):
    if not query:
        return []
    params = {
        "action":"query","generator":"search","gsrsearch":f"{query} travel landmark",
        "gsrnamespace":6,"gsrlimit":max(limit*2,6),"prop":"imageinfo",
        "iiprop":"url|extmetadata","iiurlwidth":1000,"format":"json","origin":"*",
    }
    try:
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params=params, timeout=12,
            headers={"User-Agent":"VoyageSchoolAIProject/1.0"}
        )
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        out = []
        for p in pages.values():
            info = (p.get("imageinfo") or [{}])[0]
            img = info.get("thumburl") or info.get("url")
            if img:
                out.append({"url":img,"title":p.get("title","").replace("File:",""),
                            "source":info.get("descriptionurl","")})
            if len(out) >= limit:
                break
        return out
    except Exception:
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
                "Create the first complete itinerary now. If destination is not decided, "
                "recommend one best-fit destination first and build the itinerary for it. "
                "Include budget, example stays, transport, packing, practical local information, "
                "and responsible travel."
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
    if not profile:
        st.info("Create a trip first.")
    else:
        city = st.text_input("City for weather",
                             value=profile.get("destination","") if profile.get("destination")!="not decided" else "")
        if city:
            try:
                forecast = live_forecast(city)
            except Exception as e:
                forecast = None
                st.warning(f"Weather service unavailable: {e}")
            if forecast:
                loc = forecast["location"]; daily = forecast["daily"]
                st.write(f"**Live forecast near {loc.get('name')}, {loc.get('country','')}**")
                rows = []
                for i,d in enumerate(daily.get("time",[])):
                    rows.append({
                        "Date":d,
                        "Conditions":weather_code_text(daily.get("weather_code",[None])[i]),
                        "Min °C":daily.get("temperature_2m_min",[None])[i],
                        "Max °C":daily.get("temperature_2m_max",[None])[i],
                        "Rain chance %":daily.get("precipitation_probability_max",[None])[i],
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)
                st.caption("Exact forecast is only available within the live forecast window. Later dates should be treated as seasonal estimates.")

with tab_photos:
    if not profile:
        st.info("Create a trip first.")
    else:
        photo_place = st.text_input("Place or attraction",
                                    value=profile.get("destination","") if profile.get("destination")!="not decided" else "",
                                    placeholder="e.g. Lake Como, Colosseum, Kyoto")
        if photo_place:
            imgs = commons_images(photo_place, 4)
            if imgs:
                cols = st.columns(2)
                for i,img in enumerate(imgs):
                    with cols[i%2]:
                        st.image(img["url"], caption=img["title"], use_container_width=True)
                        if img["source"]:
                            st.link_button("Image source / license", img["source"])
            else:
                st.warning("No suitable Wikimedia Commons photos were found.")

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
