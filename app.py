import os, json
import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig, Tool, GoogleSearch

load_dotenv()

st.set_page_config(page_title="AI Travel Assistant", page_icon="✈️", layout="wide")

SYSTEM_PROMPT = """
You are Voyage, an intelligent AI Travel Assistant.
Understand a traveler's budget, season, duration, destination, group size, interests,
pace, accommodation and transport preferences. Ask concise follow-up questions when
important information is missing. Once enough information is known, create a realistic
personalized itinerary. Explain the overall mood/style of the trip. Give practical
food, transport, accommodation, hidden-gem, packing and safety advice.
Include "Important local things to know" for unusual laws, customs, etiquette, permits,
photography restrictions, environmental rules and similar destination-specific issues.
Never invent current rules, closures, weather or prices. Use web grounding for current
information when available. When the user asks for a change, revise the existing plan
rather than starting from scratch. Use clear Markdown and useful headings.
"""

@st.cache_resource
def make_client():
    key = os.getenv("GOOGLE_API_KEY")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    if key:
        return genai.Client(vertexai=True, api_key=key)
    if project:
        return genai.Client(vertexai=True, project=project, location=location,
                            http_options=HttpOptions(api_version="v1"))
    return None

client = make_client()

def ask_ai(messages):
    if not client:
        return "Your Google AI connection is not configured yet. Add your Google credential to `.env`, then restart the app."
    contents = [
        {"role": "user" if m["role"] == "user" else "model",
         "parts": [{"text": m["content"]}]}
        for m in messages
    ]
    cfg = GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.55,
        tools=[Tool(google_search=GoogleSearch())]
    )
    response = client.models.generate_content(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
        contents=contents,
        config=cfg
    )
    return response.text

@st.cache_data(ttl=86400)
def destination_image(destination):
    try:
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action":"query", "generator":"search",
                "gsrsearch":f"{destination} travel", "gsrnamespace":6,
                "gsrlimit":5, "prop":"imageinfo", "iiprop":"url",
                "iiurlwidth":1200, "format":"json"
            },
            headers={"User-Agent":"AI-Travel-Assistant-School-Project/1.0"},
            timeout=8
        )
        for p in r.json().get("query", {}).get("pages", {}).values():
            info = p.get("imageinfo", [{}])[0]
            if info.get("thumburl") or info.get("url"):
                return info.get("thumburl") or info.get("url")
    except Exception:
        pass
    return None

if "messages" not in st.session_state:
    st.session_state.messages = []
if "profile" not in st.session_state:
    st.session_state.profile = {}

st.markdown("""
<div style="padding:1.5rem;border-radius:20px;background:linear-gradient(135deg,#eaf4ff,#f7f0ff)">
<h1>✈️ AI Travel Assistant</h1>
<p style="font-size:1.1rem">Plan smarter. Travel better. Refine your journey through conversation.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🧳 Start your trip")
    destination = st.text_input("Destination (optional)", placeholder="e.g. Kyoto")
    budget = st.number_input("Budget", min_value=0, value=30000, step=1000)
    currency = st.selectbox("Currency", ["INR","USD","EUR","GBP","JPY"])
    days = st.number_input("Number of days", min_value=1, max_value=60, value=5)
    season = st.selectbox("Travel season", ["Summer","Winter","Spring","Autumn / Fall","Monsoon","Flexible"])
    travellers = st.number_input("Number of travellers", min_value=1, max_value=50, value=2)
    interests = st.multiselect("Interests", [
        "Adventure","Nature","Beaches","Mountains","Culture & History","Food",
        "Shopping","Architecture / Skyscrapers","Nightlife","Photography",
        "Relaxation","Family activities","Hidden gems"
    ])
    pace = st.select_slider("Preferred pace", ["Relaxed","Balanced","Packed"], value="Balanced")
    accommodation = st.selectbox("Accommodation", ["Budget","Mid-range","Luxury","Hostel / Homestay","Flexible"])
    transport = st.multiselect("Preferred transport",
                               ["Walking","Public transport","Train","Bus","Cab","Rental car","Flight"])

    if st.button("✨ Start / Update Trip", use_container_width=True):
        profile = {
            "destination": destination or "not decided yet",
            "budget": f"{currency} {budget:,}", "days": days, "season": season,
            "travellers": travellers, "interests": interests, "pace": pace,
            "accommodation": accommodation, "transport": transport
        }
        st.session_state.profile = profile
        st.session_state.messages.append({
            "role":"user",
            "content": "INITIAL TRIP PROFILE:\n" + json.dumps(profile, indent=2) +
                       "\nStart planning. Do not ask for information already supplied."
        })
        with st.spinner("Planning your trip..."):
            answer = ask_ai(st.session_state.messages)
        st.session_state.messages.append({"role":"assistant","content":answer})
        st.rerun()

    if st.button("🗑️ Reset", use_container_width=True):
        st.session_state.messages = []
        st.session_state.profile = {}
        st.rerun()

if not st.session_state.messages:
    st.subheader("Tell me what kind of trip you want 🌍")
    st.write("Set the basics on the left, then continue the planning conversation below.")
    a,b,c = st.columns(3)
    a.info("🗺️ Personalized itineraries")
    b.info("💰 Budget-aware planning")
    c.info("🌱 Responsible travel")
else:
    dest = st.session_state.profile.get("destination")
    if dest and dest != "not decided yet":
        img = destination_image(dest)
        if img:
            st.image(img, caption=f"Destination inspiration: {dest}", use_container_width=True)

    for m in st.session_state.messages:
        if m["role"] == "user":
            if m["content"].startswith("INITIAL TRIP PROFILE:"):
                continue
            with st.chat_message("user"):
                st.markdown(m["content"])
        else:
            with st.chat_message("assistant", avatar="✈️"):
                st.markdown(m["content"])

    text = st.chat_input("Ask me to change or improve your trip...")
    if text:
        st.session_state.messages.append({"role":"user","content":text})
        with st.chat_message("user"):
            st.markdown(text)
        with st.chat_message("assistant", avatar="✈️"):
            with st.spinner("Thinking..."):
                answer = ask_ai(st.session_state.messages)
            st.markdown(answer)
        st.session_state.messages.append({"role":"assistant","content":answer})

st.divider()
st.caption("AI Travel Assistant • Class XII AI Project Prototype • Verify time-sensitive travel information before acting on it.")
