import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# Load model & encoders
model = joblib.load("flight_delay_model.pkl")
encoders = joblib.load("encoders.pkl")

# Load dataset (for dropdown options)
df = pd.read_csv("flights_weather_ready.csv")

# Weather API Key (signup at https://openweathermap.org/ for free API key)
API_KEY = ""

def get_weather(city):
    """Fetch current weather condition from OpenWeather API"""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        res = requests.get(url).json()
        if "weather" in res:
            condition = res["weather"][0]["main"].lower()  # clear, rain, snow...
            temp = res["main"]["temp"]
            return condition, temp
    except:
        return "clear", None
    return "clear", None

st.set_page_config(page_title="Flight Delay Predictor", layout="wide")

# Background image
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("flight image.jpeg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title("✈️ Flight Delay Predictor")
st.write("Select your flight details to predict delay chances.")

# Dropdowns
airline = st.selectbox("Airline", sorted(df['airline'].unique()))
departure = st.selectbox("Departure City", sorted(df['departure'].unique()))
destination = st.selectbox("Destination City", sorted(df['destination'].unique()))
distance = st.number_input("Distance (km)", min_value=100, max_value=10000, step=50)

# Show live weather for departure & destination
dep_weather, dep_temp = get_weather(departure)
dest_weather, dest_temp = get_weather(destination)

st.subheader("🌦 Current Weather Updates")
col1, col2 = st.columns(2)
with col1:
    st.info(f"**{departure}** → {dep_weather.title()} ({dep_temp}°C)" if dep_temp else f"{departure} → {dep_weather.title()}")
with col2:
    st.info(f"**{destination}** → {dest_weather.title()} ({dest_temp}°C)" if dest_temp else f"{destination} → {dest_weather.title()}")

# Predict button
if st.button("Predict Delay"):
    # Use departure weather for prediction (you can also combine)
    weather_used = dep_weather if dep_weather in encoders['weather'].classes_ else "clear"

    # Encode inputs
    input_data = pd.DataFrame({
        'airline': [encoders['airline'].transform([airline])[0]],
        'departure': [encoders['departure'].transform([departure])[0]],
        'destination': [encoders['destination'].transform([destination])[0]],
        'distance': [distance],
        'weather': [encoders['weather'].transform([weather_used])[0]]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Delay Category: **{prediction}**")
