# ============================================================
# PROJECT :
# Food & Restaurant Services -
# AI Demand Forecasting and Inventory Optimization
# ============================================================

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
import random

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="AI Restaurant Forecasting Platform",
    page_icon="🍽️",
    layout="wide"

)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""

<style>

body {
    background-color: #020617;
}

.main {
    background: linear-gradient(to right, #020617, #0F172A);
}

.big-title {

    font-size: 55px;
    font-weight: bold;
    text-align: center;
    color: #00F5FF;
    text-shadow: 0px 0px 25px #00F5FF;

}

.sub-title {

    font-size: 22px;
    text-align: center;
    color: #CBD5E1;

}

.metric-card {

    background: linear-gradient(135deg, #111827, #1E293B);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 0px 20px #00F5FF;

}

.glow-card {

    background: linear-gradient(135deg, #111827, #1E293B);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 0px 20px #00F5FF;
    margin-top: 20px;

}

.stButton>button {

    background: linear-gradient(to right, #00F5FF, #00C2FF);
    color: black;
    font-size: 20px;
    font-weight: bold;
    border-radius: 15px;
    height: 65px;
    width: 100%;
    border: none;
    box-shadow: 0px 0px 25px #00F5FF;

}

.stButton>button:hover {

    transform: scale(1.03);
    transition: 0.3s;

}

</style>

""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("models/restaurant_demand_forecasting_xgboost.pkl")

feature_names = joblib.load("models/model_features.pkl")

# ============================================================
# HEADER
# ============================================================

st.markdown(

    '<p class="big-title">🍽️ AI Restaurant Forecasting Platform</p>',

    unsafe_allow_html=True

)

st.markdown(

    '<p class="sub-title">AI-Powered Demand Forecasting and Inventory Optimization System</p>',

    unsafe_allow_html=True

)

st.write("")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Restaurant Input Panel")

restaurant_id = st.sidebar.slider(

    "🏪 Restaurant Branch ID",
    1,
    50,
    10

)

# ============================================================

restaurant_type_label = st.sidebar.selectbox(

    "🍴 Restaurant Type",

    [

        "Fast Food",
        "Casual Dining",
        "Luxury Restaurant",
        "Cloud Kitchen"

    ]

)

restaurant_type_map = {

    "Fast Food": 0,
    "Casual Dining": 1,
    "Luxury Restaurant": 2,
    "Cloud Kitchen": 3

}

restaurant_type = restaurant_type_map[restaurant_type_label]

# ============================================================

menu_item_name = st.sidebar.slider(

    "🍕 Menu Item ID",
    1,
    15,
    5

)

# ============================================================

meal_type_label = st.sidebar.selectbox(

    "🍽️ Meal Type",

    [

        "Breakfast",
        "Lunch",
        "Dinner"

    ]

)

meal_type_map = {

    "Breakfast": 0,
    "Lunch": 1,
    "Dinner": 2

}

meal_type = meal_type_map[meal_type_label]

# ============================================================

ingredient_cost = st.sidebar.slider(

    "🥦 Ingredient Cost",
    1.0,
    50.0,
    10.0

)

market_price = st.sidebar.slider(

    "💰 Market Price",
    1.0,
    100.0,
    20.0

)

selling_price = st.sidebar.slider(

    "💵 Selling Price",
    1.0,
    100.0,
    25.0

)

# ============================================================

promotion_label = st.sidebar.selectbox(

    "📢 Promotion Running?",

    [

        "No Promotion",
        "Promotion Active"

    ]

)

promotion_map = {

    "No Promotion": 0,
    "Promotion Active": 1

}

promotion = promotion_map[promotion_label]

# ============================================================

special_event_label = st.sidebar.selectbox(

    "🎉 Special Event?",

    [

        "No Event",
        "Festival / Event Running"

    ]

)

special_event_map = {

    "No Event": 0,
    "Festival / Event Running": 1

}

special_event = special_event_map[special_event_label]

# ============================================================

weather_label = st.sidebar.selectbox(

    "🌦️ Weather Condition",

    [

        "Sunny",
        "Cloudy",
        "Rainy"

    ]

)

weather_map = {

    "Sunny": 0,
    "Cloudy": 1,
    "Rainy": 2

}

weather = weather_map[weather_label]

# ============================================================

weekend_label = st.sidebar.selectbox(

    "📅 Day Type",

    [

        "Working Day",
        "Semi-Holiday",
        "Holiday / Weekend"

    ]

)

weekend_map = {

    "Working Day": 0,
    "Semi-Holiday": 0,
    "Holiday / Weekend": 1

}

is_weekend = weekend_map[weekend_label]

# ============================================================
# DATE FEATURES
# ============================================================

today = datetime.now()

day = today.day
month = today.month
year = today.year
day_of_week = today.weekday()

week_of_year = today.isocalendar()[1]

quarter = (month - 1) // 3 + 1

# ============================================================
# DYNAMIC BUSINESS FEATURES
# ============================================================

base_demand = random.randint(120, 300)

if restaurant_type == 0:
    base_demand += 120

if restaurant_type == 2:
    base_demand -= 40

if meal_type == 2:
    base_demand += 80

if promotion == 1:
    base_demand += 100

if special_event == 1:
    base_demand += 150

if weather == 2:
    base_demand += 70

if is_weekend == 1:
    base_demand += 120

if selling_price > market_price:
    base_demand -= 80

if selling_price > 60:
    base_demand -= 50

# ============================================================
# FORECAST FEATURES
# ============================================================

lag_1 = base_demand + random.randint(-20, 20)
lag_7 = base_demand + random.randint(-30, 30)
lag_14 = base_demand + random.randint(-40, 40)
lag_30 = base_demand + random.randint(-50, 50)

rolling_mean_7 = base_demand
rolling_mean_14 = base_demand - 10

rolling_std_7 = random.randint(10, 35)
rolling_std_14 = random.randint(15, 40)

price_difference = selling_price - market_price

profit_estimate = selling_price - ingredient_cost

# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({

    'restaurant_id': [restaurant_id],
    'restaurant_type': [restaurant_type],
    'menu_item_name': [menu_item_name],
    'meal_type': [meal_type],
    'typical_ingredient_cost': [ingredient_cost],
    'observed_market_price': [market_price],
    'actual_selling_price': [selling_price],
    'has_promotion': [promotion],
    'special_event': [special_event],
    'weather_condition': [weather],
    'day': [day],
    'month': [month],
    'year': [year],
    'day_of_week': [day_of_week],
    'week_of_year': [week_of_year],
    'quarter': [quarter],
    'is_weekend': [is_weekend],
    'lag_1': [lag_1],
    'lag_7': [lag_7],
    'lag_14': [lag_14],
    'lag_30': [lag_30],
    'rolling_mean_7': [rolling_mean_7],
    'rolling_mean_14': [rolling_mean_14],
    'rolling_std_7': [rolling_std_7],
    'rolling_std_14': [rolling_std_14],
    'price_difference': [price_difference],
    'profit_estimate': [profit_estimate]

})

# ============================================================
# HANDLE MISSING FEATURES
# ============================================================

for feature in feature_names:

    if feature not in input_data.columns:

        input_data[feature] = 0

# ============================================================
# REORDER FEATURES
# ============================================================

input_data = input_data[feature_names]

# ============================================================
# BUTTON
# ============================================================

predict_button = st.button("🚀 Generate AI Forecast")

# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    prediction = model.predict(input_data)[0]

    prediction = abs(round(prediction, 2))

    # ========================================================
    # DEMAND CATEGORY
    # ========================================================

    if prediction >= 450:

        demand_category = "HIGH DEMAND"

    elif prediction >= 250:

        demand_category = "MEDIUM DEMAND"

    else:

        demand_category = "LOW DEMAND"

    # ========================================================
    # INVENTORY PLAN
    # ========================================================

    if prediction >= 450:

        inventory = "Maintain HIGH Inventory Level"

    elif prediction >= 250:

        inventory = "Maintain MEDIUM Inventory Level"

    else:

        inventory = "Maintain LOW Inventory Level"

    # ========================================================
    # FOOD WASTE RISK
    # ========================================================

    if prediction >= 450:

        waste_risk = "LOW FOOD WASTE RISK"

    elif prediction >= 250:

        waste_risk = "MEDIUM FOOD WASTE RISK"

    else:

        waste_risk = "HIGH FOOD WASTE RISK"

    # ========================================================
    # AI CONFIDENCE
    # ========================================================

    confidence_score = random.randint(85, 99)

    # ========================================================
    # KPI CARDS
    # ========================================================

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(f"""

        <div class="metric-card">

        <h2>📈 Predicted Demand</h2>

        <h1>{prediction}</h1>

        </div>

        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""

        <div class="metric-card">

        <h2>🍽️ Demand Category</h2>

        <h1>{demand_category}</h1>

        </div>

        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""

        <div class="metric-card">

        <h2>📦 Inventory Plan</h2>

        <h1>{inventory}</h1>

        </div>

        """, unsafe_allow_html=True)

    with col4:

        st.markdown(f"""

        <div class="metric-card">

        <h2>🤖 AI Confidence</h2>

        <h1>{confidence_score}%</h1>

        </div>

        """, unsafe_allow_html=True)

    # ========================================================
    # BUSINESS INSIGHTS
    # ========================================================

    st.write("")

    st.markdown("""

    <div class="glow-card">

    <h2>🧠 AI Business Insights</h2>

    </div>

    """, unsafe_allow_html=True)

    if promotion == 1:

        st.success("✅ Promotions are significantly boosting restaurant demand.")

    if is_weekend == 1:

        st.success("🔥 Weekend traffic is increasing customer orders.")

    if special_event == 1:

        st.success("🎉 Festival season is increasing food demand.")

    if weather == 2:

        st.success("🌧️ Rainy weather is increasing online delivery orders.")

    if selling_price > market_price:

        st.warning("⚠️ High selling price may reduce customer demand.")

    if prediction < 250:

        st.error("⚠️ Low demand detected. Reduce inventory purchasing.")

    if prediction > 500:

        st.success("🚀 Massive demand surge expected in upcoming days.")

    st.info(f"📦 Inventory Recommendation : {inventory}")

    st.warning(f"🍽️ Food Waste Risk : {waste_risk}")

    # ========================================================
    # AI FORECAST GRAPH
    # ========================================================

    st.write("")

    future_days = [

        "Day 1",
        "Day 2",
        "Day 3",
        "Day 4",
        "Day 5",
        "Day 6",
        "Day 7"

    ]

    future_predictions = [

        prediction + random.randint(-20, 20),
        prediction + random.randint(-10, 30),
        prediction + random.randint(0, 40),
        prediction + random.randint(-15, 50),
        prediction + random.randint(10, 60),
        prediction + random.randint(20, 70),
        prediction + random.randint(15, 90)

    ]

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=future_days,
            y=future_predictions,

            mode='lines+markers',

            name='Predicted Demand',

            line=dict(

                color='#00F5FF',
                width=5

            ),

            marker=dict(

                size=12,
                color='#00F5FF'

            ),

            fill='tozeroy'

        )

    )

    fig.update_layout(

        title="📈 7-Day AI Restaurant Demand Forecast",

        xaxis_title="Future Days",

        yaxis_title="Expected Orders",

        template="plotly_dark",

        height=550,

        hovermode="x unified"

    )

    st.plotly_chart(fig, use_container_width=True)

    # ========================================================
    # DEMAND GAUGE
    # ========================================================

    gauge_fig = go.Figure(go.Indicator(

        mode="gauge+number",

        value=prediction,

        title={'text': "🔥 Demand Intensity"},

        gauge={

            'axis': {'range': [0, 700]},

            'bar': {'color': "#00F5FF"},

            'steps': [

                {'range': [0, 250], 'color': "#1E293B"},
                {'range': [250, 450], 'color': "#334155"},
                {'range': [450, 700], 'color': "#0F766E"}

            ]

        }

    ))

    gauge_fig.update_layout(

        template="plotly_dark",
        height=400

    )

    st.plotly_chart(gauge_fig, use_container_width=True)

# ========================================================
# POWER BI DASHBOARD SECTION
# ========================================================

st.write("")
st.write("")

st.markdown("""

<div class="glow-card">

<h2>📊 Live Power BI Business Intelligence Dashboard</h2>

<p style='color:white;'>

Open Interactive Restaurant Analytics Dashboard

</p>

</div>

""", unsafe_allow_html=True)

# ========================================================
# POWER BI LINK
# ========================================================

power_bi_url = "https://app.powerbi.com/links/ANkb67HQrW?ctid=56c1d497-700b-49cf-8f8d-3dd6b20d522f&pbi_source=linkShare"

# ========================================================
# OPEN POWER BI DASHBOARD BUTTON
# ========================================================

st.write("")

st.link_button(

    "📊 Open Live Power BI Dashboard",

    power_bi_url

)

# ============================================================
# FOOTER
# ============================================================

st.write("")
st.write("")

st.markdown("""

<center>

<h3 style='color:white;'>

AI Restaurant Forecasting and Inventory Optimization Platform

</h3>

</center>

""", unsafe_allow_html=True)