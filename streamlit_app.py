import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import MySQLdb as mysql
import mysql.connector
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from prophet import Prophet

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('global_prophet_model_best.pkl', 'rb') as f:
            model = pickle.load(f)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
    return model

# Function to retrieve data for the last 14 days from the database
def get_game_data(host, port, database, user, password, user_input):
    try:
        conn = mysql.connector.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        st.success("Database connection successful.")
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

    # Extract the part after the underscore (e.g., LN69130_342409 -> LN342409)
    table_suffix = user_input.split('_')[-1]
    table_name = f"LN{table_suffix}_sales"

    # SQL query to get the last 14 days of data from the sales table
    query = f"""
    SELECT date, unit_sold
    FROM {table_name}
    ORDER BY date DESC
    LIMIT 14;
    """

    try:
        game_data = pd.read_sql_query(query, conn)
        st.success("Data retrieval successful.")
    except Exception as e:
        st.error(f"Error retrieving data: {e}")
        return None
    finally:
        conn.close()

    # Process the data to match the format needed by the Prophet model
    game_data['ds'] = pd.to_datetime(game_data['date'])
    game_data = game_data.rename(columns={'unit_sold': 'y'})
    game_data = game_data.sort_values('ds')  # Ensure the data is in chronological order

    return game_data

# Function to make predictions using the Prophet model
def make_predictions(model, game_data, forecast_days=5):
    future = pd.DataFrame({
        'ds': pd.date_range(start=game_data['ds'].max() + pd.Timedelta(days=1), periods=forecast_days)
    })

    # Make predictions
    future = future[['ds']]
    forecast = model.predict(future)
    forecast['yhat'] = np.maximum(0, forecast['yhat'])  # Ensure no negative predictions
    return forecast

# Initialize the model
model = load_model()

# -------- Interface 1: Team and Database Connection -------- #
if "current_page" not in st.session_state:
    st.session_state.current_page = "Interface 1"
if "save_count" not in st.session_state:
    st.session_state.save_count = 1

def change_page(new_page):
    st.session_state.current_page = new_page

if st.session_state.current_page == "Interface 1":
    st.title("Interface 1: Team and Database Connection")

    team_name = st.text_input("Team Name")
    user_input = st.text_input("User Input (e.g., LN69130_342409)")
    host = st.text_input("Host")
    port = st.text_input("Port")
    database = st.text_input("Database Name")
    user = st.text_input("User")
    password = st.text_input("Password")

    if st.button("Save Setting"):
        st.success(f"Team {team_name} with database connection settings saved successfully!")
        st.session_state.team_name = team_name
        st.session_state.user_input = user_input
        st.session_state.host = host
        st.session_state.port = port
        st.session_state.database = database
        st.session_state.user = user
        st.session_state.password = password
        change_page("Interface 2")

# -------- Interface 2: Prediction Input and Buy Decision -------- #
elif st.session_state.current_page == "Interface 2":
    st.title("Interface 2: Prediction Input and Buy Decision")

    if st.button("Show Prediction"):
        game_data = get_game_data(
            st.session_state.host,
            st.session_state.port,
            st.session_state.database,
            st.session_state.user,
            st.session_state.password,
            st.session_state.user_input
        )

        if game_data is not None and model is not None:
            forecast = make_predictions(model, game_data, forecast_days=5)

            # Combine past sales with future predictions for plotting
            full_data = pd.concat([game_data[['ds', 'y']], forecast[['ds', 'yhat']].rename(columns={'yhat': 'y'})])
            full_data.set_index('ds', inplace=True)

            # Plot actual vs predicted sales
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(full_data.index, full_data['y'], label='Units Sold', marker='o')
            ax.set_title(f"Actual vs Predicted Sales")
            ax.set_xlabel("Date")
            ax.set_ylabel("Units Sold")
            ax.legend()
            st.pyplot(fig)

    units_to_buy = st.number_input("จำนวนที่จะซื้อ", min_value=0, step=1)
    if st.button("Next"):
        change_page("Interface 3")

# -------- Interface 3: Discount Input -------- #
elif st.session_state.current_page == "Interface 3":
    st.title("Interface 3: Discount Input")

    discount_percent = st.number_input("Discount %", min_value=0.0, step=0.1)
    min_units_for_discount = st.number_input("จำนวนขั้นต่ำที่จะซื้อ", min_value=0, step=1)

    if st.button("Next"):
        change_page("Interface 4")

# -------- Interface 4: Final Decision and Survey -------- #
elif st.session_state.current_page == "Interface 4":
    st.title("Interface 4: Final Decision and Survey")

    final_units_to_buy = st.number_input("จำนวนสุดท้ายที่จะซื้อ", min_value=0, step=1)

    st.write("1. I relied on the AI suggestion in the game tasks")
    ai_reliance = st.radio("Strongly Disagree - Strongly Agree", [1, 2, 3, 4, 5, 6, 7], horizontal=True)

    st.write("2. Perceived Level of Agreement with AI Suggestion")
    ai_agreement = st.radio("Strongly Disagree - Strongly Agree", [1, 2, 3, 4, 5, 6, 7], horizontal=True)

    st.write("3. I trusted the AI suggestion in the game tasks")
    ai_trust = st.radio("Strongly Disagree - Strongly Agree", [1, 2, 3, 4, 5, 6, 7], horizontal=True)

    if st.button("Save Result"):
        result = {
            "team_name": st.session_state.team_name,
            "final_units_to_buy": final_units_to_buy,
            "ai_reliance": ai_reliance,
            "ai_agreement": ai_agreement,
            "ai_trust": ai_trust,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "save_count": st.session_state.save_count
        }

        datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{st.session_state.team_name}.{datetime_str}.{st.session_state.save_count}.json"
        file_content = json.dumps(result)

        st.session_state.save_count += 1
        st.success(f"Results saved as {filename}.")
        change_page("Interface 2")

