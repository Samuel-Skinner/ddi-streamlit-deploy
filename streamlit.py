import streamlit as st
import pandas as pd
import joblib

st.title("DDI Streamlit Assignment")
st.write("This is the streamlit app for the DDI Streamlit assignment. It predicts mpg based off other features on a car.")

cars_df = pd.read_csv("data/cars.csv")
st.dataframe(cars_df)

cylinders = st.slider("Select the number of cylinders in the car", min_value=1, max_value=8)
displacement = st.slider("Select displacement of the car", min_value=50, max_value=500)
horsepower = st.slider("Select horsepower of the car", min_value=30, max_value=300)
weight = st.slider("Select the weight of the car", min_value=1500, max_value=5500)
acceleration = st.slider("Select the acceleration of the car", min_value=5, max_value=30)
origin = st.selectbox("Select car origin", options=[1, 2, 3], format_func=lambda x: ["USA", "Europe", "Japan"][x-1])

car_df = pd.DataFrame({
    "cylinders": [cylinders],
    "displacement": [displacement],
    "horsepower": [horsepower],
    "weight": [weight],
    "acceleration": [acceleration],
    "origin": [origin]
})
chosen = st.selectbox('Choose your model', ['Linear Regression', 'Random Forest'])

st.dataframe(car_df)
if chosen == 'Linear Regression':
    data = joblib.load("data/cars_mpg_predictor.joblib")
elif chosen == 'Random Forest':
    data = joblib.load('data/cars_mpg_forest.joblib')
model = data["model"]
mpg_prediction = model.predict(car_df)

st.write("Predicted MPG:", mpg_prediction)

avg_mpg = cars_df["mpg"].mean()

compare_df = pd.DataFrame({
    "Type": ["Predicted MPG", "Average MPG"],
    "MPG": [mpg_prediction[0], avg_mpg]
})

st.bar_chart(compare_df.set_index("Type"))

st.bar_chart(data["feature importance"])