import os
import streamlit as st
import pickle
import numpy as np
from PIL import Image
import pandas as pd
from os.path import isfile, join
import re

# Set the page configuration of the app, including the page title, icon, and layout.
st.set_page_config(page_title="Timelytics", page_icon=":clock:", layout="wide")

col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    # Display image in the center
    img = Image.open("./assets/supply_chain_optimisation.jpg")
    st.image(img)
with col3:
    st.write(' ')

# Display the title and captions for the app.
st.title("Timelytics: Order to Delivery Time Prediction")
st.markdown("Timelytics is a powerful ensemble model designed to predict Order to Delivery (OTD) times with high accuracy. It integrates three machine learning algorithms—XGBoost, Random Forests, and Support Vector Machines (SVM)—to leverage their combined strengths. This fusion of techniques ensures a robust and dependable forecasting system, enabling businesses to enhance their supply chain efficiency.")
st.markdown("By utilizing Timelytics, companies can detect potential bottlenecks and anticipate delays in their supply chain processes. This predictive capability allows businesses to take proactive steps to mitigate disruptions, ultimately reducing lead times and improving overall delivery performance. With better visibility into supply chain operations, organizations can make data-driven decisions to streamline logistics.")
st.markdown("By utilizing Timelytics, companies can detect potential bottlenecks and anticipate delays in their supply chain processes. This predictive capability allows businesses to take proactive steps to mitigate disruptions, ultimately reducing lead times and improving overall delivery performance. With better visibility into supply chain operations, organizations can make data-driven decisions to streamline logistics.")


@st.experimental_memo
def get_data():
    return pd.read_csv("./final_orders.csv", index_col=0)


@st.experimental_memo
def get_data_with_encoding():
    return pd.read_csv("./final_orders_encoded_states_and_city.csv", index_col=0).rename(columns={
        'geolocation_state_customer': 'geolocation_state_customer_enc',
        'geolocation_state_seller': 'geolocation_state_seller_enc',
        'geolocation_city_customer': 'geolocation_city_customer_enc',
        'geolocation_city_seller': 'geolocation_city_seller_enc'
    })


@st.experimental_memo
def get_enc_list_map_and_inv(c):
    map_list = tuple(pd.concat([get_data()[c], get_data_with_encoding()[c+"_enc"]], axis=1).value_counts().index.to_list())
    name_to_int = {k: v for k, v in map_list}
    int_to_name = {v: k for k, v in map_list}
    return (tuple(k for k, v in map_list), name_to_int, int_to_name)


INT_TO_MONTH_MAP = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}
MONTH_TO_INT_MAP = {v: k for k, v in INT_TO_MONTH_MAP.items()}
MONTH_LIST = tuple(INT_TO_MONTH_MAP.values())

INT_TO_WEEK_MAP = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

WEEK_TO_INT_MAP = {v: i for i, v in enumerate(INT_TO_WEEK_MAP)}


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def pickle_join(source_dir, dest_file):
    # Create a new destination file
    output_file = open(dest_file, 'wb')
    parts = [f for f in os.listdir(source_dir) if isfile(join(source_dir, f))]
    sort_nicely(parts)
    print(parts)
    # Go through each portion one by one
    for file in parts:
        # Assemble the full path to the file
        path = source_dir + file
        # Open the part
        input_file = open(path, 'rb')
        while True:
            # Read all bytes of the part
            bytes = input_file.read()
            # Break out of loop if we are at end of file
            if not bytes:
                break
            # Write the bytes to the output file
            output_file.write(bytes)
        # Close the input file
        input_file.close()
    # Close the output file
    output_file.close()


@st.experimental_singleton  # Caching the model throughout the app for faster loading
def get_model():
    modelfile = "./voting_model.pkl"
    pickle_join(source_dir='./ModelFiles/', dest_file=modelfile)

    # Load the trained ensemble model from the saved pickle file.
    voting_model = pickle.load(open(modelfile, "rb"))
    return voting_model


@st.experimental_memo
# Define the function for the wait time predictor using the loaded model. This function takes in the input parameters and returns a predicted wait time in days.
def waitime_predictor(
    purchase_dow: str,
    purchase_month: str,
    year,
    product_size_cm3,
    product_weight_g,
    geolocation_state_customer: str,
    geolocation_state_seller: str,
    distance,
):
    _, state_customer_to_int, _ = get_enc_list_map_and_inv("geolocation_state_customer")
    _, state_seller_to_int, _ = get_enc_list_map_and_inv("geolocation_state_seller")

    prediction = get_model().predict(np.array([[
        WEEK_TO_INT_MAP[purchase_dow],
        MONTH_TO_INT_MAP[purchase_month],
        year,
        product_size_cm3,
        product_weight_g,
        state_customer_to_int[geolocation_state_customer],
        state_seller_to_int[geolocation_state_seller],
        distance,
    ]]))
    return round(prediction[0])


@st.experimental_memo
def get_sample_df():
    # Get a sample dataset for demonstration purposes.
    df = get_data().sample(100, random_state=43).copy()
    enc_df = get_data_with_encoding().loc[df.index]
    df["Actual Wait Time"] = df["wait_time"]
    df["predict"] = get_model().predict(enc_df[["purchase_dow", "purchase_month", "year", "product_size_cm3", "product_weight_g", "geolocation_state_customer_enc", "geolocation_state_seller_enc", "distance"]].to_numpy())
    df["purchase_dow"] = df["purchase_dow"].map(lambda v: INT_TO_WEEK_MAP[v])
    df["purchase_month"] = df["purchase_month"].map(lambda v: INT_TO_MONTH_MAP[v])
    df.drop(["geolocation_city_customer", "geolocation_city_seller", "wait_time", "est_wait_time", "delay"], axis=1, inplace=True)
    df.rename(columns={
        "purchase_dow": "Purchased Weekday",
        "purchase_month": "Purchased Month",
        "year": "Purchased Year",
        "product_size_cm3": "Product Size in cm^3",
        "product_weight_g": "Product Weight in grams",
        "geolocation_state_customer": "Geolocation State Customer",
        "geolocation_state_seller": "Geolocation State Seller",
        "distance": "Distance",
        "predict": "Predicted Wait Time",
        "price": "Price",
        "freight_value": "Freight Charge",
    }, inplace=True)
    return df


# Define the input parameters using Streamlit's sidebar. These parameters include the purchased day of the week, month, and year, product size, weight, geolocation state of the customer and seller, and distance.
with st.sidebar:
    st.header("Input Parameters")
    purchase_dow = st.selectbox("Purchased Day of the Week", options=INT_TO_WEEK_MAP, index=2)
    purchase_month = st.selectbox("Purchased Month", options=MONTH_LIST, index=4)
    year = st.number_input("Purchased Year", value=2017)
    product_size_cm3 = st.number_input("Product Size in cm^3", value=21200)
    product_weight_g = st.number_input("Product Weight in grams", value=1825)
    state_customer, _, _ = get_enc_list_map_and_inv("geolocation_state_customer")
    geolocation_state_customer = st.selectbox("Geolocation State of the Customer", options=state_customer, index=7)
    state_seller, _, _ = get_enc_list_map_and_inv("geolocation_state_seller")
    geolocation_state_seller = st.selectbox("Geolocation State of the Seller", options=state_seller, index=0)
    distance = st.number_input("Distance", value=845.76)
    submit = st.button("Submit")


# Define the submit button for the input parameters.
with st.container():
    # Define the output container for the predicted wait time.
    st.header("Output: Wait Time in Days: ")

    # When the submit button is clicked, call the wait time predictor function and display the predicted wait time in the output container.
    if submit:
        with st.spinner(text="This may take a moment..."):
            prediction = waitime_predictor(
                purchase_dow,
                purchase_month,
                year,
                product_size_cm3,
                product_weight_g,
                geolocation_state_customer,
                geolocation_state_seller,
                distance,
            )
            st.markdown(f"#### {prediction} days")

    # Display the sample dataset in the Streamlit app.
    st.header("Sample of 100 from the actual dataset")
    st.write(get_sample_df())
