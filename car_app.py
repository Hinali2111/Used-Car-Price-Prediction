import streamlit as st
import pandas as pd
import joblib
import traceback



# Page configuration
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    h1 {color: #e74c3c; padding-bottom: 1rem;}
    </style>
    """, unsafe_allow_html=True)


# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('used_car_price_prediction.pkl')
        return model
    except FileNotFoundError:
        return None


# Header
st.title("🚗 Used Car Price Prediction System")
st.markdown("### Get Instant Valuation for Your Used Car")

# Load model
model = load_model()
try:
    label_encoders = joblib.load("label_encoders.pkl")
    st.write(label_encoders.keys())
except FileNotFoundError:
    st.error("label_encoders.pkl not found.")
    st.stop()

if model is None:
    st.error("**Model file not found!**")
    st.info("""
    Please run the following command first:
    ```
    python used_car_price_prediction.py
    ```
    This will train and save the model.
    """)
    st.stop()
@st.cache_data
def load_data():
    return pd.read_csv("used_car_dataset.csv")   

df = load_data()
st.write(df.columns.tolist())

# Sidebar inputs
st.sidebar.title("🚗 Car Details")

brand = st.sidebar.selectbox(
    "Brand",
    sorted(df["Brand"].unique())
)

model_name = st.sidebar.selectbox(
    "Model",
    sorted(df[df["Brand"] == brand]["model"].unique())
)

year = st.sidebar.number_input(
    "Manufacturing Year",
    min_value=int(df["Year"].min()),
    max_value=int(df["Year"].max()),
    value=2020
)

km_Driven = st.sidebar.number_input(
    "Kilometers Driven",
    min_value=0,
    value=50000,
    step=1000
)

transmission = st.sidebar.selectbox(
    "Transmission",
    sorted(df["Transmission"].unique())
)

owner = st.sidebar.selectbox(
    "Owner",
    sorted(df["Owner"].unique())
)

FuelType = st.sidebar.selectbox(
    "FuelType Type",
    sorted(df["FuelType"].unique())
)

# Predict button
st.sidebar.markdown("---")
predict_btn = st.sidebar.button("Get Price Estimate", type="primary", use_container_width=True)

# Main content
if predict_btn:

    input_data = pd.DataFrame({
        "Brand": [brand],
        "model": [model_name],
        "Year": [year],
        "km_Driven": [km_Driven],
        "Transmission": [transmission],
        "Owner": [owner],
        "FuelType": [FuelType]
    })
   
    try:
    # Encode categorical columns
        for col in ["Brand", "model", "Transmission", "Owner", "FuelType"]:
            input_data[col] = label_encoders[col].transform(input_data[col])

    

    # Prediction
        predicted_price = model.predict(input_data)[0]

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
   
    
    # Display results
    st.markdown("---")
    st.header("Price Estimation Results")
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("### 💰 Predicted Used Car Price")

        st.metric(
            label="Estimated Price",
            value=f"₹ {predicted_price:,.0f}"
        )
    
    # Gauge chart for price range
    st.markdown("---")
    st.subheader("Price Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
       
        
        # Price breakdown
        st.write("**Price Factors:**")
        
        factors = []
        
        
        if km_Driven < 30000:
            factors.append("Low mileage - adds value")
        elif km_Driven < 80000:
            factors.append("Average mileage")
        else:
            factors.append("High mileage - reduces value")
        
        if transmission == 'Automatic':
            factors.append("Automatic transmission - premium pricing")
        
        if FuelType == 'Diesel':
            factors.append("Diesel - preferred for high usage")
        elif FuelType == 'Petrol':
            factors.append("Petrol - standard option")
        
    
        
        for factor in factors:
            st.markdown(f"- {factor}")
    
    
       

    # Car details summary
    st.markdown("---")
    st.subheader("Your Car Details")
    
    details_col1, details_col2 = st.columns(2)
    
    with details_col1:
        st.write(f"**Brand:** {brand}")
        st.write(f"**Model:** {model_name}")
        st.write(f"**Manufacturing Year:** {year}")
        st.write(f"**Kilometers Driven:** {km_Driven:,} km")

    with details_col2:
        st.write(f"**FuelType:** {FuelType}")
        st.write(f"**Transmission:** {transmission}")
        st.write(f"**Owner:** {owner}")
       
    
    # Tips for selling
    st.markdown("---")
    st.subheader("Tips to Get Better Price")
    
    
else:
    # Initial page
    st.markdown("---")
    st.info("Enter your car details in the sidebar and click **Get Price Estimate**")
    
    # Show example cars
    st.subheader("Example Valuations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Honda City**")
        st.write("Year: 2018")
        st.write("KM: 45,000")
        st.write("Estimated Price: ₹6,50,000")

    with col2:
        st.write("**Hyundai Creta**")
        st.write("Year: 2020")
        st.write("KM: 28,000")
        st.write("Estimated Price: ₹10,20,000")

    with col3:
        st.write("**Maruti Swift**")
        st.write("Year: 2016")
        st.write("KM: 72,000")
        st.write("Estimated Price: ₹4,80,000")
    
    st.markdown("---")
    
    # Model info
    st.subheader("Model Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Algorithm", "Random Forest")
    col2.metric("R² Score", "0.305")
    col3.metric("Dataset", "1168 cars")
