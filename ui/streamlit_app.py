import streamlit as st
import requests
import pandas as pd
import joblib
import os

# Page config
st.set_page_config(
    page_title="India House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("India House Price Prediction")
st.markdown("### Predict residential property prices across India")

# Load model for direct prediction (alternative to API)
@st.cache_resource
def load_model():
    model_path = 'models/model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# Sidebar for inputs
st.sidebar.header("Property Details")

# Location
st.sidebar.subheader("Location")

# State to City mapping
state_city_map = {
    'Andhra Pradesh': ['Vijayawada', 'Vishakhapatnam'],
    'Assam': ['Guwahati', 'Silchar'],
    'Bihar': ['Gaya', 'Patna'],
    'Chhattisgarh': ['Bilaspur', 'Raipur'],
    'Delhi': ['Dwarka', 'New Delhi'],
    'Gujarat': ['Ahmedabad', 'Surat'],
    'Haryana': ['Faridabad', 'Gurgaon'],
    'Jharkhand': ['Jamshedpur', 'Ranchi'],
    'Karnataka': ['Bangalore', 'Mangalore', 'Mysore'],
    'Kerala': ['Kochi', 'Trivandrum'],
    'Madhya Pradesh': ['Bhopal', 'Indore'],
    'Maharashtra': ['Mumbai', 'Nagpur', 'Pune'],
    'Odisha': ['Bhubaneswar', 'Cuttack'],
    'Punjab': ['Amritsar', 'Ludhiana'],
    'Rajasthan': ['Jaipur', 'Jodhpur'],
    'Tamil Nadu': ['Chennai', 'Coimbatore'],
    'Telangana': ['Hyderabad', 'Warangal'],
    'Uttar Pradesh': ['Lucknow', 'Noida'],
    'Uttarakhand': ['Dehradun', 'Haridwar'],
    'West Bengal': ['Durgapur', 'Kolkata']
}

state = st.sidebar.selectbox(
    "State",
    list(state_city_map.keys())
)

city = st.sidebar.selectbox("City", state_city_map[state])

# Property Details
st.sidebar.subheader("Property Details")
property_type = st.sidebar.selectbox(
    "Property Type",
    ["Apartment", "Independent House", "Villa"]
)

bhk = st.sidebar.slider("BHK", 1, 5, 2)
size = st.sidebar.number_input("Size (sq ft)", 500, 5000, 1500, 100)
price_per_sqft = st.sidebar.number_input(
    "Est. Price per Sq Ft",
    0.01, 1.0, 0.08, 0.01,
    help="Estimated market rate per square foot in your area"
)

# Building Details
st.sidebar.subheader("Building Details")
year_built = st.sidebar.slider("Year Built", 1990, 2025, 2015)
floor_no = st.sidebar.slider("Floor Number", 0, 30, 5)
total_floors = st.sidebar.slider("Total Floors", 1, 30, 10)

# Furnishing & Amenities
st.sidebar.subheader("Furnishing & Amenities")
furnished_status = st.sidebar.selectbox(
    "Furnished Status",
    ["Furnished", "Semi-furnished", "Unfurnished"]
)
parking = st.sidebar.checkbox("Parking Available", value=True)
security = st.sidebar.checkbox("Security", value=True)
facing = st.sidebar.selectbox("Facing", ["East", "North", "South", "West"])

# Surroundings
st.sidebar.subheader("Surroundings")
nearby_schools = st.sidebar.slider("Nearby Schools", 1, 10, 5)
nearby_hospitals = st.sidebar.slider("Nearby Hospitals", 1, 10, 3)
transport = st.sidebar.selectbox(
    "Public Transport",
    ["High", "Medium", "Low"]
)

# Availability
owner_type = st.sidebar.selectbox("Owner Type", ["Owner", "Builder", "Broker"])
availability = st.sidebar.selectbox(
    "Availability",
    ["Ready_to_Move", "Under_Construction"]
)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Property Summary")
    
    summary_data = {
        "Feature": [
            "Location", "Property Type", "Size", "BHK", "Year Built",
            "Floor", "Furnished Status", "Parking", "Security"
        ],
        "Value": [
            f"{city}, {state}",
            property_type,
            f"{size} sq ft",
            f"{bhk} BHK",
            str(year_built),
            f"{floor_no}/{total_floors}",
            furnished_status,
            "Yes" if parking else "No",
            "Yes" if security else "No"
        ]
    }
    
    st.table(pd.DataFrame(summary_data))

with col2:
    st.subheader("Price Prediction")
    
    if st.button("Predict Price", type="primary", use_container_width=True):
        if model is not None:
            # Create input data
            state_map = {'Andhra Pradesh': 0, 'Assam': 1, 'Bihar': 2, 'Chhattisgarh': 3, 
                        'Delhi': 4, 'Gujarat': 5, 'Haryana': 6, 'Jharkhand': 7, 
                        'Karnataka': 8, 'Kerala': 9, 'Madhya Pradesh': 10, 'Maharashtra': 11, 
                        'Odisha': 12, 'Punjab': 13, 'Rajasthan': 14, 'Tamil Nadu': 15, 
                        'Telangana': 16, 'Uttar Pradesh': 17, 'Uttarakhand': 18, 'West Bengal': 19}
            city_map = {'Ahmedabad': 0, 'Amritsar': 1, 'Bangalore': 2, 'Bhopal': 3, 'Bhubaneswar': 4,
                       'Bilaspur': 5, 'Chennai': 6, 'Coimbatore': 7, 'Cuttack': 8, 'Dehradun': 9,
                       'Durgapur': 10, 'Dwarka': 11, 'Faridabad': 12, 'Gaya': 13, 'Gurgaon': 14,
                       'Guwahati': 15, 'Haridwar': 16, 'Hyderabad': 17, 'Indore': 18, 'Jaipur': 19,
                       'Jamshedpur': 20, 'Jodhpur': 21, 'Kochi': 22, 'Kolkata': 23, 'Lucknow': 24,
                       'Ludhiana': 25, 'Mangalore': 26, 'Mumbai': 27, 'Mysore': 28, 'Nagpur': 29,
                       'New Delhi': 30, 'Noida': 31, 'Patna': 32, 'Pune': 33, 'Raipur': 34,
                       'Ranchi': 35, 'Silchar': 36, 'Surat': 37, 'Trivandrum': 38, 'Vijayawada': 39,
                       'Vishakhapatnam': 40, 'Warangal': 41}
            property_map = {'Apartment': 0, 'Independent House': 1, 'Villa': 2}
            furnished_map = {'Furnished': 0, 'Semi-furnished': 1, 'Unfurnished': 2}
            facing_map = {'East': 0, 'North': 1, 'South': 2, 'West': 3}
            owner_map = {'Owner': 0, 'Builder': 1, 'Broker': 2}
            availability_map = {'Ready_to_Move': 0, 'Under_Construction': 1}
            transport_map = {'High': 0, 'Low': 1, 'Medium': 2}
            
            input_data = {
                'State': state_map.get(state, 0),
                'City': city_map.get(city, 0),
                'Property_Type': property_map.get(property_type, 0),
                'BHK': bhk,
                'Size_in_SqFt': size,
                'Price_per_SqFt': price_per_sqft,
                'Year_Built': year_built,
                'Furnished_Status': furnished_map.get(furnished_status, 1),
                'Floor_No': floor_no,
                'Total_Floors': total_floors,
                'Age_of_Property': 2025 - year_built,
                'Nearby_Schools': nearby_schools,
                'Nearby_Hospitals': nearby_hospitals,
                'Public_Transport_Accessibility': transport_map.get(transport, 2),
                'Parking_Space': 1 if parking else 0,
                'Security': 1 if security else 0,
                'Facing': facing_map.get(facing, 0),
                'Owner_Type': owner_map.get(owner_type, 0),
                'Availability_Status': availability_map.get(availability, 0),
                'Room_Size_Ratio': size / bhk,
                'Floor_Position': floor_no / (total_floors + 1)
            }
            
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            
            # Display prediction
            st.success("Prediction Complete")
            st.metric(
                label="Estimated Price",
                value=f"Rs {prediction:.2f} Lakhs",
                delta=f"Rs {prediction * 100000:,.0f}"
            )
            
            # Price breakdown
            st.info(f"Price per sq ft: Rs {(prediction * 100000) / size:,.0f}")
            
        else:
            st.error("Model not found. Please train the model first.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>India House Price Prediction System | Built with Streamlit & scikit-learn</p>
    </div>
    """,
    unsafe_allow_html=True
)