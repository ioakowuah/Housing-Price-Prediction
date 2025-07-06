import joblib
import streamlit as st
import numpy as np
import requests
from io import BytesIO

# Load trained model
def load_model():
    url = 'https://huggingface.co/datasets/ioakowuah/HousingPricePrediction/resolve/main/XGBRegressor.pkl'
    response = requests.get(url)
    model = joblib.load(BytesIO(response.content))
    return model

model = load_model()


# Feature mappings
OverallQual_map = {
    'Very Poor': 1, 'Poor': 2, 'Fair': 3, 'Below Average': 4, 'Average': 5,
    'Above Average': 6, 'Good': 7, 'Very Good': 8, 'Excellent': 9, 'Very Excellent': 10
}
KitchenQual_map = {'Excellent': 0, 'Fair': 1, 'Good': 2, 'Average': 3}
ExterQual_map = {'Excellent': 0, 'Fair': 1, 'Good': 2, 'Average': 3}
GarageType_map = {
    'More than one type of garage': 0, 'Attached to home': 1, 'Basement Garage': 2,
    'Built-In (Garage part of house - typically has room above garage)': 3,
    'Car Port': 4, 'Detached from home': 5
}
BsmtQual_map = {'Excellent': 0, 'Fair': 1, 'Good': 2, 'Average': 3}

def main():
    st.title("🏡 House Price Estimator")
    st.markdown("Fill in the house features below and click **Predict** to estimate the price.")

    # User inputs
    FullBath = st.number_input('🚿 Number of full bathrooms above grade', min_value=0)
    OverallQual = st.selectbox('🏗️ Overall material and finish quality', list(OverallQual_map.keys()))
    KitchenQual = st.selectbox('👨‍🍳 Kitchen quality', list(KitchenQual_map.keys()))
    ExterQual = st.selectbox('🏘️ Exterior quality', list(ExterQual_map.keys()))
    GarageType = st.selectbox('🚗 Garage location/type', list(GarageType_map.keys()))
    SdFlrSF = st.number_input('📐 Second floor square feet', min_value=0)
    BsmtQual = st.selectbox('🏚️ Basement quality and height', list(BsmtQual_map.keys()))
    TotRmsAbvGrd = st.number_input('🛏️ Total rooms above grade (excluding bathrooms)', min_value=2, max_value=20)
    GrLivArea = st.number_input('📏 Above-grade living area (sqft)', min_value=334, max_value=5642)
    TotalBsmtSF = st.number_input('🏠 Basement area (sqft)', min_value=0, max_value=6110)

    # Encode inputs
    encoded_input = [
        FullBath,
        OverallQual_map[OverallQual],
        KitchenQual_map[KitchenQual],
        ExterQual_map[ExterQual],
        GarageType_map[GarageType],
        SdFlrSF,
        BsmtQual_map[BsmtQual],
        TotRmsAbvGrd,
        GrLivArea,
        TotalBsmtSF
    ]

    # Predict button
    if st.button("💰 Predict House Price"):
        try:
            prediction = model.predict([encoded_input])
            price = prediction[0]
            st.success(f"🏷️ Estimated House Price: ${price:,.2f}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# Launch the app
if __name__ == '__main__':
    main()
