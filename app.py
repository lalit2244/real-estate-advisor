# Real Estate Investment Advisor - Streamlit Application
# Save this as: 03_Streamlit_App.py
# Run with: streamlit run 03_Streamlit_App.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD MODELS AND DATA
# ============================================
@st.cache_resource
def load_models():
    """Load trained models and preprocessing objects"""
    try:
        with open('models/classifier_model.pkl', 'rb') as f:
            classifier = pickle.load(f)
        with open('models/regressor_model.pkl', 'rb') as f:
            regressor = pickle.load(f)
        with open('models/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        return classifier, regressor, encoders, scaler, feature_cols
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

@st.cache_data
def load_data():
    """Load processed dataset"""
    try:
        df = pd.read_csv('data/processed_housing_data.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load everything
classifier, regressor, label_encoders, scaler, feature_columns = load_models()
df = load_data()

# ============================================
# HELPER FUNCTIONS
# ============================================
def prepare_input(input_data, encoders, feature_cols):
    """Prepare user input for model prediction"""
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical features
    for col, encoder in encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = encoder.transform(input_df[col].astype(str))
            except:
                input_df[col] = 0  # Default value for unseen categories
    
    # Ensure all feature columns are present
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns
    input_df = input_df[feature_cols]
    
    return input_df

def get_investment_recommendation(probability, future_price, current_price):
    """Generate investment recommendation"""
    appreciation = ((future_price - current_price) / current_price) * 100
    
    if probability >= 0.7 and appreciation >= 30:
        return "🟢 Excellent Investment", "This property shows strong investment potential with high returns expected."
    elif probability >= 0.5 and appreciation >= 20:
        return "🟡 Good Investment", "This property is a decent investment with moderate returns."
    elif probability >= 0.3:
        return "🟠 Fair Investment", "This property may yield average returns. Consider other options."
    else:
        return "🔴 Not Recommended", "This property shows weak investment indicators. Look for better opportunities."

# ============================================
# MAIN APP
# ============================================
def main():
    # Title and Description
    st.title("🏠 Real Estate Investment Advisor")
    st.markdown("### Predict Property Profitability & Future Value")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["🔮 Prediction", "📊 Market Insights", "🗺️ Data Explorer"])
    
    if page == "🔮 Prediction":
        prediction_page()
    elif page == "📊 Market Insights":
        insights_page()
    elif page == "🗺️ Data Explorer":
        explorer_page()

# ============================================
# PAGE 1: PREDICTION
# ============================================
def prediction_page():
    st.header("🔮 Property Investment Prediction")
    
    if classifier is None or regressor is None:
        st.error("⚠️ Models not loaded. Please ensure all model files are present.")
        return
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Location Details")
        
        state = st.selectbox("State", options=sorted(df['State'].unique()) if df is not None else [])
        city_options = sorted(df[df['State'] == state]['City'].unique()) if df is not None and state else []
        city = st.selectbox("City", options=city_options)
        
        property_type = st.selectbox("Property Type", 
                                      options=['Apartment', 'Villa', 'House', 'Studio', 'Penthouse'])
        
        st.subheader("🏗️ Property Specifications")
        bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=2)
        size_sqft = st.number_input("Size (Sq Ft)", min_value=100, max_value=10000, value=1000)
        price = st.number_input("Current Price (Lakhs)", min_value=1.0, max_value=1000.0, value=50.0)
        
        year_built = st.number_input("Year Built", min_value=1950, max_value=2024, value=2015)
        
    with col2:
        st.subheader("🛋️ Property Features")
        
        furnished_status = st.selectbox("Furnished Status", 
                                         options=['Unfurnished', 'Semi-Furnished', 'Fully-Furnished'])
        
        floor_no = st.number_input("Floor Number", min_value=0, max_value=50, value=2)
        total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=10)
        
        parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)
        
        st.subheader("🏙️ Nearby Amenities")
        
        schools = st.slider("Nearby Schools", 0, 20, 5)
        hospitals = st.slider("Nearby Hospitals", 0, 10, 3)
        transport = st.slider("Public Transport Access (1-10)", 1, 10, 7)
        
        facing = st.selectbox("Property Facing", 
                              options=['North', 'South', 'East', 'West', 'North-East', 
                                       'North-West', 'South-East', 'South-West'])
    
    # Prediction button
    st.markdown("---")
    col_button = st.columns([1, 2, 1])
    with col_button[1]:
        predict_button = st.button("🔍 Analyze Investment Potential", type="primary")
    
    if predict_button:
        with st.spinner("Analyzing property..."):
            # Calculate derived features
            price_per_sqft = (price * 100000) / size_sqft
            age = 2024 - year_built
            infra_score = ((schools/20) + (hospitals/10) + (transport/10)) / 3 * 100
            
            # Prepare input
            input_data = {
                'State': state,
                'City': city,
                'Property_Type': property_type,
                'BHK': bhk,
                'Size_in_SqFt': size_sqft,
                'Price_in_Lakhs': price,
                'Price_per_SqFt': price_per_sqft,
                'Year_Built': year_built,
                'Furnished_Status': furnished_status,
                'Floor_No': floor_no,
                'Total_Floors': total_floors,
                'Age_of_Property': age,
                'Nearby_Schools': schools,
                'Nearby_Hospitals': hospitals,
                'Public_Transport_Accessibility': transport,
                'Parking_Space': parking,
                'Facing': facing,
                'Infrastructure_Score': infra_score
            }
            
            # Prepare for model
            input_df = prepare_input(input_data, label_encoders, feature_columns)
            
            # Make predictions
            classification_prob = classifier.predict_proba(input_df)[0][1]
            classification_result = classifier.predict(input_df)[0]
            future_price = regressor.predict(input_df)[0]
            
            # Get recommendation
            recommendation, explanation = get_investment_recommendation(
                classification_prob, future_price, price
            )
            
            # Display results
            st.markdown("---")
            st.header("📊 Analysis Results")
            
            # Main metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    "Investment Rating",
                    "Good Investment" if classification_result == 1 else "Not Recommended",
                    f"{classification_prob*100:.1f}% Confidence"
                )
            
            with metric_col2:
                appreciation = ((future_price - price) / price) * 100
                st.metric(
                    "5-Year Price Forecast",
                    f"₹{future_price:.2f} L",
                    f"+{appreciation:.1f}%"
                )
            
            with metric_col3:
                annual_return = (appreciation / 5)
                st.metric(
                    "Expected Annual Return",
                    f"{annual_return:.1f}%",
                    "Per Year"
                )
            
            # Recommendation
            st.markdown("---")
            st.subheader("💡 Investment Recommendation")
            st.info(f"**{recommendation}**\n\n{explanation}")
            
            # Detailed breakdown
            st.markdown("---")
            st.subheader("📈 Detailed Analysis")
            
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown("**Property Strengths:**")
                strengths = []
                if classification_prob > 0.6:
                    strengths.append("✅ High investment probability")
                if infra_score > 60:
                    strengths.append("✅ Good infrastructure access")
                if age < 10:
                    strengths.append("✅ Relatively new property")
                if bhk in [2, 3]:
                    strengths.append("✅ High-demand BHK configuration")
                
                for strength in strengths:
                    st.markdown(strength)
            
            with detail_col2:
                st.markdown("**Investment Metrics:**")
                st.markdown(f"• Current Price/SqFt: ₹{price_per_sqft:.2f}")
                st.markdown(f"• Property Age: {age} years")
                st.markdown(f"• Infrastructure Score: {infra_score:.1f}/100")
                st.markdown(f"• Expected 5Y Value: ₹{future_price:.2f} L")
            
            # Visualization
            st.markdown("---")
            st.subheader("📊 Price Projection")
            
            years = list(range(0, 6))
            projected_prices = [price * ((1 + (appreciation/100)/5) ** year) for year in years]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years,
                y=projected_prices,
                mode='lines+markers',
                name='Projected Price',
                line=dict(color='#00CC96', width=3),
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                title="5-Year Price Projection",
                xaxis_title="Years from Now",
                yaxis_title="Price (Lakhs ₹)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, key="price_projection")

# ============================================
# PAGE 2: MARKET INSIGHTS
# ============================================
def insights_page():
    st.header("📊 Real Estate Market Insights")
    
    if df is None:
        st.error("Data not loaded")
        return
    
    # Key Metrics
    st.subheader("🔑 Key Market Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", f"{len(df):,}")
    with col2:
        st.metric("Avg Price", f"₹{df['Price_in_Lakhs'].mean():.2f} L")
    with col3:
        st.metric("Avg Size", f"{df['Size_in_SqFt'].mean():.0f} sqft")
    with col4:
        good_inv_pct = df['Good_Investment'].mean() * 100
        st.metric("Good Investments", f"{good_inv_pct:.1f}%")
    
    st.markdown("---")
    
    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("💰 Price Distribution by City")
        top_cities = df.groupby('City')['Price_in_Lakhs'].mean().nlargest(10)
        fig = px.bar(
            x=top_cities.index,
            y=top_cities.values,
            labels={'x': 'City', 'y': 'Average Price (Lakhs)'},
            color=top_cities.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, key="city_price")
    
    with chart_col2:
        st.subheader("🏘️ Property Type Distribution")
        prop_counts = df['Property_Type'].value_counts()
        fig = px.pie(
            values=prop_counts.values,
            names=prop_counts.index,
            hole=0.4
        )
        st.plotly_chart(fig, key="property_type")
    
    st.markdown("---")
    
    # Additional insights
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.subheader("📈 Investment Quality by BHK")
        bhk_investment = df.groupby('BHK', observed=False)['Good_Investment'].mean() * 100
        fig = px.bar(
            x=bhk_investment.index,
            y=bhk_investment.values,
            labels={'x': 'BHK', 'y': 'Good Investment %'},
            color=bhk_investment.values,
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, key="bhk_investment")
    
    with insight_col2:
        st.subheader("🏗️ Age vs Investment Quality")
        age_bins = pd.cut(df['Age_of_Property'], bins=5)
        age_investment = df.groupby(age_bins, observed=False)['Good_Investment'].mean() * 100
        fig = px.line(
            x=[str(x) for x in age_investment.index],
            y=age_investment.values,
            labels={'x': 'Property Age Range', 'y': 'Good Investment %'},
            markers=True
        )
        st.plotly_chart(fig, key="age_investment")

# ============================================
# PAGE 3: DATA EXPLORER
# ============================================
def explorer_page():
    st.header("🗺️ Property Data Explorer")
    
    if df is None:
        st.error("Data not loaded")
        return
    
    st.subheader("🔍 Filter Properties")
    
    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        state_filter = st.multiselect("State", options=sorted(df['State'].unique()))
        bhk_filter = st.multiselect("BHK", options=sorted(df['BHK'].unique()))
    
    with filter_col2:
        price_range = st.slider("Price Range (Lakhs)", 
                                 float(df['Price_in_Lakhs'].min()),
                                 float(df['Price_in_Lakhs'].max()),
                                 (float(df['Price_in_Lakhs'].min()), 
                                  float(df['Price_in_Lakhs'].max())))
    
    with filter_col3:
        size_range = st.slider("Size Range (SqFt)",
                                int(df['Size_in_SqFt'].min()),
                                int(df['Size_in_SqFt'].max()),
                                (int(df['Size_in_SqFt'].min()),
                                 int(df['Size_in_SqFt'].max())))
    
    # Apply filters
    filtered_df = df.copy()
    
    if state_filter:
        filtered_df = filtered_df[filtered_df['State'].isin(state_filter)]
    if bhk_filter:
        filtered_df = filtered_df[filtered_df['BHK'].isin(bhk_filter)]
    
    filtered_df = filtered_df[
        (filtered_df['Price_in_Lakhs'] >= price_range[0]) &
        (filtered_df['Price_in_Lakhs'] <= price_range[1]) &
        (filtered_df['Size_in_SqFt'] >= size_range[0]) &
        (filtered_df['Size_in_SqFt'] <= size_range[1])
    ]
    
    st.info(f"📋 Showing {len(filtered_df)} properties")
    
    # Display data
    display_cols = ['City', 'Property_Type', 'BHK', 'Size_in_SqFt', 
                    'Price_in_Lakhs', 'Price_per_SqFt', 'Age_of_Property',
                    'Good_Investment', 'Future_Price_5Y']
    
    st.dataframe(
        filtered_df[display_cols].head(100),
        height=400
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data",
        data=csv,
        file_name="filtered_properties.csv",
        mime="text/csv"
    )

# ============================================
# RUN APP
# ============================================
if __name__ == "__main__":
    main()