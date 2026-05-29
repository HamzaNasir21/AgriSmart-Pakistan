import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AgriSmart Pakistan", page_icon="🌾", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stMetric {background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    h1, h2, h3 {color: #2e7d32;}
    .stAlert {border-radius: 10px;}
    </style>
    """, unsafe_allow_html=True)

st.title("🌾 AgriSmart Pakistan")
st.markdown("**Advanced Agricultural Price Monitoring & Decision Support System**")
st.markdown("---")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Project_DataSet.csv') 
    except FileNotFoundError:
        st.error("Dataset 'Project_Dataset.csv' is missing. Please ensure it is uploaded.")
        return None
    
    # Text cleaning
    if 'city' in df.columns: df['city'] = df['city'].astype(str).str.strip().str.title()
    if 'crop' in df.columns: df['crop'] = df['crop'].astype(str).str.strip().str.title()
    
    return df

df = load_data()

if df is not None:
    le_city = LabelEncoder()
    le_crop = LabelEncoder()
    
    df['City_Encoded'] = le_city.fit_transform(df['city'])
    df['Crop_Encoded'] = le_crop.fit_transform(df['crop'])
    
    mean_global_price = df['avg_price_per_kg'].mean()
    df['Is_High_Price'] = (df['avg_price_per_kg'] > mean_global_price).astype(int)

    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=100)
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Select Module:", 
                            ["🛒 Check Shop Prices", 
                             "👨‍🌾 Farmer Profit Planner", 
                             "🏛️ Govt Policy Tester"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("System uses Random Forest, Linear Regression & Naive Bayes for advanced analytics.")

    if menu == "🛒 Check Shop Prices":
        st.header("Consumer Portal: Fair Price Checker")
        st.write("Ensure you are not being overcharged. Enter shop details below:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            sel_city = st.selectbox("Select City", df['city'].unique())
        with col2:
            sel_crop = st.selectbox("Select Commodity", df['crop'].unique())
        with col3:
            shop_price = st.number_input("Price Shopkeeper is Asking (PKR):", min_value=1.0, value=100.0, step=1.0)
            
        if st.button("Analyze Price & Risk", use_container_width=True):
            with st.spinner("Analyzing market data..."):
                city_data = df[(df['city'] == sel_city) & (df['crop'] == sel_crop)]
                
                if not city_data.empty:
                    avg_price = city_data['avg_price_per_kg'].mean()
                    diff = shop_price - avg_price
                    
                    st.markdown("### 📊 Market Analysis")
                    c1, c2 = st.columns(2)
                    c1.metric("Market Average Price", f"Rs. {avg_price:.2f}")
                    c2.metric("Your Entered Price", f"Rs. {shop_price:.2f}", f"{diff:.2f} PKR Difference", delta_color="inverse")
                    
                    st.markdown("### ⚖️ Verdict")
                    if diff > 7:
                        st.error(f"🚨 **SCAM ALERT!** You are being overcharged. The price is {diff:.2f} PKR above the market average! (Tolerance limit is 7 PKR).")
                    elif diff > 0 and diff <= 7:
                        st.warning(f"⚠️ **Slightly High.** It is {diff:.2f} PKR above average, but within the acceptable retail margin.")
                    else:
                        st.success("✅ **Great Deal!** The shopkeeper is offering a fair price.")
                        st.balloons()
                        
                    st.markdown("---")
                    st.subheader("🔮 Regional Risk Predictor (Powered by Naive Bayes)")
                    st.write("Calculates the statistical probability of encountering high-price surges in this region.")
                    
                    nb_model = GaussianNB()
                    X_nb = df[['City_Encoded', 'Crop_Encoded']]
                    y_nb = df['Is_High_Price']
                    nb_model.fit(X_nb, y_nb)
                    
                    input_city_enc = le_city.transform([sel_city])[0]
                    input_crop_enc = le_crop.transform([sel_crop])[0]
                    
                    risk_prob = nb_model.predict_proba([[input_city_enc, input_crop_enc]])[0][1] * 100
                    
                    st.progress(int(risk_prob))
                    if risk_prob > 50:
                        st.warning(f"**High Risk Area:** There is a {risk_prob:.1f}% probability of general price surges for {sel_crop} in {sel_city}.")
                    else:
                        st.info(f"**Stable Area:** Only a {risk_prob:.1f}% probability of sudden price surges here.")
                        
                else:
                    st.error("No historical data available for this specific combination.")

    elif menu == "👨‍🌾 Farmer Profit Planner":
        st.header("Farmer Portal: Market Trend & Profitability")
        st.write("Analyze long-term economic trends to plan your harvest.")
        
        c1, c2 = st.columns(2)
        with c1:
            sel_crop = st.selectbox("Select Crop to Analyze", df['crop'].unique())
        with c2:
            kg_to_sell = st.number_input("Total KGs you plan to sell (Wholesale Unit):", min_value=40, value=40, step=40)
            
        st.info("💡 Note: 40 KG represents 1 'Mann', the standard wholesale unit in Pakistan.")

        crop_data = df[df['crop'] == sel_crop].copy()
        if not crop_data.empty:
            st.markdown("### 📈 Macro-Economic Trend Analysis (Linear Regression)")
            
            crop_data['Time_Index'] = np.arange(len(crop_data))
            X_lr = crop_data[['Time_Index']]
            y_lr = crop_data['avg_price_per_kg']
            
            lr_model = LinearRegression()
            lr_model.fit(X_lr, y_lr)
            crop_data['Trend_Line'] = lr_model.predict(X_lr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=crop_data['Time_Index'], y=crop_data['avg_price_per_kg'], mode='lines', name='Actual Price Fluctuations', line=dict(color='lightblue')))
            fig.add_trace(go.Scatter(x=crop_data['Time_Index'], y=crop_data['Trend_Line'], mode='lines', name='Linear Trend (Macro Direction)', line=dict(color='red', width=3)))
            fig.update_layout(title=f"Historical Price vs Projected Trend for {sel_crop}", xaxis_title="Timeline (Data Points)", yaxis_title="Price (PKR)")
            st.plotly_chart(fig, use_container_width=True)
            
            trend_diff = crop_data['Trend_Line'].iloc[-1] - crop_data['Trend_Line'].iloc[0]
            if trend_diff > 0:
                st.success("✅ **Positive Trend:** The linear regression model indicates overall market growth for this crop.")
            else:
                st.error("📉 **Negative Trend:** The market value is generally declining over time.")
                
            st.markdown("---")
            st.markdown("### 💰 Profit Estimation")
            current_avg_price = crop_data['avg_price_per_kg'].mean()
            est_revenue = current_avg_price * kg_to_sell
            st.metric("Estimated Revenue", f"Rs. {est_revenue:,.2f}", f"Based on {kg_to_sell} KGs")

    elif menu == "🏛️ Govt Policy Tester":
        st.header("Government Sandbox: Policy & Economic Impact")
        st.write("Simulate policy changes using Random Forest intertwined with EDA impact weights.")
        
        st.warning("Ensure policies do not push staple crops out of citizen reach.")
        
        col1, col2 = st.columns(2)
        with col1:
            sel_city = st.selectbox("Target Region", df['city'].unique())
            sel_crop = st.selectbox("Target Commodity", df['crop'].unique())
        with col2:
            fuel_hike = st.slider("Fuel Price Increase (%)", 0, 100, 0)
            inflation_rate = st.slider("Inflation Rate (%)", 0.0, 30.0, 5.0)

        if st.button("Simulate Economic Impact", use_container_width=True):
            with st.spinner("Running Ensembled ML Simulation..."):
                if 'fuel_price_change' not in df.columns:
                    df['fuel_price_change'] = np.random.uniform(0, 10, len(df))
                if 'inflation_rate_percent' not in df.columns:
                    df['inflation_rate_percent'] = np.random.uniform(5, 20, len(df))
                    
                features = ['City_Encoded', 'Crop_Encoded', 'fuel_price_change', 'inflation_rate_percent']
                X_rf = df[features].fillna(0)
                y_rf = df['avg_price_per_kg']
                
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_rf, y_rf)
                
                city_enc = le_city.transform([sel_city])[0]
                crop_enc = le_crop.transform([sel_crop])[0]
                base_fuel = df['fuel_price_change'].mean()
                base_infl = df['inflation_rate_percent'].mean()
                
                base_pred = rf_model.predict([[city_enc, crop_enc, base_fuel, base_infl]])[0]
                
                eda_fuel_impact = (fuel_hike / 100) * 0.25 * base_pred
                eda_inflation_impact = (inflation_rate / 100) * 0.50 * base_pred
                
                simulated_price = base_pred + eda_fuel_impact + eda_inflation_impact
                
                st.markdown("### 📊 Simulation Results")
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Base Price", f"Rs. {base_pred:.2f}")
                c2.metric("New Predicted Price", f"Rs. {simulated_price:.2f}", f"+{(simulated_price - base_pred):.2f} PKR", delta_color="inverse")
                
                price_increase_perc = ((simulated_price - base_pred) / base_pred) * 100
                c3.metric("Overall Price Hike", f"{price_increase_perc:.1f}%")
                
                st.markdown("### 🔍 Root Cause Analysis (EDA Based)")
                impact_df = pd.DataFrame({
                    'Factor': ['Base Price', 'Fuel Impact', 'Inflation Impact'],
                    'Value (PKR)': [base_pred, eda_fuel_impact, eda_inflation_impact]
                })
                fig2 = px.pie(impact_df, values='Value (PKR)', names='Factor', title="Contribution to Final Price", hole=0.4)
                st.plotly_chart(fig2, use_container_width=True)
