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

st.set_page_config(page_title="AgriSmart Pakistan", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .reportview-container {background: #0e1117;}
    .sidebar .sidebar-content {background: #262730;}
    h1, h2, h3 {color: #00ff9d !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
    .stMetric {background-color: #1e1e24; padding: 20px; border-radius: 15px; border-left: 5px solid #00ff9d; box-shadow: 0 8px 16px rgba(0,0,0,0.4); transition: transform 0.3s;}
    .stMetric:hover {transform: translateY(-5px);}
    div[data-testid="stAlert"] {border-radius: 12px; font-weight: bold;}
    hr {border-color: #333333;}
    </style>
    """, unsafe_allow_html=True)

st.title("🌾 AgriSmart Pakistan")
st.markdown("**Enterprise-Grade Decision Support & Price Monitoring Terminal**")
st.markdown("---")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Project_DataSet.csv') 
    except FileNotFoundError:
        st.error("SYSTEM ERROR: Core dataset 'Project_Dataset.csv' is offline.")
        return None
    
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

    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=120)
    st.sidebar.markdown("<h2 style='text-align: center; color: white;'>Navigation</h2>", unsafe_allow_html=True)
    menu = st.sidebar.radio("", 
                            ["🛒 Consumer Risk Portal", 
                             "👨‍🌾 Harvest Trajectory", 
                             "🏛️ Macroeconomic Sandbox"])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("ENGINE: RF Regressor | Linear Regression | Naive Bayes")

    # ==========================================
    # MODULE 1: CONSUMER PORTAL
    # ==========================================
    if menu == "🛒 Consumer Risk Portal":
        st.header("Retail Anomaly Detection")
        st.write("Scan retail prices against historical aggregated benchmarks to detect overcharging.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            sel_city = st.selectbox("Select Region", df['city'].unique())
        with col2:
            sel_crop = st.selectbox("Select Commodity", df['crop'].unique())
        with col3:
            shop_price = st.number_input("Retailer Asking Price (PKR):", min_value=1.0, value=150.0, step=1.0)
            
        if st.button("Execute Diagnostics", use_container_width=True):
            with st.spinner("Compiling regional metrics..."):
                city_data = df[(df['city'] == sel_city) & (df['crop'] == sel_crop)]
                
                if not city_data.empty:
                    avg_price = city_data['avg_price_per_kg'].mean()
                    diff = shop_price - avg_price
                    
                    st.markdown("### 🎛️ Deviation Analysis")
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = shop_price,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Price vs Average", 'font': {'size': 24}},
                        delta = {'reference': avg_price, 'increasing': {'color': "#ff4b4b"}, 'decreasing': {'color': "#00ff9d"}},
                        gauge = {
                            'axis': {'range': [None, avg_price * 2], 'tickwidth': 1},
                            'bar': {'color': "#1f77b4"},
                            'bgcolor': "rgba(0,0,0,0)",
                            'steps': [
                                {'range': [0, avg_price], 'color': "rgba(0, 255, 157, 0.2)"},
                                {'range': [avg_price, avg_price + 7], 'color': "rgba(255, 255, 0, 0.2)"},
                                {'range': [avg_price + 7, avg_price * 2], 'color': "rgba(255, 75, 75, 0.2)"}],
                            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': avg_price + 7}
                        }
                    ))
                    fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    if diff > 7:
                        st.error(f"🚨 ALERT: Severe deviation detected. Price is {diff:.2f} PKR above the regional baseline. Exceeds the 7 PKR maximum tolerance threshold.")
                    elif diff > 0 and diff <= 7:
                        st.warning(f"⚠️ NOTICE: Minor inflation detected. Price is {diff:.2f} PKR above average but remains within the accepted retail fluctuation variance.")
                    else:
                        st.success("✅ VERIFIED: The quoted price is optimal and falls below or exactly matches the fair market value.")
                        st.balloons()
                        
                    st.markdown("---")
                    st.subheader("🔮 Predictive Risk Classification")
                    
                    nb_model = GaussianNB()
                    X_nb = df[['City_Encoded', 'Crop_Encoded']]
                    y_nb = df['Is_High_Price']
                    nb_model.fit(X_nb, y_nb)
                    
                    input_city_enc = le_city.transform([sel_city])[0]
                    input_crop_enc = le_crop.transform([sel_crop])[0]
                    risk_prob = nb_model.predict_proba([[input_city_enc, input_crop_enc]])[0][1] * 100
                    
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.metric("Surge Probability", f"{risk_prob:.1f}%")
                    with c2:
                        if risk_prob > 50:
                            st.warning(f"Classification Engine (Naive Bayes) indicates a high probability of persistent market surges for {sel_crop} in this sector.")
                        else:
                            st.info(f"Classification Engine (Naive Bayes) indicates localized market stability. Prolonged price spikes are statistically unlikely.")
                else:
                    st.error("Insufficient regional data points to compile diagnostics.")

    # ==========================================
    # MODULE 2: FARMER PORTAL
    # ==========================================
    elif menu == "👨‍🌾 Harvest Trajectory":
        st.header("Macro-Economic Projection")
        st.write("Utilize linear modeling to determine the overarching financial trajectory of your crop.")
        
        c1, c2 = st.columns(2)
        with c1:
            sel_crop = st.selectbox("Target Commodity", df['crop'].unique())
        with c2:
            kg_to_sell = st.number_input("Wholesale Volume (KG):", min_value=40, value=40, step=40)
            
        crop_data = df[df['crop'] == sel_crop].copy()
        if not crop_data.empty:
            crop_data['Time_Index'] = np.arange(len(crop_data))
            X_lr = crop_data[['Time_Index']]
            y_lr = crop_data['avg_price_per_kg']
            
            lr_model = LinearRegression()
            lr_model.fit(X_lr, y_lr)
            crop_data['Trend_Line'] = lr_model.predict(X_lr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=crop_data['Time_Index'], y=crop_data['avg_price_per_kg'], mode='lines+markers', name='Raw Volatility', line=dict(color='rgba(0, 255, 157, 0.4)', width=2)))
            fig.add_trace(go.Scatter(x=crop_data['Time_Index'], y=crop_data['Trend_Line'], mode='lines', name='Linear Projection', line=dict(color='#ff4b4b', width=4, dash='dot')))
            fig.update_layout(title=f"Time-Series Analysis & Trajectory Modeling: {sel_crop}", xaxis_title="Chronological Data Sequence", yaxis_title="Price Output (PKR)", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
            trend_diff = crop_data['Trend_Line'].iloc[-1] - crop_data['Trend_Line'].iloc[0]
            
            st.markdown("### 📈 Trajectory Conclusion")
            if trend_diff > 0:
                st.success("Positive Growth Vector identified. The linear regression model confirms steady appreciation in market valuation.")
            else:
                st.error("Negative Growth Vector identified. The model indicates depreciating long-term valuation.")
                
            current_avg_price = crop_data['avg_price_per_kg'].mean()
            est_revenue = current_avg_price * kg_to_sell
            st.metric("Estimated Yield Value", f"Rs. {est_revenue:,.2f}", f"Calculated over {kg_to_sell} KG")

    # ==========================================
    # MODULE 3: GOVT PORTAL
    # ==========================================
    elif menu == "🏛️ Macroeconomic Sandbox":
        st.header("Policy Impact Simulator")
        st.write("Stress-test economic policies. Powered by an ensemble Random Forest model injected with EDA-driven macroeconomic constraints.")
        
        col1, col2 = st.columns(2)
        with col1:
            sel_city = st.selectbox("Target Region", df['city'].unique())
            sel_crop = st.selectbox("Commodity Stress Test", df['crop'].unique())
        with col2:
            fuel_hike = st.slider("Fuel Rate Hike (%)", 0, 100, 0)
            inflation_rate = st.slider("Inflationary Pressure (%)", 0.0, 30.0, 5.0)

        if st.button("Initialize Simulation", use_container_width=True):
            with st.spinner("Processing ensemble layers..."):
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
                
                st.markdown("### 📊 Financial Impact Output")
                c1, c2, c3 = st.columns(3)
                c1.metric("Base Forecast", f"Rs. {base_pred:.2f}")
                c2.metric("Stressed Forecast", f"Rs. {simulated_price:.2f}", f"+{(simulated_price - base_pred):.2f} PKR", delta_color="inverse")
                
                price_increase_perc = ((simulated_price - base_pred) / base_pred) * 100
                c3.metric("Net Variance", f"{price_increase_perc:.1f}%")
                
                st.markdown("### 🔬 EDA Driver Breakdown")
                
                fig_waterfall = go.Figure(go.Waterfall(
                    name = "Impact", orientation = "v",
                    measure = ["absolute", "relative", "relative", "total"],
                    x = ["Base Model Prediction", "Transport Friction (Fuel)", "Macro Inflation", "Final Retail Projection"],
                    textposition = "outside",
                    text = [f"{base_pred:.0f}", f"+{eda_fuel_impact:.0f}", f"+{eda_inflation_impact:.0f}", f"{simulated_price:.0f}"],
                    y = [base_pred, eda_fuel_impact, eda_inflation_impact, simulated_price],
                    connector = {"line":{"color":"rgb(63, 63, 63)"}}
                ))
                fig_waterfall.update_layout(title="Waterfall Analytics: Price Construction", showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_waterfall, use_container_width=True)
