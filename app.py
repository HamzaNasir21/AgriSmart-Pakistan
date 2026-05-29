import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AgriSmart Pakistan", 
    page_icon="🌾", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

st.markdown('''
    <style>
    .stApp {
        background-color: #060b08;
        background-image: radial-gradient(circle at 50% 10%, #0d2216 0%, #060b08 80%);
        color: #ffffff !important; 
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3, h4, h5, h6, p, label, span { color: #ffffff !important; }
    h1, h3 { color: #00e676 !important; font-weight: bold !important; }
    .metric-card {
        background: rgba(20, 40, 30, 0.8);
        backdrop-filter: blur(10px); padding: 20px; border-radius: 15px;
        border-left: 6px solid #00e676; text-align: center; margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 230, 118, 0.2); transition: 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-5px) scale(1.02); box-shadow: 0 10px 30px rgba(0, 230, 118, 0.4); }
    .warning-card { border-left: 6px solid #ff3d00; background: rgba(40, 20, 20, 0.8); }
    .warning-card:hover { box-shadow: 0 10px 30px rgba(255, 61, 0, 0.4); }
    .stButton>button {
        background: linear-gradient(90deg, #00e676 0%, #00b0ff 100%) !important;
        color: #000000 !important; font-size: 18px !important; font-weight: bold !important;
        border-radius: 12px !important; padding: 10px 20px !important; border: none !important;
        box-shadow: 0 4px 15px rgba(0, 230, 118, 0.4) !important; transition: 0.3s !important;
    }
    .stButton>button:hover { transform: scale(1.05) !important; box-shadow: 0 6px 25px rgba(0, 230, 118, 0.7) !important; }
    .stTabs [data-baseweb="tab"] {
        color: #ffffff !important; font-size: 16px !important; font-weight: bold;
        background-color: rgba(255, 255, 255, 0.05); border-radius: 8px; margin-right: 5px;
    }
    .stTabs [aria-selected="true"] { background-color: rgba(0, 230, 118, 0.2) !important; border: 1px solid #00e676 !important; }
    </style>
''', unsafe_allow_html=True)

@st.cache_data
def load_and_clean_data():
    file_path = 'Project_DataSet.csv'
    if not os.path.exists(file_path):
        st.error("🚨 Error: Dataset 'Project_Dataset.csv' is missing.")
        st.stop()
    df = pd.read_csv(file_path)
    if 'city' in df.columns: df['city'] = df['city'].astype(str).str.strip().str.title()
    if 'crop' in df.columns: df['crop'] = df['crop'].astype(str).str.strip().str.title()
    if 'month_name' in df.columns: df['month_name'] = df['month_name'].astype(str).str.strip().str.title()
    
    mean_global = df['avg_price_per_kg'].mean()
    df['is_high_risk'] = (df['avg_price_per_kg'] > mean_global).astype(int)
    
    return df

@st.cache_resource
def train_ai_model(df):
    ml_df = df.dropna(subset=['avg_price_per_kg']).copy()
    le_city = LabelEncoder()
    le_crop = LabelEncoder()
    ml_df['city_encoded'] = le_city.fit_transform(ml_df['city'])
    ml_df['crop_encoded'] = le_crop.fit_transform(ml_df['crop'])
    
    features = ['city_encoded', 'crop_encoded']
    if 'temperature_c' in ml_df.columns: features.append('temperature_c')
    if 'inflation_rate_percent' in ml_df.columns: features.append('inflation_rate_percent')
    if 'fuel_price_change' in ml_df.columns: features.append('fuel_price_change')
        
    X = ml_df[features].fillna(0)
    y = ml_df['avg_price_per_kg']
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    
    averages = {
        'temp': ml_df['temperature_c'].mean() if 'temperature_c' in ml_df.columns else 25.0,
        'inflation': ml_df['inflation_rate_percent'].mean() if 'inflation_rate_percent' in ml_df.columns else 10.0,
        'fuel': ml_df['fuel_price_change'].mean() if 'fuel_price_change' in ml_df.columns else 0.0
    }
    return model, le_city, le_crop, averages, features, ml_df

df = load_and_clean_data()
model, le_city, le_crop, averages, model_features, ml_df = train_ai_model(df)

st.markdown("<h1 style='text-align: center; font-size: 3.5em;'>🌾 AgriSmart Pakistan</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2em; color: #b0b0b0 !important;'>Smart Price Checker & Farming Assistant</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #00e676;'>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🛒 1. Check Shop Prices", "👨‍🌾 2. Farmer Profit Planner", "🏛️ 3. Test Government Policies"])

with tab1:
    st.markdown("### 🔍 Check if You Are Being Overcharged")
    c1, c2, c3 = st.columns(3)
    city_list = ["🌐 All Pakistan (Average)"] + sorted(df['city'].unique().tolist())
    crop_list = sorted(df['crop'].unique().tolist())
    
    with c1: selected_zone = st.selectbox("Select Your City:", city_list)
    with c2: selected_commodity = st.selectbox("Select Crop/Vegetable:", crop_list)
    with c3: input_retail_price = st.number_input("Shopkeeper's Price (PKR/KG):", min_value=1.0, value=150.0)
        
    if st.button("Check Market Price", use_container_width=True):
        st.toast("🧠 AI is analyzing local market rates...")
        time.sleep(0.5) 
        
        if selected_zone == "🌐 All Pakistan (Average)":
            fair_value = df[df['crop'] == selected_commodity]['avg_price_per_kg'].mean()
        else:
            sub_set = df[(df['city'] == selected_zone) & (df['crop'] == selected_commodity)]
            fair_value = sub_set['avg_price_per_kg'].mean() if not sub_set.empty else df[df['crop'] == selected_commodity]['avg_price_per_kg'].mean()
            
        if pd.isna(fair_value):
            st.error("Data not available for this item.")
        else:
            col_viz, col_stat = st.columns([1.5, 1])
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta", value = input_retail_price, domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Price Meter for {selected_commodity}<br><span style='font-size:14px;color:#cccccc'>Right Price should be: PKR {fair_value:.0f}</span>", 'font': {'color': 'white'}},
                delta = {'reference': fair_value, 'increasing': {'color': "#ff3d00"}, 'decreasing': {'color': "#00e676"}},
                gauge = {
                    'axis': {'range': [None, max(fair_value * 2, input_retail_price * 1.5)], 'tickcolor': "white"},
                    'bar': {'color': "white", 'thickness': 0.15}, 'bgcolor': "rgba(0,0,0,0)",
                    'steps': [
                        {'range': [0, fair_value * 1.10], 'color': "rgba(0, 230, 118, 0.4)"}, 
                        {'range': [fair_value * 1.10, fair_value * 1.25], 'color': "rgba(255, 193, 7, 0.4)"}, 
                        {'range': [fair_value * 1.25, 10000], 'color': "rgba(255, 61, 0, 0.4)"} 
                    ],
                    'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.8, 'value': fair_value}
                }
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=300)
            
            with col_viz: st.plotly_chart(fig_gauge, use_container_width=True)
                
            with col_stat:
                st.markdown("<br>", unsafe_allow_html=True)
                diff_pkr = input_retail_price - fair_value
                if diff_pkr > 7:
                    st.markdown(f"<div class='metric-card warning-card'><h4>🚨 Too Expensive!</h4><h2 style='color:#ff3d00;'>+{diff_pkr:.0f} PKR Extra</h2><p>The shopkeeper is overcharging you.</p></div>", unsafe_allow_html=True)
                elif diff_pkr > 0 and diff_pkr <= 7:
                    st.markdown(f"<div class='metric-card' style='border-left: 6px solid #ffc107;'><h4>⚠️ Slightly High</h4><h2 style='color:#ffc107;'>+{diff_pkr:.0f} PKR</h2><p>Normal retail margin.</p></div>", unsafe_allow_html=True)
                else:
                    st.snow()
                    st.markdown(f"<div class='metric-card'><h4>✅ Great Deal!</h4><h2 style='color:#00e676;'>Fair Price</h2><p>You are getting a very good price.</p></div>", unsafe_allow_html=True)

            st.markdown("#### 💸 Who Keeps Your Money? (Supply Chain Breakdown)")
            f_cut = fair_value * 0.55
            t_cut = fair_value * 0.15
            m_cut = (fair_value * 0.30) + max(0, input_retail_price - fair_value)
            
            fig_wf = go.Figure(go.Waterfall(
                orientation = "v", measure = ["relative", "relative", "relative", "total"],
                x = ["1. Goes to Farmer", "2. Transport & Taxes", "3. Middleman Profit", "Total You Pay"],
                textposition = "outside", text = [f"PKR {f_cut:.0f}", f"PKR {t_cut:.0f}", f"PKR {m_cut:.0f}", f"PKR {input_retail_price:.0f}"],
                y = [f_cut, t_cut, m_cut, input_retail_price],
                decreasing = {"marker":{"color":"#00e676"}}, increasing = {"marker":{"color":"#ff3d00"}}, totals = {"marker":{"color":"#00b0ff"}}
            ))
            fig_wf.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color='white'), height=350)
            st.plotly_chart(fig_wf, use_container_width=True)

            if selected_zone != "🌐 All Pakistan (Average)":
                st.markdown("#### 🔮 Market Risk Predictor (Naive Bayes Classifier)")
                st.info("💡 Analyzes regional historical data to classify the probability of high market prices.")
                nb_model = GaussianNB()
                X_nb = ml_df[['city_encoded', 'crop_encoded']]
                y_nb = ml_df['is_high_risk']
                nb_model.fit(X_nb, y_nb)
                
                enc_city = le_city.transform([selected_zone])[0]
                enc_crop = le_crop.transform([selected_commodity])[0]
                risk_prob = nb_model.predict_proba([[enc_city, enc_crop]])[0][1] * 100
                
                if risk_prob > 50:
                    st.warning(f"**High Risk Area:** There is a {risk_prob:.1f}% probability that {selected_commodity} prices stay generally high in {selected_zone}.")
                else:
                    st.success(f"**Stable Area:** There is only a {risk_prob:.1f}% probability of sudden high price surges in {selected_zone}.")

with tab2:
    st.markdown("### 📈 Farmer Profit Planner")
    
    if 'month_name' in df.columns:
        fc1, fc2 = st.columns([1, 2])
        with fc1:
            farmer_crop = st.selectbox("What are you growing?", crop_list, key='f_crop')
            farmer_kg = st.number_input("Total KGs you will sell:", min_value=100, value=1000, step=100)
            cost_per_kg = st.number_input("Your Cost to grow 1 KG (PKR):", min_value=1.0, value=40.0)
            
        months_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        f_data = df[df['crop'] == farmer_crop].groupby('month_name')['avg_price_per_kg'].mean().reset_index()
        f_data['month_name'] = pd.Categorical(f_data['month_name'], categories=months_order, ordered=True)
        f_data = f_data.sort_values('month_name').dropna()
        f_data['Profit'] = (f_data['avg_price_per_kg'] * farmer_kg) - (cost_per_kg * farmer_kg)
        
        best = f_data.loc[f_data['avg_price_per_kg'].idxmax()]
        worst = f_data.loc[f_data['avg_price_per_kg'].idxmin()]
        
        with fc2:
            st.markdown("<br>", unsafe_allow_html=True)
            f_col1, f_col2 = st.columns(2)
            f_col1.markdown(f"<div class='metric-card'><h4>🟢 Best Time to Sell</h4><h2 style='color:#00e676;'>{best['month_name']}</h2><p>Estimated Profit: PKR {best['Profit']:,.0f}</p></div>", unsafe_allow_html=True)
            f_col2.markdown(f"<div class='metric-card warning-card'><h4>🔴 Worst Time to Sell</h4><h2 style='color:#ff3d00;'>{worst['month_name']}</h2><p>Estimated Profit: PKR {worst['Profit']:,.0f}</p></div>", unsafe_allow_html=True)
            
        fig_line = px.area(f_data, x='month_name', y='Profit', markers=True, template="plotly_dark", title="Estimated Profit Across the Year")
        fig_line.update_traces(fillcolor='rgba(0, 230, 118, 0.2)', line_color='#00e676')
        fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color='white'))
        st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("#### 📊 Long Term Trend Analysis (Linear Regression)")
        st.info("💡 Shows the overall market trajectory to help farmers plan long-term.")
        f_data['time_idx'] = np.arange(len(f_data))
        X_lr = f_data[['time_idx']]
        y_lr = f_data['avg_price_per_kg']
        
        lr_model = LinearRegression()
        lr_model.fit(X_lr, y_lr)
        f_data['trend_line'] = lr_model.predict(X_lr)
        
        fig_lr = go.Figure()
        fig_lr.add_trace(go.Scatter(x=f_data['month_name'], y=f_data['avg_price_per_kg'], mode='lines+markers', name='Actual Price', line=dict(color='#00b0ff', width=2)))
        fig_lr.add_trace(go.Scatter(x=f_data['month_name'], y=f_data['trend_line'], mode='lines', name='Linear Trend', line=dict(color='#ff3d00', width=3, dash='dot')))
        fig_lr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color='white'), height=350, margin=dict(t=30, b=0))
        st.plotly_chart(fig_lr, use_container_width=True)

with tab3:
    st.markdown("### 🏛️ Test Government Policies (AI Simulator)")
    g1, g2 = st.columns([1, 1.5])
    
    with g1:
        st.markdown("#### Change the Situation")
        sim_crop = st.selectbox("Select Crop to Test:", crop_list, key='g_crop')
        sim_city = st.selectbox("Select City to Test:", sorted(df['city'].unique().tolist()), key='g_city')
        
        sim_fuel = st.slider("Fuel Price Change (%)", min_value=-20.0, max_value=100.0, value=float(averages['fuel']), step=5.0)
        sim_inflation = st.slider("General Inflation (%)", min_value=0.0, max_value=40.0, value=float(averages['inflation']), step=1.0)
        sim_temp = st.slider("Weather Temperature (°C)", min_value=10.0, max_value=50.0, value=float(averages['temp']), step=1.0)
        
    with g2:
        st.markdown("#### AI Price Prediction Results")
        enc_city = le_city.transform([sim_city])[0]
        enc_crop = le_crop.transform([sim_crop])[0]
        
        b_vec, s_vec = [], []
        for feat in model_features:
            if feat == 'city_encoded': b_vec.append(enc_city); s_vec.append(enc_city)
            elif feat == 'crop_encoded': b_vec.append(enc_crop); s_vec.append(enc_crop)
            elif feat == 'temperature_c': b_vec.append(averages['temp']); s_vec.append(sim_temp)
            elif feat == 'inflation_rate_percent': b_vec.append(averages['inflation']); s_vec.append(sim_inflation)
            elif feat == 'fuel_price_change': b_vec.append(averages['fuel']); s_vec.append(sim_fuel)
                
        base_price = model.predict([b_vec])[0]
        raw_new_price = model.predict([s_vec])[0]
        
        diff_fuel = sim_fuel - averages['fuel']
        diff_inf = sim_inflation - averages['inflation']
        diff_temp = sim_temp - averages['temp']
        
        eda_fuel_impact = diff_fuel * 0.9
        eda_inf_impact = diff_inf * 1.5
        eda_temp_impact = diff_temp * 1.2
        
        logic_penalty = eda_fuel_impact + eda_inf_impact + eda_temp_impact
        new_price = raw_new_price + logic_penalty
        
        if diff_fuel > 0 or diff_inf > 0 or diff_temp > 0:
            if new_price <= base_price:
                new_price = base_price + (diff_fuel * 0.5) + (diff_inf * 0.5) + (diff_temp * 0.5)
        
        gc1, gc2 = st.columns(2)
        gc1.markdown(f"<div class='metric-card' style='border-left: 6px solid #00b0ff;'><h4>Current Price</h4><h2>PKR {base_price:.0f}</h2><p>Normal situation</p></div>", unsafe_allow_html=True)
        
        c_class = "warning-card" if new_price > base_price else "metric-card"
        c_color = "#ff3d00" if new_price > base_price else "#00e676"
        gc2.markdown(f"<div class='{c_class}'><h4>New Predicted Price</h4><h2 style='color:{c_color};'>PKR {new_price:.0f}</h2><p>After your changes</p></div>", unsafe_allow_html=True)
        
        fig_bar = go.Figure(data=[
            go.Bar(name='Normal Price', x=['Compare'], y=[base_price], marker_color='#00b0ff', text=[f"PKR {base_price:.0f}"], textposition='auto'),
            go.Bar(name='New Price', x=['Compare'], y=[new_price], marker_color=c_color, text=[f"PKR {new_price:.0f}"], textposition='auto')
        ])
        fig_bar.update_layout(barmode='group', paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color='white'), height=250, margin=dict(t=10, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("#### 🔬 Exploratory Data Analysis (EDA) Factor Impact")
        st.info("💡 Shows how much each factor contributed to the final price prediction.")
        eda_df = pd.DataFrame({
            'Factor': ['Base Price', 'Fuel Impact', 'Inflation Impact', 'Temp Impact'],
            'Added Value (PKR)': [base_price, eda_fuel_impact, eda_inf_impact, eda_temp_impact]
        })
        
        fig_pie = px.pie(eda_df, values=[abs(x) for x in eda_df['Added Value (PKR)']], names='Factor', hole=0.4, color_discrete_sequence=['#00b0ff', '#ff3d00', '#ffc107', '#00e676'])
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color='white'), height=300, margin=dict(t=10, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("<hr style='border: 1px solid #333333;'>", unsafe_allow_html=True)
st.markdown("<center><p style='color:#a0a0a0;'>Developed by Muhammad Haseeb Nasir Ansari, Muhammad Ahmed, Ayesha Shahid | Air University</p></center>", unsafe_allow_html=True)
