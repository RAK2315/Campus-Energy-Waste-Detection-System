import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime, timedelta
import json

# Page config
st.set_page_config(
    page_title="Campus Energy Waste Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #1f77b4;
        color: #000000;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        color: #000000;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        color: #000000;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load(r"D:\REHAAN\1. Ml Projects\6. Campus Energy Waste Detection\models\rf_forecaster.pkl")
        iso_model = joblib.load(r"D:\REHAAN\1. Ml Projects\6. Campus Energy Waste Detection\models\isolation_forest_anomaly.pkl")
        return rf_model, iso_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_data
def load_data():
    try:
        # Try original data first (has sub-metering)
        df = pd.read_csv(r"D:\REHAAN\1. Ml Projects\6. Campus Energy Waste Detection\data\dataset.txt",
                         sep=';', 
                         low_memory=False)
        
        # Clean it
        df = df.replace('?', np.nan)
        numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
        df = df.set_index('Datetime')
        df = df.drop(['Date', 'Time'], axis=1)
        df = df.sort_index()
        df = df.fillna(method='ffill').dropna()
        
        # Resample to hourly
        numeric_only = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        df = df[numeric_only].resample('H').mean()
        
        # Add time features
        df['Hour'] = df.index.hour
        df['Day'] = df.index.day
        df['Month'] = df.index.month
        df['Year'] = df.index.year
        df['DayOfWeek'] = df.index.dayofweek
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Add derived features for ML
        df['lag_1h'] = df['Global_active_power'].shift(1)
        df['lag_24h'] = df['Global_active_power'].shift(24)
        df['lag_168h'] = df['Global_active_power'].shift(168)
        df['rolling_mean_24h'] = df['Global_active_power'].rolling(window=24).mean()
        df['rolling_std_24h'] = df['Global_active_power'].rolling(window=24).std()
        df['rolling_max_24h'] = df['Global_active_power'].rolling(window=24).max()
        df['rolling_min_24h'] = df['Global_active_power'].rolling(window=24).min()
        df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df = df.dropna()
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load
with st.spinner("🔄 Loading models and data..."):
    rf_model, iso_model = load_models()
    df = load_data()

if df is None:
    st.error("❌ Failed to load data. Please check file paths.")
    st.stop()

# Header
st.markdown('<h1 class="main-header">⚡ Campus Energy Waste Detection System</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.2rem;'>AI-Powered Energy Monitoring for Sustainable Campuses</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
st.sidebar.title("🎛️ Dashboard")
st.sidebar.markdown("---")

view_mode = st.sidebar.radio(
    "Select View:",
    ["🏠 Overview", "🔍 Real-time Monitor", "🚨 Waste Analysis", "🔮 Predictions", "💰 Savings Calculator"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
**📊 Dataset Info:**
- **Samples:** 34,421 hours
- **Features:** 26 variables
- **Model:** Random Forest + Isolation Forest
""")

# Calculate real metrics from data
total_consumption = df['Global_active_power'].sum()
mean_consumption = df['Global_active_power'].mean()
peak_consumption = df['Global_active_power'].max()

# Detect anomalies
anomaly_features = ['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
                   'Hour', 'DayOfWeek', 'IsWeekend']
X_anomaly = df[anomaly_features]

if iso_model:
    anomaly_labels = iso_model.predict(X_anomaly)
    df['is_anomaly'] = (anomaly_labels == -1).astype(int)
else:
    df['is_anomaly'] = 0

n_anomalies = df['is_anomaly'].sum()
anomaly_pct = (n_anomalies / len(df)) * 100

# ===== OVERVIEW PAGE =====
if view_mode == "🏠 Overview":
    st.header("📊 System Overview")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📈 Avg Consumption", 
            f"{mean_consumption:.2f} kW",
            delta=f"{((mean_consumption - 1.0) / 1.0 * 100):.1f}% vs baseline"
        )
    
    with col2:
        st.metric(
            "⚠️ Waste Events", 
            f"{n_anomalies:,}",
            delta=f"{anomaly_pct:.1f}% of time",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "💰 Potential Savings", 
            "₹11 Lakh+/year",
            help="For 1000-student campus"
        )
    
    with col4:
        st.metric(
            "🎯 Model Accuracy", 
            "90%",
            delta="R² Score"
        )
    
    st.markdown("---")
    
    # Two column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Consumption Trend (Last 30 Days)")
        
        last_30_days = df.tail(720)
        anomalies_30 = last_30_days[last_30_days['is_anomaly'] == 1]
        
        fig = go.Figure()
        
        # Normal consumption
        fig.add_trace(go.Scatter(
            x=last_30_days.index,
            y=last_30_days['Global_active_power'],
            mode='lines',
            name='Consumption',
            line=dict(color='steelblue', width=2),
            fill='tozeroy',
            fillcolor='rgba(70, 130, 180, 0.1)'
        ))
        
        # Anomalies
        fig.add_trace(go.Scatter(
            x=anomalies_30.index,
            y=anomalies_30['Global_active_power'],
            mode='markers',
            name='Waste Events',
            marker=dict(color='red', size=8, symbol='x')
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Power (kW)",
            height=400,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        recent_mean = last_30_days['Global_active_power'].mean()
        recent_anomalies = len(anomalies_30)
        
        st.markdown(f"""
        <div class="info-box">
        <b>📊 Last 30 Days Summary:</b><br>
        • Average consumption: <b>{recent_mean:.2f} kW</b><br>
        • Waste events detected: <b>{recent_anomalies}</b><br>
        • Peak consumption: <b>{last_30_days['Global_active_power'].max():.2f} kW</b><br>
        • Lowest consumption: <b>{last_30_days['Global_active_power'].min():.2f} kW</b>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🏠 Energy Distribution")
        
        sub_totals = {
            'Kitchen': df['Sub_metering_1'].sum(),
            'Laundry': df['Sub_metering_2'].sum(),
            'AC/Heater': df['Sub_metering_3'].sum()
        }
        
        fig = px.pie(
            values=list(sub_totals.values()),
            names=list(sub_totals.keys()),
            color_discrete_sequence=['#ff9999', '#66b3ff', '#99ff99'],
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="warning-box">
        <b>🎯 Key Finding:</b><br>
        AC/Heater accounts for <b>73%</b> of total consumption - primary optimization target!
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Hourly and weekly patterns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏰ Average Hourly Pattern")
        hourly_avg = df.groupby('Hour')['Global_active_power'].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(24)),
            y=hourly_avg.values,
            marker_color=['red' if x in [19, 20, 21] else 'lightblue' for x in range(24)],
            name='Hourly Average'
        ))
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Power (kW)",
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        peak_hour = hourly_avg.idxmax()
        st.markdown(f"""
        <div class="warning-box">
        <b>⚡ Peak Hour:</b> {peak_hour}:00 ({hourly_avg[peak_hour]:.2f} kW)<br>
        <b>💤 Low Hour:</b> {hourly_avg.idxmin()}:00 ({hourly_avg.min():.2f} kW)
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📅 Weekly Consumption Pattern")
        
        weekly = df.groupby('DayOfWeek')['Global_active_power'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=days,
            y=weekly.values,
            marker_color=['orange' if x in [5, 6] else 'lightgreen' for x in range(7)],
            name='Daily Average'
        ))
        fig.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Power (kW)",
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        weekday_avg = weekly[:5].mean()
        weekend_avg = weekly[5:].mean()
        diff_pct = ((weekend_avg - weekday_avg) / weekday_avg) * 100
        
        st.markdown(f"""
        <div class="info-box">
        <b>📊 Weekly Pattern:</b><br>
        • Weekday avg: <b>{weekday_avg:.2f} kW</b><br>
        • Weekend avg: <b>{weekend_avg:.2f} kW</b><br>
        • Difference: <b>{diff_pct:+.1f}%</b> (weekends higher)
        </div>
        """, unsafe_allow_html=True)

# ===== REAL-TIME MONITOR =====
elif view_mode == "🔍 Real-time Monitor":
    st.header("🔍 Real-time Energy Monitor")
    
    # Get current reading (last data point with actual values)
    current_data = df.iloc[-1]
    current_consumption = current_data['Global_active_power']
    hour_now = current_data['Hour']
    
 
    
    # Status
    if current_consumption > mean_consumption * 1.5:
        status = "🔴 HIGH"
        status_color = "danger-box"
        status_msg = "Consumption is significantly above normal. Check for waste!"
    elif current_consumption > mean_consumption:
        status = "🟡 MODERATE"
        status_color = "warning-box"
        status_msg = "Consumption is slightly elevated. Monitor closely."
    else:
        status = "🟢 NORMAL"
        status_color = "success-box"
        status_msg = "Consumption is within normal range."
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("⚡ Current Consumption", f"{current_consumption:.2f} kW")
    
    with col2:
        st.metric("📊 vs Average", f"{((current_consumption / mean_consumption - 1) * 100):+.1f}%")
    
    with col3:
        st.metric("🕐 Current Hour", f"{int(hour_now)}:00")
    
    st.markdown(f"""
    <div class="{status_color}">
    <b>Status: {status}</b><br>
    {status_msg}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Live meters with fallback to averages if current is 0
    st.subheader("📊 Current Sub-metering Breakdown")
    
    # Get values with fallback to recent averages
    kitchen_val = current_data.get('Sub_metering_1', 0)
    laundry_val = current_data.get('Sub_metering_2', 0)
    ac_val = current_data.get('Sub_metering_3', 0)
    
    # If all zeros, use last 24h average for visualization
    if kitchen_val == 0 and laundry_val == 0 and ac_val == 0:
        last_24h = df.tail(24)
        kitchen_val = last_24h['Sub_metering_1'].mean()
        laundry_val = last_24h['Sub_metering_2'].mean()
        ac_val = last_24h['Sub_metering_3'].mean()
        st.info("ℹ️ Showing 24-hour average values (current readings unavailable)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**🍳 Kitchen**")
        st.metric("", f"{kitchen_val:.2f} Wh")
        progress_val = min(kitchen_val / 10, 1.0) if kitchen_val > 0 else 0.01
        st.progress(progress_val)
        st.caption(f"Range: 0-10 Wh | Current: {(kitchen_val/10*100):.0f}%")
    
    with col2:
        st.markdown(f"**🧺 Laundry**")
        st.metric("", f"{laundry_val:.2f} Wh")
        progress_val = min(laundry_val / 10, 1.0) if laundry_val > 0 else 0.01
        st.progress(progress_val)
        st.caption(f"Range: 0-10 Wh | Current: {(laundry_val/10*100):.0f}%")
    
    with col3:
        st.markdown(f"**❄️ AC/Heater**")
        st.metric("", f"{ac_val:.2f} Wh")
        progress_val = min(ac_val / 20, 1.0) if ac_val > 0 else 0.01
        st.progress(progress_val)
        st.caption(f"Range: 0-20 Wh | Current: {(ac_val/20*100):.0f}%")
    
    st.markdown("---")
    
    # Recent history
    st.subheader("📈 Last 24 Hours")
    last_24h = df.tail(24)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(24)),
        y=last_24h['Global_active_power'].values,
        mode='lines+markers',
        name='Consumption',
        line=dict(color='purple', width=3),
        marker=dict(size=8)
    ))
    fig.add_hline(y=mean_consumption, line_dash="dash", line_color="green", 
                  annotation_text="Average", annotation_position="right")
    fig.update_layout(
        xaxis_title="Hours Ago",
        yaxis_title="Power (kW)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# ===== WASTE ANALYSIS =====
elif view_mode == "🚨 Waste Analysis":
    st.header("🚨 Waste Detection & Analysis")
    
    anomalies_df = df[df['is_anomaly'] == 1]
    normal_df = df[df['is_anomaly'] == 0]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🚨 Total Waste Events", f"{len(anomalies_df):,}")
    
    with col2:
        st.metric("📊 Waste Percentage", f"{anomaly_pct:.2f}%")
    
    with col3:
        waste_avg = anomalies_df['Global_active_power'].mean()
        normal_avg = normal_df['Global_active_power'].mean()
        st.metric("⚡ Avg Waste Consumption", f"{waste_avg:.2f} kW", 
                 delta=f"{((waste_avg/normal_avg - 1)*100):+.0f}% vs normal",
                 delta_color="inverse")
    
    with col4:
        excess_kwh = (waste_avg - normal_avg) * len(anomalies_df)
        st.metric("💸 Excess Energy", f"{excess_kwh:,.0f} kWh")
    
    st.markdown("---")
    
    # Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Waste by Day of Week")
        waste_by_day = anomalies_df.groupby('DayOfWeek').size()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig = px.bar(
            x=days,
            y=[waste_by_day.get(i, 0) for i in range(7)],
            labels={'x': 'Day', 'y': 'Waste Events'},
            color=[waste_by_day.get(i, 0) for i in range(7)],
            color_continuous_scale='Reds'
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        weekend_waste = waste_by_day.get(5, 0) + waste_by_day.get(6, 0)
        weekday_waste = sum([waste_by_day.get(i, 0) for i in range(5)])
        
        st.markdown(f"""
        <div class="warning-box">
        <b>📊 Finding:</b> Weekend waste is <b>{(weekend_waste/(weekend_waste+weekday_waste)*100):.1f}%</b> of total!<br>
        Implement stricter weekend monitoring.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("⏰ Waste by Hour")
        waste_by_hour = anomalies_df.groupby('Hour').size().sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=waste_by_hour.index,
            y=waste_by_hour.values,
            labels={'x': 'Hour', 'y': 'Waste Events'},
            color=waste_by_hour.values,
            color_continuous_scale='Oranges'
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        top_hour = waste_by_hour.index[0]
        st.markdown(f"""
        <div class="danger-box">
        <b>⚠️ Peak Waste Hour:</b> <b>{top_hour}:00</b> with <b>{waste_by_hour.values[0]}</b> events<br>
        Focus intervention during evening hours.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Night waste analysis
    st.subheader("🌙 Night-time Waste (12 AM - 6 AM)")
    
    night_df = df[(df['Hour'] >= 0) & (df['Hour'] < 6)]
    night_waste = night_df[night_df['is_anomaly'] == 1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🌙 Night Hours", f"{len(night_df):,}")
    
    with col2:
        night_waste_pct = (len(night_waste) / len(night_df)) * 100
        st.metric("🚨 Night Waste %", f"{night_waste_pct:.1f}%")
    
    with col3:
        night_savings = (night_waste['Global_active_power'].mean() - night_df['Global_active_power'].mean()) * len(night_waste) * 6  # ₹/kWh
        st.metric("💰 Potential Savings", f"₹{night_savings:,.0f}")
    
    # Sub-metering during waste
    st.subheader("🔍 What's Running During Waste?")
    
    waste_submeter = anomalies_df[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].mean()
    normal_submeter = normal_df[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].mean()
    
    comparison = pd.DataFrame({
        'Area': ['Kitchen', 'Laundry', 'AC/Heater'],
        'Normal': [normal_submeter['Sub_metering_1'], normal_submeter['Sub_metering_2'], normal_submeter['Sub_metering_3']],
        'During Waste': [waste_submeter['Sub_metering_1'], waste_submeter['Sub_metering_2'], waste_submeter['Sub_metering_3']]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Normal', x=comparison['Area'], y=comparison['Normal'], marker_color='lightgreen'))
    fig.add_trace(go.Bar(name='During Waste', x=comparison['Area'], y=comparison['During Waste'], marker_color='red'))
    fig.update_layout(barmode='group', height=400, yaxis_title="Energy (Wh)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="info-box">
    <b>💡 Actionable Recommendations:</b><br>
    1. <b>🌡️ AC/Heater Control:</b> Install smart thermostats with scheduling (auto-off after 11 PM)<br>
    2. <b>🌙 Motion Sensors:</b> Auto-shutoff in unoccupied areas during night<br>
    3. <b>📅 Weekend Audits:</b> Increase facility checks on Saturdays/Sundays<br>
    4. <b>⏰ Evening Limits:</b> Set consumption caps during 7-10 PM peak hours<br>
    5. <b>📊 Real-time Alerts:</b> SMS notifications when consumption exceeds thresholds
    </div>
    """, unsafe_allow_html=True)

# ===== PREDICTIONS =====
elif view_mode == "🔮 Predictions":
    st.header("🔮 Consumption Forecasting")
    
    st.markdown("""
    <div class="info-box">
    <b>🤖 Model Information:</b><br>
    • <b>Algorithm:</b> Random Forest Regressor<br>
    • <b>Accuracy:</b> 90% (R² Score)<br>
    • <b>RMSE:</b> 0.23 kW (23% of mean)<br>
    • <b>Features:</b> 18 time-series and consumption features
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Get last 24 hours + predict next 24
    last_24h = df.tail(24).copy()
    
    # Feature columns for prediction
    feature_cols = ['Hour', 'DayOfWeek', 'IsWeekend', 'Month', 
                    'lag_1h', 'lag_24h', 'lag_168h',
                    'rolling_mean_24h', 'rolling_std_24h', 
                    'rolling_max_24h', 'rolling_min_24h',
                    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    
    # Predict next 24 hours
    if rf_model:
        predictions = []
        for i in range(24):
            X_pred = last_24h.iloc[-1:][feature_cols]
            pred = rf_model.predict(X_pred)[0]
            predictions.append(pred)
            
            # Create next row (simplified)
            next_row = last_24h.iloc[-1].copy()
            next_row['Global_active_power'] = pred
            next_row['Hour'] = (next_row['Hour'] + 1) % 24
            last_24h = pd.concat([last_24h, next_row.to_frame().T])
        
        # Visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Next 24 Hours Forecast")
            
            hours = list(range(24))
            
            fig = go.Figure()
            
            # Historical
            fig.add_trace(go.Scatter(
                x=hours,
                y=df.tail(24)['Global_active_power'].values,
                mode='lines+markers',
                name='Historical (Last 24h)',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
            
            # Predicted
            fig.add_trace(go.Scatter(
                x=hours,
                y=predictions,
                mode='lines+markers',
                name='Predicted (Next 24h)',
                line=dict(color='green', width=3, dash='dash'),
                marker=dict(size=10, symbol='diamond')
            ))
            
            # Average line
            fig.add_hline(y=mean_consumption, line_dash="dot", line_color="red",
                         annotation_text="Overall Average", annotation_position="right")
            
            fig.update_layout(
                xaxis_title="Hour",
                yaxis_title="Power (kW)",
                height=450,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Forecast Summary")
            
            pred_total = sum(predictions)
            pred_mean = np.mean(predictions)
            pred_peak = max(predictions)
            pred_peak_hour = predictions.index(pred_peak)
            pred_low = min(predictions)
            pred_low_hour = predictions.index(pred_low)
            
            st.metric("📦 Total Forecast", f"{pred_total:.2f} kWh")
            st.metric("📊 Average", f"{pred_mean:.2f} kW")
            st.metric("🔥 Peak", f"{pred_peak:.2f} kW", delta=f"at {pred_peak_hour}:00")
            st.metric("💤 Low", f"{pred_low:.2f} kW", delta=f"at {pred_low_hour}:00")
            
            # Status
            if pred_peak > mean_consumption * 1.5:
                st.markdown("""
                <div class="danger-box">
                <b>⚠️ Alert:</b> High consumption expected at <b>{pred_peak_hour}:00</b>!<br>
                Prepare for potential waste event.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                <b>✅ Status:</b> Normal consumption pattern expected.<br>
                No major anomalies forecasted.
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Hourly breakdown table
        st.subheader("📋 Detailed Hourly Forecast")
        
        # More sensitive thresholds
        high_threshold = mean_consumption * 1.2  # 20% above average
        moderate_threshold = mean_consumption * 0.9  # 10% below average
        
        def get_status(pred):
            if pred > high_threshold:
                return '🔴 High'
            elif pred > mean_consumption:
                return '🟡 Moderate'
            else:
                return '🟢 Normal'
        
        forecast_df = pd.DataFrame({
            'Hour': [f"{i:02d}:00" for i in range(24)],
            'Predicted (kW)': [f"{p:.2f}" for p in predictions],
            'vs Average': [f"{((p/mean_consumption - 1)*100):+.1f}%" for p in predictions],
            'Status': [get_status(p) for p in predictions]
        })
        
        # Highlight rows by status
        def highlight_status(row):
            if row['Status'] == '🔴 High':
                return ['background-color: #f8d7da; color: #000000'] * len(row)
            elif row['Status'] == '🟡 Moderate':
                return ['background-color: #fff3cd; color: #000000'] * len(row)
            else:
                return ['background-color: #d4edda; color: #000000'] * len(row)
        
        styled_df = forecast_df.style.apply(highlight_status, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        st.markdown("""
        <div class="info-box">
        <b>🎯 How to Use This Forecast:</b><br>
        • <b>Red hours:</b> Pre-allocate resources, increase monitoring<br>
        • <b>Yellow hours:</b> Standard vigilance, track trends<br>
        • <b>Green hours:</b> Opportunity for maintenance/low-priority tasks<br>
        • <b>Compare daily:</b> Track accuracy to improve future predictions
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.error("❌ Model not loaded. Cannot generate predictions.")

# ===== SAVINGS CALCULATOR =====
elif view_mode == "💰 Savings Calculator":
    st.header("💰 Energy Savings Calculator")
    
    st.markdown("""
    <div class="info-box">
    <b>💡 Calculate Your Potential Savings</b><br>
    Adjust parameters below to estimate cost savings from reducing wasteful consumption.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏫 Campus Parameters")
        campus_size = st.slider("Campus Size (students)", 100, 10000, 1000, 100)
        buildings = st.number_input("Number of Buildings", 1, 50, 10, 1)
        cost_per_kwh = st.number_input("Electricity Cost (₹/kWh)", 1.0, 20.0, 6.0, 0.5)
    
    with col2:
        st.subheader("🎯 Intervention Plan")
        waste_reduction = st.slider("Waste Reduction Target (%)", 0, 100, 50, 5)
        implementation_months = st.selectbox("Implementation Period (months)", [1, 3, 6, 12], index=2)
        investment = st.number_input("Initial Investment (₹)", 0, 1000000, 50000, 10000)
    
    st.markdown("---")
    
    # Calculate savings
    base_annual_kwh = 18257  # From analysis
    scaled_annual = base_annual_kwh * (campus_size / 100)
    reduced_waste_kwh = scaled_annual * (waste_reduction / 100)
    
    daily_kwh = reduced_waste_kwh / 365
    monthly_kwh = reduced_waste_kwh / 12
    
    daily_savings = daily_kwh * cost_per_kwh
    monthly_savings = monthly_kwh * cost_per_kwh
    annual_savings = reduced_waste_kwh * cost_per_kwh
    
    roi_months = investment / monthly_savings if monthly_savings > 0 else 999
    
    # Results
    st.subheader("📊 Projected Savings")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💵 Daily Savings", f"₹{daily_savings:,.0f}")
    
    with col2:
        st.metric("💵 Monthly Savings", f"₹{monthly_savings:,.0f}")
    
    with col3:
        st.metric("💵 Annual Savings", f"₹{annual_savings:,.0f}")
    
    with col4:
        st.metric("⏱️ ROI Period", f"{roi_months:.1f} months")
    
    # Savings over time
    st.subheader("📈 Cumulative Savings Over Time")
    
    months = list(range(1, implementation_months + 1))
    cumulative = [monthly_savings * m for m in months]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months,
        y=cumulative,
        mode='lines+markers',
        name='Cumulative Savings',
        line=dict(color='green', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 0, 0.1)'
    ))
    fig.add_hline(y=investment, line_dash="dash", line_color="red",
                 annotation_text=f"Investment: ₹{investment:,}", annotation_position="right")
    fig.update_layout(
        xaxis_title="Months",
        yaxis_title="Savings (₹)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💡 Savings by Category")
        
        categories = ['AC/Heater', 'Night Waste', 'Weekend Excess', 'Other']
        percentages = [73, 15, 8, 4]
        category_savings = [annual_savings * (p/100) for p in percentages]
        
        breakdown_df = pd.DataFrame({
            'Category': categories,
            'Percentage': percentages,
            'Annual Savings (₹)': category_savings
        })
        
        fig = px.bar(
            breakdown_df, 
            x='Category', 
            y='Annual Savings (₹)',
            color='Annual Savings (₹)',
            color_continuous_scale='Greens',
            text='Annual Savings (₹)'
        )
        fig.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Implementation Priorities")
        
        priorities = pd.DataFrame({
            'Action': [
                '🌡️ Smart Thermostats',
                '🌙 Motion Sensors',
                '⚡ Real-time Monitoring',
                '📅 Weekend Protocols',
                '🔔 Alert System'
            ],
            'Impact': ['High', 'High', 'Medium', 'Medium', 'Low'],
            'Cost': ['Medium', 'Low', 'High', 'Low', 'Low'],
            'Priority': [1, 2, 3, 4, 5]
        })
        
        st.dataframe(priorities, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="success-box">
        <b>✅ Quick Wins (Start Here):</b><br>
        1. Motion sensors in common areas<br>
        2. Weekend monitoring protocols<br>
        3. Alert system for high consumption<br>
        <br>
        <b>💰 ROI: 2-3 months for these actions</b>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Environmental impact
    st.subheader("🌍 Environmental Impact")
    
    co2_per_kwh = 0.82  # kg CO2 per kWh (India average)
    co2_saved = reduced_waste_kwh * co2_per_kwh
    trees_equivalent = co2_saved / 20  # Rough: 1 tree absorbs ~20kg CO2/year
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🌱 CO₂ Reduced", f"{co2_saved:,.0f} kg/year")
    
    with col2:
        st.metric("🌳 Trees Equivalent", f"{trees_equivalent:,.0f} trees")
    
    with col3:
        st.metric("♻️ Carbon Offset", f"{(co2_saved/1000):.1f} tons")
    
    st.markdown("""
    <div class="info-box">
    <h3 style="color: #000000; margin-top: 0;">🌍 Sustainability Impact</h3>
    By reducing waste by <b>{:.0f}%</b>, your campus will:<br>
    • Save <b>₹{:,.0f}</b> annually<br>
    • Reduce CO₂ emissions by <b>{:,.0f} kg</b><br>
    • Equivalent to planting <b>{:,.0f} trees</b><br>
    • Contribute to UN SDG #7 (Affordable & Clean Energy) and #13 (Climate Action)
    </div>
    """.format(waste_reduction, annual_savings, co2_saved, trees_equivalent), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 5px; font-size: 0.85rem; color: #666; margin-top: 30px;'>
⚡ <b>Campus Energy Waste Detection System</b> | <br>
| 🤖 Random Forest + Isolation Forest | 📊 90% Accuracy
</div>
""", unsafe_allow_html=True)