import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re

# Page config
st.set_page_config(page_title="Campus Energy Waste Detector", page_icon="⚡", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .info-box, .warning-box, .success-box, .danger-box, .metric-box {
        padding: 15px; border-radius: 8px; color: #000000; margin: 10px 0;
    }
    .info-box { background-color: #e3f2fd; border-left: 5px solid #2196f3; }
    .warning-box { background-color: #fff3cd; border-left: 5px solid #ffc107; }
    .success-box { background-color: #d4edda; border-left: 5px solid #28a745; }
    .danger-box { background-color: #f8d7da; border-left: 5px solid #dc3545; }
    .metric-box { background-color: #f3e5f5; border-left: 5px solid #9c27b0; }
    .grade-A { color: #28a745; font-size: 3rem; font-weight: bold; }
    .grade-B { color: #8bc34a; font-size: 3rem; font-weight: bold; }
    .grade-C { color: #ffc107; font-size: 3rem; font-weight: bold; }
    .grade-D { color: #ff5722; font-size: 3rem; font-weight: bold; }
    .grade-F { color: #dc3545; font-size: 3rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Helper functions
def smart_datetime_parse(series):
    formats = [
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M',
        '%m/%d/%Y %H:%M',
        '%Y/%m/%d %H:%M:%S',
    ]
    try:
        return pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
    except:
        pass
    for fmt in formats:
        try:
            result = pd.to_datetime(series, format=fmt, errors='coerce')
            if result.notna().sum() > len(series) * 0.5:
                return result
        except:
            continue
    return pd.to_datetime(series, errors='coerce')

def get_energy_grade(anomaly_pct):
    if anomaly_pct < 10:
        return "A", "Excellent", "#28a745"
    elif anomaly_pct < 20:
        return "B", "Good", "#8bc34a"
    elif anomaly_pct < 35:
        return "C", "Average", "#ffc107"
    elif anomaly_pct < 50:
        return "D", "Poor", "#ff5722"
    else:
        return "F", "Critical", "#dc3545"

def process_uploaded_data(raw_df, datetime_col, power_col, sub1_col, sub2_col, sub3_col):
    try:
        df = raw_df.copy()
        df['Datetime'] = smart_datetime_parse(df[datetime_col])
        valid_dates = df['Datetime'].notna()
        if valid_dates.sum() == 0:
            return None, "❌ Could not parse any datetime values"
        df = df[valid_dates].copy()
        df['Power'] = pd.to_numeric(df[power_col], errors='coerce')
        valid_power = df['Power'].notna() & (df['Power'] >= 0)
        if valid_power.sum() == 0:
            return None, "❌ No valid power values found"
        df = df[valid_power].copy()
        df['Sub1'] = pd.to_numeric(df[sub1_col], errors='coerce').fillna(0) if sub1_col != '-- None --' else 0
        df['Sub2'] = pd.to_numeric(df[sub2_col], errors='coerce').fillna(0) if sub2_col != '-- None --' else 0
        df['Sub3'] = pd.to_numeric(df[sub3_col], errors='coerce').fillna(0) if sub3_col != '-- None --' else 0
        df = df.set_index('Datetime').sort_index()
        df = df[['Power', 'Sub1', 'Sub2', 'Sub3']].copy()
        time_diff = df.index.to_series().diff().median()
        if pd.notna(time_diff) and time_diff < pd.Timedelta(minutes=30):
            df = df.resample('H').mean()
        df = df.dropna(subset=['Power'])
        if len(df) < 24:
            return None, "❌ Insufficient data (need at least 24 hours)"
        df['Hour'] = df.index.hour
        df['DayOfWeek'] = df.index.dayofweek
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df['Month'] = df.index.month
        mean_power = df['Power'].mean()
        std_power = df['Power'].std()
        threshold_statistical = mean_power + 1.2 * std_power
        night_mask = (df['Hour'] >= 0) & (df['Hour'] < 6)
        night_threshold = df[night_mask]['Power'].quantile(0.6) if night_mask.any() else mean_power
        weekend_mask = df['IsWeekend'] == 1
        weekend_threshold = df[weekend_mask]['Power'].quantile(0.7) if weekend_mask.any() else mean_power
        df['is_anomaly'] = (
            (df['Power'] > threshold_statistical) |
            ((df['Hour'] >= 0) & (df['Hour'] < 6) & (df['Power'] > night_threshold)) |
            ((df['IsWeekend'] == 1) & (df['Power'] > weekend_threshold))
        ).astype(int)
        # Tag waste reason
        df['waste_reason'] = 'Normal'
        df.loc[(df['Power'] > threshold_statistical), 'waste_reason'] = 'Statistically High'
        df.loc[((df['Hour'] >= 0) & (df['Hour'] < 6) & (df['Power'] > night_threshold)), 'waste_reason'] = 'Night-time Waste'
        df.loc[((df['IsWeekend'] == 1) & (df['Power'] > weekend_threshold)), 'waste_reason'] = 'Weekend Waste'
        return df, None
    except Exception as e:
        return None, f"❌ Processing error: {str(e)}"

# Title
st.title("⚡ Campus Energy Waste Detection System")
st.markdown("AI-powered energy monitoring for smarter, greener campuses")

# Sidebar
st.sidebar.header("📤 Data Upload")

if st.session_state.processed_data is not None:
    st.sidebar.success("✅ Data loaded!")
    st.sidebar.info(f"Records: {len(st.session_state.processed_data):,}")
    if st.sidebar.button("🔄 Upload New Data", type="secondary"):
        st.session_state.df = None
        st.session_state.processed_data = None
        st.rerun()

uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx', 'xls'])

if uploaded_file and st.session_state.processed_data is None:
    try:
        if uploaded_file.name.endswith('.csv'):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)
        st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")
        st.sidebar.info(f"{len(raw_df):,} rows × {len(raw_df.columns)} cols")
        st.session_state.df = raw_df
    except Exception as e:
        st.sidebar.error(f"❌ Load error: {e}")

# Column mapping interface
if st.session_state.df is not None and st.session_state.processed_data is None:
    raw_df = st.session_state.df
    st.header("📋 Data Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)
    st.markdown("---")
    st.header("🔗 Map Your Columns")
    cols = ['-- Select Column --'] + list(raw_df.columns)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📍 Required Fields")
        datetime_col = st.selectbox("📅 Datetime/Timestamp", cols, key='dt')
        power_col = st.selectbox("⚡ Power Consumption", cols, key='pw')
        if datetime_col != '-- Select Column --':
            st.caption(f"Sample: {raw_df[datetime_col].head(2).tolist()}")
        if power_col != '-- Select Column --':
            st.caption(f"Sample: {raw_df[power_col].head(2).tolist()}")
    with col2:
        st.subheader("🏫 Optional Sub-metering (Campus Zones)")
        sub1 = st.selectbox("Zone 1 (Labs / Canteen)", ['-- None --'] + list(raw_df.columns), key='s1')
        sub2 = st.selectbox("Zone 2 (Classrooms / Offices)", ['-- None --'] + list(raw_df.columns), key='s2')
        sub3 = st.selectbox("Zone 3 (HVAC / Common Areas)", ['-- None --'] + list(raw_df.columns), key='s3')
    st.markdown("---")
    if st.button("🚀 Process Data & Generate Dashboard", type="primary", use_container_width=True):
        if datetime_col == '-- Select Column --' or power_col == '-- Select Column --':
            st.error("❌ Please select Datetime and Power columns!")
        else:
            with st.spinner("🔄 Processing your campus energy data..."):
                processed_df, error = process_uploaded_data(raw_df, datetime_col, power_col, sub1, sub2, sub3)
                if error:
                    st.error(error)
                else:
                    st.session_state.processed_data = processed_df
                    st.success(f"✅ Processed {len(processed_df):,} records!")
                    st.balloons()
                    st.rerun()

elif st.session_state.processed_data is None:
    st.markdown("""
    <div class="info-box">
    <h3>👋 Welcome to Campus Energy Waste Detector!</h3>
    <p><b>Step 1:</b> Upload your campus energy CSV/Excel in the sidebar<br>
    <b>Step 2:</b> Map your columns to our format<br>
    <b>Step 3:</b> Click "Process Data" to get instant AI-powered insights!</p>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("📋 Expected Data Format")
    sample = pd.DataFrame({
        'Timestamp': ['2024-01-01 00:00:00', '2024-01-01 01:00:00'],
        'Power_kW': [1.5, 1.2],
        'Labs_kW': [0.3, 0.2],
        'Classrooms_kW': [0.5, 0.3],
        'HVAC_kW': [0.7, 0.7],
    })
    st.dataframe(sample, use_container_width=True)
    if st.button("🎮 Try Demo Data (90-Day Campus Simulation)", type="secondary"):
        dates = pd.date_range('2024-01-01', periods=2160, freq='H')
        np.random.seed(42)
        base = 50 + 30 * np.sin(np.arange(2160) * 2 * np.pi / 24)
        demo = pd.DataFrame({
            'Power': base + np.random.normal(0, 5, 2160),
            'Sub1': np.random.uniform(5, 15, 2160),
            'Sub2': np.random.uniform(5, 15, 2160),
            'Sub3': np.random.uniform(20, 40, 2160),
        }, index=dates)
        demo['Hour'] = demo.index.hour
        demo['DayOfWeek'] = demo.index.dayofweek
        demo['IsWeekend'] = (demo['DayOfWeek'] >= 5).astype(int)
        demo['Month'] = demo.index.month
        mean_p = demo['Power'].mean()
        std_p = demo['Power'].std()
        demo['is_anomaly'] = ((demo['Power'] > mean_p + 1.2 * std_p) |
                              ((demo['Hour'] < 6) & (demo['Power'] > mean_p))).astype(int)
        demo['waste_reason'] = 'Normal'
        demo.loc[demo['Power'] > mean_p + 1.2 * std_p, 'waste_reason'] = 'Statistically High'
        demo.loc[(demo['Hour'] < 6) & (demo['Power'] > mean_p), 'waste_reason'] = 'Night-time Waste'
        demo.loc[(demo['IsWeekend'] == 1) & (demo['Power'] > mean_p * 0.9), 'waste_reason'] = 'Weekend Waste'
        st.session_state.processed_data = demo
        st.rerun()

# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
if st.session_state.processed_data is not None:
    df = st.session_state.processed_data

    st.sidebar.markdown("---")
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.radio("", ["📈 Overview", "🚨 Waste Analysis", "🔮 Predictions", "💰 Savings"])

    mean_power = df['Power'].mean()
    std_power = df['Power'].std()
    n_anomalies = df['is_anomaly'].sum()
    anomaly_pct = (n_anomalies / len(df)) * 100
    grade, grade_label, grade_color = get_energy_grade(anomaly_pct)

    # ── OVERVIEW ──────────────────────────────
    if page == "📈 Overview":
        st.header("📊 Campus Energy Consumption Overview")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📅 Period", f"{(df.index.max() - df.index.min()).days} days")
        col2.metric("📊 Avg Consumption", f"{mean_power:.2f} kW")
        col3.metric("🚨 Waste Events", f"{n_anomalies:,}", f"{anomaly_pct:.1f}% of time")
        col4.metric("🔥 Peak Load", f"{df['Power'].max():.2f} kW")
        col5.metric("📉 Min Load", f"{df['Power'].min():.2f} kW")

        st.markdown("---")

        # Energy grade card + stats summary side by side
        col_grade, col_stats = st.columns([1, 2])

        with col_grade:
            st.markdown(f"""
            <div style="text-align:center; padding:20px; background:#f8f9fa; border-radius:12px; border: 2px solid {grade_color};">
                <div style="font-size:1rem; color:#555; margin-bottom:5px;">🏫 Campus Energy Grade</div>
                <div style="font-size:5rem; font-weight:bold; color:{grade_color}; line-height:1.1">{grade}</div>
                <div style="font-size:1.2rem; color:{grade_color}; font-weight:600">{grade_label}</div>
                <div style="font-size:0.85rem; color:#888; margin-top:8px">{anomaly_pct:.1f}% wasteful hours</div>
            </div>
            """, unsafe_allow_html=True)

        with col_stats:
            s1, s2, s3 = st.columns(3)
            s1.metric("📊 Std Deviation", f"{std_power:.2f} kW")
            s2.metric("📐 Median Load", f"{df['Power'].median():.2f} kW")
            s3.metric("📈 Load Factor", f"{(mean_power / df['Power'].max() * 100):.1f}%")

            s4, s5, s6 = st.columns(3)
            night_avg = df[df['Hour'].between(0, 5)]['Power'].mean()
            day_avg = df[df['Hour'].between(8, 18)]['Power'].mean()
            weekend_avg = df[df['IsWeekend'] == 1]['Power'].mean()
            s4.metric("🌙 Night Avg", f"{night_avg:.2f} kW")
            s5.metric("☀️ Day Avg", f"{day_avg:.2f} kW")
            s6.metric("🏖️ Weekend Avg", f"{weekend_avg:.2f} kW")

        st.markdown("---")

        # Main trend chart
        plot_data = df.tail(min(1000, len(df)))
        anomalies = plot_data[plot_data['is_anomaly'] == 1]
        normals = plot_data[plot_data['is_anomaly'] == 0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=normals.index, y=normals['Power'], mode='lines', name='Normal',
                                 line=dict(color='steelblue', width=1.5)))
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Power'], mode='markers', name='🚨 Waste Event',
                                 marker=dict(color='red', size=5, symbol='circle')))
        fig.add_hline(y=mean_power, line_dash="dot", line_color="green",
                      annotation_text=f"Campus Average: {mean_power:.1f} kW", annotation_position="top left")
        fig.update_layout(title="Campus Power Consumption Trend", height=380,
                          xaxis_title="Time", yaxis_title="Power (kW)",
                          legend=dict(orientation='h', yanchor='bottom', y=1.02))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        col1, col2, col3 = st.columns(3)

        with col1:
            hourly = df.groupby('Hour')['Power'].mean()
            colors = ['#dc3545' if (h < 6 or h > 21) else '#ffc107' if h > 18 else '#28a745' for h in hourly.index]
            fig = go.Figure(go.Bar(x=hourly.index, y=hourly.values, marker_color=colors))
            fig.update_layout(title="⏰ Avg Load by Hour of Day", xaxis_title="Hour", yaxis_title="kW", height=300)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with col2:
            weekly = df.groupby('DayOfWeek')['Power'].mean()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            day_colors = ['#6c757d' if i >= 5 else '#2196f3' for i in range(7)]
            fig = go.Figure(go.Bar(x=days, y=weekly.values, marker_color=day_colors))
            fig.update_layout(title="📅 Avg Load by Day of Week", xaxis_title="Day", yaxis_title="kW", height=300)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with col3:
            # Sub-metering breakdown (if non-zero)
            sub_totals = {
                'Labs / Canteen': df['Sub1'].mean(),
                'Classrooms / Offices': df['Sub2'].mean(),
                'HVAC / Common Areas': df['Sub3'].mean(),
                'Other / Unmetered': max(0, mean_power - df['Sub1'].mean() - df['Sub2'].mean() - df['Sub3'].mean())
            }
            sub_totals = {k: v for k, v in sub_totals.items() if v > 0}
            if sum(sub_totals.values()) > 0.5:
                fig = px.pie(values=list(sub_totals.values()), names=list(sub_totals.keys()),
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_traces(textposition='inside', textinfo='label+percent')
                fig.update_layout(title="🏫 Campus Zone Breakdown", height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.markdown("""
                <div class="info-box" style="margin-top:40px">
                <b>🏫 Zone Breakdown</b><br>
                Map sub-metering columns to see which campus zone consumes the most energy.
                </div>
                """, unsafe_allow_html=True)

        # Monthly trend if data spans >1 month
        if df['Month'].nunique() > 1:
            st.markdown("---")
            monthly = df.groupby('Month')['Power'].mean()
            month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                           7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
            fig = go.Figure(go.Bar(
                x=[month_names[m] for m in monthly.index],
                y=monthly.values,
                marker_color='#2196f3',
                text=[f"{v:.1f}" for v in monthly.values],
                textposition='outside'
            ))
            fig.update_layout(title="📆 Monthly Average Consumption (Campus-Wide)", height=300,
                              yaxis_title="kW", xaxis_title="Month")
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ── WASTE ANALYSIS ────────────────────────
    elif page == "🚨 Waste Analysis":
        st.header("🚨 Campus Waste Detection Analysis")

        waste_df = df[df['is_anomaly'] == 1]
        normal_df = df[df['is_anomaly'] == 0]

        if len(waste_df) == 0:
            st.success("✅ No significant waste detected in your campus data!")
        else:
            waste_avg = waste_df['Power'].mean()
            normal_avg = normal_df['Power'].mean()
            excess_per_hour = waste_avg - normal_avg
            total_excess_kwh = excess_per_hour * n_anomalies

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🚨 Waste Events", f"{len(waste_df):,}")
            col2.metric("📊 % of Campus Hours", f"{anomaly_pct:.1f}%")
            col3.metric("⚡ Avg Excess Load", f"+{excess_per_hour:.2f} kW")
            col4.metric("🔋 Total Excess Energy", f"{total_excess_kwh:.0f} kWh")

            st.markdown("---")

            # ── Row 1: Heatmap + Waste vs Normal comparison
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🗓️ Waste Heatmap — Hour vs Day")
                heatmap_data = df.groupby(['DayOfWeek', 'Hour'])['is_anomaly'].mean().unstack(fill_value=0)
                days_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                fig = go.Figure(go.Heatmap(
                    z=heatmap_data.values,
                    x=[f"{h:02d}:00" for h in heatmap_data.columns],
                    y=[days_labels[i] for i in heatmap_data.index],
                    colorscale='Reds',
                    colorbar=dict(title='Waste Rate'),
                    hovertemplate="Day: %{y}<br>Hour: %{x}<br>Waste Rate: %{z:.1%}<extra></extra>"
                ))
                fig.update_layout(title="Darker = More Waste", height=300,
                                  xaxis_title="Hour of Day", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.caption("💡 Use this to identify structural waste patterns — e.g. labs left on overnight, AC running on weekends.")

            with col2:
                st.subheader("📊 Waste vs Normal — Load Distribution")
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=normal_df['Power'], name='Normal Hours', nbinsx=40,
                                           marker_color='steelblue', opacity=0.7))
                fig.add_trace(go.Histogram(x=waste_df['Power'], name='Waste Hours', nbinsx=40,
                                           marker_color='red', opacity=0.7))
                fig.update_layout(barmode='overlay', height=300,
                                  xaxis_title="Power (kW)", yaxis_title="Count",
                                  legend=dict(orientation='h'))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.caption("💡 The further the red distribution is from blue, the more severe your campus waste problem is.")

            st.markdown("---")

            # ── Row 2: Waste by Day + Waste by Hour
            col1, col2 = st.columns(2)

            with col1:
                by_day = waste_df.groupby('DayOfWeek').size()
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                counts = [by_day.get(i, 0) for i in range(7)]
                colors = ['#dc3545' if c == max(counts) else '#ffc107' if c > np.mean(counts) else '#28a745' for c in counts]
                fig = go.Figure(go.Bar(x=days, y=counts, marker_color=colors,
                                       text=counts, textposition='outside'))
                fig.update_layout(title="🗓️ Waste Events by Day of Week", height=300,
                                  xaxis_title="", yaxis_title="Waste Events")
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            with col2:
                by_hour = waste_df.groupby('Hour').size()
                hour_counts = [by_hour.get(h, 0) for h in range(24)]
                colors_h = ['#dc3545' if c == max(hour_counts) else '#ffc107' if c > np.mean(hour_counts) else '#28a745' for c in hour_counts]
                fig = go.Figure(go.Bar(x=list(range(24)), y=hour_counts, marker_color=colors_h))
                fig.update_layout(title="⏰ Waste Events by Hour of Day", height=300,
                                  xaxis_title="Hour", yaxis_title="Waste Events")
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # ── Row 3: Waste trend + Reason breakdown
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📈 Waste Trend Over Time")
                if df['Month'].nunique() > 1:
                    trend = df.groupby(df.index.to_period('W').start_time)['is_anomaly'].mean() * 100
                else:
                    trend = df.groupby(df.index.to_period('D').start_time)['is_anomaly'].mean() * 100
                fig = go.Figure(go.Scatter(x=trend.index, y=trend.values, fill='tozeroy',
                                           line=dict(color='#dc3545', width=2)))
                fig.add_hline(y=anomaly_pct, line_dash="dot", line_color="orange",
                              annotation_text=f"Overall avg: {anomaly_pct:.1f}%")
                fig.update_layout(height=280, yaxis_title="% Waste Hours", xaxis_title="")
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.caption("💡 An upward trend means your campus waste problem is worsening — act now!")

            with col2:
                st.subheader("🔍 Waste Reason Breakdown")
                if 'waste_reason' in waste_df.columns:
                    reason_counts = waste_df['waste_reason'].value_counts()
                    fig = px.pie(values=reason_counts.values, names=reason_counts.index,
                                 color_discrete_sequence=['#dc3545', '#ff7043', '#ffc107'])
                    fig.update_traces(textposition='inside', textinfo='label+percent')
                    fig.update_layout(height=280, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    st.caption("💡 Night-time and weekend waste are the easiest wins — often just scheduling issues.")

            # ── Top worst events table
            st.markdown("---")
            st.subheader("📋 Top 15 Worst Campus Waste Events")
            top_waste = (waste_df[['Power', 'Hour', 'DayOfWeek', 'waste_reason']]
                         .copy()
                         .assign(Excess_kW=lambda x: x['Power'] - normal_avg)
                         .sort_values('Excess_kW', ascending=False)
                         .head(15))
            top_waste['Day'] = top_waste['DayOfWeek'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
            top_waste['Time'] = top_waste['Hour'].apply(lambda h: f"{h:02d}:00")
            top_waste['Power (kW)'] = top_waste['Power'].round(2)
            top_waste['Excess (kW)'] = top_waste['Excess_kW'].round(2)
            top_waste = top_waste[['Day', 'Time', 'Power (kW)', 'Excess (kW)', 'waste_reason']].rename(columns={'waste_reason': 'Reason'})
            top_waste.index = top_waste.index.strftime('%Y-%m-%d')
            st.dataframe(top_waste, use_container_width=True)

            st.markdown("---")
            st.markdown(f"""
            <div class="danger-box">
            <b>🚨 Campus Waste Summary</b><br><br>
            Your campus recorded <b>{n_anomalies:,} waste events</b> ({anomaly_pct:.1f}% of all monitored hours),
            consuming an average of <b>{waste_avg:.2f} kW</b> during those periods vs a normal average of
            <b>{normal_avg:.2f} kW</b> — that's <b>{((waste_avg/normal_avg-1)*100):.0f}% excess load</b>.<br><br>
            <b>Total estimated excess consumption: {total_excess_kwh:.0f} kWh</b> over the recorded period.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="success-box">
            <b>💡 Recommended Campus Actions:</b><br>
            • <b>Night-time waste:</b> Auto-shutoff schedules for labs, ACs, and lighting after 10 PM<br>
            • <b>Weekend waste:</b> Smart building management system (BMS) to reduce load when campus is empty<br>
            • <b>Statistically high events:</b> Investigate equipment running simultaneously — stagger start times<br>
            • <b>Long-term:</b> Install occupancy sensors in classrooms and corridors for automatic control<br>
            • <b>Quick win:</b> Send weekly energy waste reports to department heads to drive accountability
            </div>
            """, unsafe_allow_html=True)

    # ── PREDICTIONS ───────────────────────────
    elif page == "🔮 Predictions":
        st.header("🔮 24-Hour Campus Consumption Forecast")

        st.markdown("""
        <div class="info-box">
        <b>📡 How this forecast works:</b> We use your historical hourly consumption pattern (by hour of day)
        to project the next 24 hours, with ±5% natural variation. This helps campus facility managers
        proactively plan load management and identify upcoming high-risk periods before they happen.
        </div>
        """, unsafe_allow_html=True)

        hourly_pattern = df.groupby('Hour')['Power'].mean()
        np.random.seed(7)
        predictions = [hourly_pattern[h % 24] * np.random.uniform(0.95, 1.05) for h in range(24)]
        upper_bound = [p * 1.12 for p in predictions]
        lower_bound = [p * 0.88 for p in predictions]

        last_24 = df.tail(24)['Power'].values if len(df) >= 24 else df['Power'].values

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = go.Figure()
            # Confidence band
            fig.add_trace(go.Scatter(
                x=list(range(24)) + list(range(23, -1, -1)),
                y=upper_bound + lower_bound[::-1],
                fill='toself', fillcolor='rgba(255,165,0,0.15)',
                line=dict(color='rgba(0,0,0,0)'), name='Forecast Range', showlegend=True
            ))
            fig.add_trace(go.Scatter(
                x=list(range(24)), y=last_24, name='Last 24h (Actual)',
                mode='lines+markers', line=dict(color='steelblue', width=2),
                marker=dict(size=5)
            ))
            fig.add_trace(go.Scatter(
                x=list(range(24)), y=predictions, name='Forecast',
                mode='lines+markers', line=dict(color='orange', width=2.5, dash='dash'),
                marker=dict(size=6, symbol='diamond')
            ))
            fig.add_hline(y=mean_power, line_dash="dot", line_color="green",
                          annotation_text=f"Campus avg: {mean_power:.1f} kW")
            fig.add_hline(y=mean_power * 1.3, line_dash="dot", line_color="red",
                          annotation_text="⚠️ High alert threshold")
            fig.update_layout(
                title="Next 24-Hour Campus Load Forecast",
                height=420, xaxis_title="Hour of Day", yaxis_title="Power (kW)",
                xaxis=dict(tickmode='linear', tick0=0, dtick=2,
                           ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
                           tickvals=list(range(0, 24, 2))),
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with col2:
            st.subheader("📊 Forecast Summary")
            st.metric("📦 Total Predicted", f"{sum(predictions):.1f} kWh")
            st.metric("📊 Predicted Avg", f"{np.mean(predictions):.2f} kW")
            st.metric("🔥 Expected Peak", f"{max(predictions):.2f} kW at {np.argmax(predictions):02d}:00")
            st.metric("🌙 Night Low", f"{min(predictions):.2f} kW at {np.argmin(predictions):02d}:00")

            peak_excess = max(predictions) - mean_power
            if max(predictions) > mean_power * 1.3:
                st.markdown(f"""
                <div class="danger-box">
                <b>⚠️ High Load Warning!</b><br>
                Peak expected at <b>{np.argmax(predictions):02d}:00</b>.<br>
                Pre-schedule non-critical loads for off-peak hours.
                </div>
                """, unsafe_allow_html=True)
            elif max(predictions) > mean_power * 1.15:
                st.markdown("""
                <div class="warning-box">
                <b>🟡 Moderate Load Day</b><br>
                Some hours above average — monitor HVAC and lab usage.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                <b>✅ Normal Pattern Expected</b><br>
                No high-risk periods forecast. Good day for energy-intensive maintenance work.
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Hourly risk table
        st.subheader("🕐 Hourly Campus Load Forecast & Risk Level")

        risk_data = []
        for h, p in enumerate(predictions):
            if p > mean_power * 1.3:
                risk = "🔴 High"
                action = "Stagger equipment start times, reduce non-essential loads"
            elif p > mean_power * 1.1:
                risk = "🟡 Medium"
                action = "Monitor HVAC and lab equipment closely"
            elif p < mean_power * 0.7:
                risk = "🔵 Low"
                action = "Good time to run maintenance or test equipment"
            else:
                risk = "🟢 Normal"
                action = "—"
            risk_data.append({
                'Hour': f"{h:02d}:00",
                'Forecast (kW)': round(p, 2),
                'vs Average': f"{((p/mean_power - 1)*100):+.1f}%",
                'Risk Level': risk,
                'Recommended Action': action
            })

        risk_df = pd.DataFrame(risk_data)
        st.dataframe(risk_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        col1, col2 = st.columns(2)
        high_hrs = [(i, p) for i, p in enumerate(predictions) if p > mean_power * 1.2]
        low_hrs = [(i, p) for i, p in enumerate(predictions) if p < mean_power * 0.8]

        with col1:
            st.markdown("### ⚠️ High-Risk Hours — Campus Alert")
            if high_hrs:
                for h, p in high_hrs[:6]:
                    excess = p - mean_power
                    st.warning(f"**{h:02d}:00** → {p:.2f} kW (+{excess:.2f} kW above avg)")
            else:
                st.success("✅ No high-risk hours expected tomorrow.")

        with col2:
            st.markdown("### 💚 Optimal Hours — Schedule Heavy Loads Here")
            if low_hrs:
                for h, p in low_hrs[:6]:
                    saving = mean_power - p
                    st.info(f"**{h:02d}:00** → {p:.2f} kW (−{saving:.2f} kW below avg)")
            else:
                st.info("All hours within normal range.")

        st.markdown("""
        <div class="metric-box">
        <b>🎯 Campus Facility Manager Tips:</b><br>
        • Schedule EV charging, laundry, and non-critical equipment during 💚 low-risk hours<br>
        • Pre-cool campus buildings 1–2 hours before the daily peak to reduce peak HVAC demand<br>
        • Use the forecast to plan generator or backup system tests on low-demand days<br>
        • Share the high-risk alerts with department heads to voluntarily reduce load
        </div>
        """, unsafe_allow_html=True)

    # ── SAVINGS ───────────────────────────────
    elif page == "💰 Savings":
        st.header("💰 Campus Energy Savings Calculator")

        st.markdown("""
        <div class="info-box">
        <b>📊 Estimate the financial, environmental and social impact of reducing energy waste across your campus.
        All savings are calculated from your actual uploaded consumption data.</b>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏫 Campus Profile")
            campus_students = st.number_input("Total Students Enrolled", 100, 100000, 5000, 100)
            campus_faculty = st.number_input("Faculty & Staff", 10, 10000, 500, 50)
            campus_buildings = st.number_input("Number of Campus Buildings", 1, 200, 15, 1)
            rate = st.number_input("Electricity Tariff (₹/kWh)", 1.0, 20.0, 6.5, 0.25)

        with col2:
            st.subheader("🎯 Reduction Plan")
            reduction = st.slider("Waste Reduction Target %", 5, 100, 60, 5)
            investment = st.number_input("Implementation Budget (₹)", 0, 50000000, 500000, 50000)
            timeline = st.selectbox("Implementation Timeline",
                                    ["Immediate (1 month)", "Short-term (3 months)",
                                     "Medium-term (6 months)", "Long-term (12 months)"])

        st.markdown("---")

        if n_anomalies > 0:
            waste_avg = df[df['is_anomaly'] == 1]['Power'].mean()
            normal_avg = df[df['is_anomaly'] == 0]['Power'].mean()

            # Excess energy from the sample data for one building equivalent
            excess_per_event_kw = waste_avg - normal_avg
            sample_excess_kwh = excess_per_event_kw * n_anomalies

            # Scale to all campus buildings
            campus_excess_kwh = sample_excess_kwh * campus_buildings

            # Apply reduction target
            reducible_kwh = campus_excess_kwh * (reduction / 100)

            hours_in_data = len(df)
            days_in_data = max(1, hours_in_data / 24)

            daily_savings_kwh = reducible_kwh / days_in_data
            monthly_savings_kwh = daily_savings_kwh * 30
            annual_savings_kwh = daily_savings_kwh * 365

            daily_cost = daily_savings_kwh * rate
            monthly_cost = monthly_savings_kwh * rate
            annual_cost = annual_savings_kwh * rate

            roi_months = investment / monthly_cost if monthly_cost > 0 else 999

            # ── Top metrics
            st.subheader("📊 Projected Campus-Wide Savings")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💵 Daily Savings", f"₹{daily_cost:,.0f}")
            col2.metric("💵 Monthly Savings", f"₹{monthly_cost:,.0f}")
            col3.metric("💵 Annual Savings", f"₹{annual_cost:,.0f}")
            col4.metric("⏱️ ROI Period", f"{roi_months:.1f} months")

            # ── Per-student breakdown
            st.markdown("---")
            st.subheader("👩‍🎓 Per-Student & Per-Staff Impact")
            total_people = campus_students + campus_faculty

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💰 Saving per Student/yr", f"₹{annual_cost / campus_students:,.0f}")
            col2.metric("👥 Total Campus People", f"{total_people:,}")
            col3.metric("⚡ kWh Saved per Person", f"{annual_savings_kwh / total_people:.0f} kWh/yr")
            col4.metric("📉 Waste Reduction", f"{reduction}% of excess")

            st.markdown(f"""
            <div class="info-box">
            <b>🎓 What ₹{annual_cost/campus_students:,.0f} per student means:</b> This saving could fund campus
            Wi-Fi upgrades, library resources, or student welfare programmes — reinvesting energy savings
            directly into the student experience.
            </div>
            """, unsafe_allow_html=True)

            # ── Savings trajectory chart
            st.markdown("---")
            st.subheader("📈 12-Month Cumulative Savings Trajectory")

            timeline_months = {'Immediate (1 month)': 1, 'Short-term (3 months)': 3,
                               'Medium-term (6 months)': 6, 'Long-term (12 months)': 12}
            ramp_months = timeline_months.get(timeline, 3)

            months = list(range(1, 13))
            cumulative = []
            for m in months:
                if m <= ramp_months:
                    ramp_factor = m / ramp_months
                else:
                    ramp_factor = 1.0
                cumulative.append(monthly_cost * ramp_factor * m if m == 1 else
                                  cumulative[-1] + monthly_cost * ramp_factor)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=months, y=cumulative, fill='tozeroy', name='Cumulative Savings',
                                     line=dict(color='#28a745', width=3)))
            fig.add_hline(y=investment, line_dash="dash", line_color="red",
                          annotation_text=f"Budget: ₹{investment:,}", annotation_position="top left")
            if roi_months <= 12:
                fig.add_vline(x=roi_months, line_dash="dot", line_color="blue",
                              annotation_text=f"Break-even: Month {roi_months:.1f}")
            fig.update_layout(title="Savings grow as implementation matures", height=380,
                              xaxis_title="Month", yaxis_title="Cumulative Savings (₹)",
                              xaxis=dict(tickmode='linear', tick0=1, dtick=1))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # ── Implementation breakdown
            st.markdown("---")
            st.subheader("💡 Recommended Campus Implementation Plan")

            col1, col2 = st.columns(2)

            with col1:
                categories = {
                    'Smart HVAC & AC Controls': (0.35, 0.40),
                    'Classroom Motion Sensors': (0.25, 0.25),
                    'Real-time Energy Monitoring': (0.15, 0.15),
                    'Faculty & Student Training': (0.10, 0.10),
                    'Awareness Campaigns': (0.15, 0.10),
                }
                for action, (budget_pct, savings_pct) in categories.items():
                    b = investment * budget_pct
                    s = annual_cost * savings_pct
                    st.markdown(f"**{action}**")
                    st.caption(f"Budget: ₹{b:,.0f} &nbsp;|&nbsp; Annual Impact: ₹{s:,.0f}")
                    st.progress(int(budget_pct * 100))

            with col2:
                savings_by_zone = {
                    'HVAC & AC (40%)': annual_cost * 0.40,
                    'Lighting (20%)': annual_cost * 0.20,
                    'Labs & Equipment (25%)': annual_cost * 0.25,
                    'Hostels & Canteen (15%)': annual_cost * 0.15,
                }
                fig = px.pie(values=list(savings_by_zone.values()),
                             names=list(savings_by_zone.keys()),
                             color_discrete_sequence=px.colors.sequential.Greens_r)
                fig.update_traces(textposition='inside', textinfo='label+percent')
                fig.update_layout(title="Savings by Campus Zone", height=340, showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # ── Environmental impact
            st.markdown("---")
            st.subheader("🌍 Environmental & Social Impact")

            co2_kg = annual_savings_kwh * 0.82  # India grid emission factor
            trees = co2_kg / 20
            cars_off = co2_kg / 4600
            homes_powered = annual_savings_kwh / 1200  # avg Indian home ~1200 kWh/yr

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🌱 CO₂ Avoided", f"{co2_kg/1000:.1f} tonnes")
            col2.metric("🌳 Trees Equivalent", f"{trees:,.0f}")
            col3.metric("🚗 Cars Off Road", f"{cars_off:.1f}/yr")
            col4.metric("🏠 Homes Powered", f"{homes_powered:.0f}")

            st.markdown(f"""
            <div class="success-box">
            <h3>🎯 Campus Sustainability Impact Summary</h3>

            <b>🏫 Campus Scale:</b> {campus_buildings} buildings | {campus_students:,} students | {campus_faculty:,} staff<br>
            <b>🎯 Target:</b> {reduction}% waste reduction via {timeline.lower()}<br><br>

            <b>💰 Financial Impact:</b><br>
            • Annual campus savings: <b>₹{annual_cost:,.0f}</b><br>
            • Per-student savings: <b>₹{annual_cost/campus_students:,.0f}/year</b> — can be redirected to campus development<br>
            • Implementation ROI in: <b>{roi_months:.1f} months</b><br><br>

            <b>🌍 Environmental Impact:</b><br>
            • Avoid <b>{co2_kg/1000:.1f} tonnes</b> of CO₂ annually (India grid: 0.82 kg/kWh)<br>
            • Equivalent to planting <b>{trees:,.0f} trees</b> or removing <b>{cars_off:.1f} cars</b> from roads<br>
            • Power <b>{homes_powered:.0f} Indian homes</b> with the saved energy<br><br>

            <b>🏆 Recognition & Compliance:</b><br>
            • Eligible for <b>Green Campus Certification</b> (Bureau of Energy Efficiency, India)<br>
            • Aligns with <b>UN SDG #7</b> (Clean Energy) and <b>SDG #13</b> (Climate Action)<br>
            • Improves NIRF and QS sustainability rankings<br>
            • Contributes to India's <b>Net Zero 2070</b> national commitment
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="success-box">
            <b>✅ Excellent Campus Energy Management!</b><br><br>
            No significant waste detected. Your campus is already operating efficiently!<br><br>
            <b>Next Steps:</b><br>
            • Continue monitoring to sustain this performance<br>
            • Upload more data (3–12 months) for deeper seasonal analysis<br>
            • Implement predictive maintenance to prevent future waste<br>
            • Consider applying for Green Campus Certification
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#666;font-size:0.85rem">⚡ Campus Energy Waste Detector | India AI Buildathon 2026</div>', unsafe_allow_html=True)