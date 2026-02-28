# ⚡ Campus Energy Waste Detection System

> AI that watches your campus energy 24/7 and tells you exactly where you're wasting money.
> Built for **AMD Slingshot Hackathon**.

🔗 **[Live Demo → WattWise](https://rak2315.github.io/Campus-Energy-Waste-Detection-System/#models)**

---

## 🤔 What Problem Does This Solve?

Most university campuses have no idea how much energy they waste. Labs left on overnight. Air conditioning running all weekend when no one's there. Equipment never switched off.

The result? Massive electricity bills, unnecessary carbon emissions, and money that could go toward students instead going to the power company.

**This tool fixes that.** Upload your campus energy data, and within seconds our ML models will tell you:
- Exactly which hours are wasteful and why
- How much it's costing you per student per year
- What the next 24 hours will look like so you can plan ahead
- How much CO₂ you're unnecessarily pumping into the atmosphere

---

## 🚀 How It Works (Simple Version)

1. **Upload** your campus energy CSV file
2. **Two ML models run automatically:**
   - *Isolation Forest* spots the unusual hours — the ones where energy is being wasted
   - *Random Forest* predicts what the next 24 hours will look like
3. **You get a full dashboard** showing waste patterns, savings potential, and actionable recommendations

No coding needed. Works on any energy dataset regardless of scale.

---

## 🧠 The ML Behind It

### Anomaly Detection — Isolation Forest
Think of it like this: the model learns what "normal" energy usage looks like for your campus. Anything that doesn't fit — a late-night spike, weekend overconsumption, unusually high load — gets flagged as waste. Every hour gets a **risk score from 0 to 100**, so you don't just get a yes/no answer, you get a severity rating.

- **Automatically retrains on whatever data you upload** — adapts to any campus, any scale
- Flags ~5% of hours as anomalies
- Detects night-time waste, weekend waste, and statistically abnormal consumption separately

### Forecasting — Random Forest Regressor
Trained on 34,421 hours of real energy data. Uses what happened 1 hour ago, 24 hours ago, and 1 week ago to predict the next 24 hours. Also uses time-of-day and day-of-week patterns encoded as cyclical features.

| Metric | Score | What It Means |
|--------|-------|---------------|
| R² | **0.897** | Model explains 89.7% of all consumption variance |
| RMSE | 0.234 kW | Average prediction error per hour |
| vs. naive baseline | **70% better** | Beats "same time yesterday" by 70% |

Key insight from training: **HVAC accounts for 49% of all consumption** — controlling air conditioning is the single biggest lever for any campus.

---

## 📊 Dashboard Pages

| Page | What You Get |
|------|-------------|
| 📈 **Overview** | Energy grade (A–F), load stats, hourly/daily/monthly patterns, campus zone breakdown |
| 🚨 **Waste Analysis** | Heatmap of worst hours, risk score scatter, top 15 worst events table, waste trend over time |
| 🔮 **Predictions** | 24h RF forecast with confidence band, hourly risk table, high-alert warnings |
| 💰 **Savings** | Per-student savings, ROI calculator, CO₂ impact, 12-month trajectory, implementation plan |

---

## 💰 Example Impact (Real Data)

On a 15-building campus with 5,000 students using real energy data:

- **5% of hours** flagged as wasteful by Isolation Forest
- **203% excess load** during those waste events vs normal hours
- **3,456 kWh** total excess detected over the recorded period
- Savings scale linearly with campus size — bigger campus, bigger numbers

---

## 🛠️ Run It Yourself

### 1. Clone
```bash
git clone https://github.com/RAK2315/Campus-Energy-Waste-Detection-System.git
cd Campus-Energy-Waste-Detection-System
```

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Click **"Try Demo Data"** to explore without uploading anything.

---

## 📦 requirements.txt

```
streamlit
pandas
numpy
plotly
scikit-learn
joblib
openpyxl
```

---

## 📂 Project Structure

```
├── app.py                             # Full Streamlit application
├── models/
│   ├── rf_forecaster.pkl              # Trained Random Forest (R²=0.897)
│   ├── isolation_forest_anomaly.pkl   # Trained Isolation Forest
│   └── feature_info.json             # Feature metadata
├── requirements.txt
└── README.md
```

---

## 📋 Supported Data Format

Minimum required — just two columns:

| Column | What It Is | Example |
|--------|-----------|---------|
| Timestamp | Date and time of reading | `2024-01-01 08:00:00` |
| Power | Consumption in kW | `45.3` |

Optional sub-metering columns for zone breakdown:

| Column | Campus Zone | Example |
|--------|------------|---------|
| Zone 1 | Labs / Canteen | `12.1` |
| Zone 2 | Classrooms / Offices | `8.4` |
| Zone 3 | HVAC / Common Areas | `24.8` |

Supports most datetime formats automatically. Auto-resamples to hourly if data is more granular.

---

## 🌍 Why This Matters

- India's grid emits **0.82 kg CO₂ per kWh** — higher than most developed nations
- University campuses are among the largest institutional energy consumers in India
- Most campuses have zero visibility into when and where waste happens
- BEE (Bureau of Energy Efficiency) has a **Green Campus certification** that rewards exactly this kind of monitoring
- Every rupee saved on energy can be reinvested into student resources
- Directly supports India's **Net Zero 2070** commitment and **UN SDG #7 & #13**

---

## 🏆 AMD Slingshot Hackathon

Built to make energy intelligence accessible to every campus in India — not just the ones that can afford enterprise software.

---

## 📄 License

MIT — free to use, modify, and deploy.