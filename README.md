# ⚡ Campus Energy Waste Detector
> AI-powered energy monitoring and waste detection for smarter, greener campuses.
> Built for **India AI Buildathon 2026**.

---
## 🚀 Live Demo
🔗 **[campus-energy-waste-detection-system.streamlit.app](https://campus-energy-waste-detection-system.streamlit.app/)**

Or click **"Try Demo Data"** inside the app to explore instantly without uploading anything.

---

## 📌 What It Does

Most campuses have no idea how much energy they waste — lights left on at night, ACs running on weekends, labs never turned off. This tool fixes that.

Upload your campus energy CSV and get:
- **Automatic waste event detection** using statistical + time-based analysis
- **Interactive dashboards** for consumption trends, heatmaps, and patterns
- **24-hour forecasts** with hourly risk levels and facility manager recommendations
- **Savings calculator** showing financial, environmental, and per-student impact

---

## 🖥️ Features

| Page | What You Get |
|------|-------------|
| 📈 **Overview** | Energy grade (A–F), load stats, hourly/daily/monthly patterns, zone breakdown |
| 🚨 **Waste Analysis** | Heatmap, worst events table, waste trend, reason breakdown, recommendations |
| 🔮 **Predictions** | 24h forecast with confidence bands, hourly risk table, high-alert warnings |
| 💰 **Savings** | Per-student savings, ROI calculator, CO₂ impact, campus sustainability report |

---

## 📂 Project Structure

```
campus-energy-detector/
│
├── app.py               # Main Streamlit application
├── README.md            # This file
└── requirements.txt     # Python dependencies
```

---

## ⚙️ Installation & Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/campus-energy-detector.git
cd campus-energy-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Requirements

Create a `requirements.txt` with:
```
streamlit
pandas
numpy
plotly
openpyxl
```

---

## 📊 Data Format

Upload a CSV or Excel file with at least:

| Column | Description | Example |
|--------|-------------|---------|
| Timestamp | Date and time of reading | `2024-01-01 08:00:00` |
| Power | Consumption in kW | `45.3` |
| Zone 1 *(optional)* | Labs / Canteen sub-meter | `12.1` |
| Zone 2 *(optional)* | Classrooms / Offices | `8.4` |
| Zone 3 *(optional)* | HVAC / Common Areas | `24.8` |

Supports most datetime formats. Automatically resamples to hourly if data is more granular.

---

## 🧠 How Waste Detection Works

Three detection methods run simultaneously:

1. **Statistical** — flags readings more than 1.2 standard deviations above the mean
2. **Night-time** (12 AM – 6 AM) — flags consumption above the 60th percentile during off-hours
3. **Weekend** — flags consumption above the 70th percentile on Saturdays and Sundays

Each waste event is tagged with a reason so you know exactly what to fix.

---

## 🌍 Impact Metrics

The savings calculator uses:
- **India grid emission factor:** 0.82 kg CO₂ per kWh
- **Average Indian household:** ~1,200 kWh/year
- **Average car emissions:** ~4,600 kg CO₂/year

Savings scale by number of **students, staff, and campus buildings**.

---

## 🏆 Alignment

- 🇮🇳 AMD Slingshot Hackathon 2026
- 🌱 UN SDG #7 — Affordable & Clean Energy  
- 🌡️ UN SDG #13 — Climate Action  
- 📋 Bureau of Energy Efficiency (BEE), India — Green Campus guidelines
- 🎯 India Net Zero 2070 commitment

---

## 📄 License

MIT License — free to use, modify, and distribute.
