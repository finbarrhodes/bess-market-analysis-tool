# UK BESS Market Analysis & Forecasting Tool

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://uk-bess-exploration.streamlit.app)

A Python toolkit for analyzing the UK's Battery Energy Storage System (BESS) markets, including historical trend analysis, price forecasting, and market insights. This repo contains data acquisition tools for NESO and Elexon data as well as dashboard visualization using Streamlit. This project serves as a personal endeavor to learn more about and uncover insights into the UK BESS landscape; some of what is outlined below has yet to be implemented/shipped; more to come!

## Project Overview

This project provides:
- **Data Collection**: Automated scraping from National Grid ESO and Elexon BMRS APIs
- **Market Analysis**: Trend analysis across DC, DR, DM, BM, and FR markets
- **Price Forecasting**: Multiple ML and statistical models for price prediction
- **Visualization**: Interactive Plotly dashboards and Streamlit web app
- **Reporting**: Automated report generation with key insights

## Key Features

- Historical data collection from public APIs (2020-present)
- Supply-demand driver analysis
- Regulatory change impact assessment
- Multiple forecasting models (ARIMA, Prophet, scikit-learn)
- Model comparison and evaluation
- Interactive web dashboard
- Automated report generation

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git

### Installation

1. **Navigate to the project directory**:
```bash
cd gb-bess-market-analysis
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure the project**:
   - Edit `config.yaml` to adjust date ranges and analysis parameters

Both the NESO Data Portal and Elexon Insights Solution APIs are fully public — no registration or API key is required.

## Project Structure

```
gb-bess-market-analysis/
├── data/
│   ├── raw/              # Raw data from APIs
│   └── processed/        # Cleaned and processed data
├── src/
│   ├── data_collection/  # API scrapers and data ingestion
│   │   ├── neso_collector.py      # National Grid ESO collector
│   │   ├── elexon_collector.py    # Elexon BMRS collector
│   │   └── collect_data.py        # Main collection script
│   ├── analysis/         # Market trend analysis
│   ├── forecasting/      # Prediction models
│   ├── visualization/    # Plotting and dashboard code
│   └── utils.py          # Utility functions
├── notebooks/            # Jupyter notebooks for exploration
├── reports/              # Generated reports and figures
├── tests/                # Unit tests
├── logs/                 # Application logs
├── config.yaml           # Project configuration
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Usage

### 1. Data Collection

Collect historical market data:

```bash
# Collect data from both sources for a date range
python src/data_collection/collect_data.py --start 2024-01-01 --end 2024-01-31

# Collect only from National Grid ESO
python src/data_collection/collect_data.py --start 2024-01-01 --end 2024-01-31 --sources neso

# Collect only from Elexon BMRS
python src/data_collection/collect_data.py --start 2024-01-01 --end 2024-01-31 --sources elexon

# Use date range from config.yaml
python src/data_collection/collect_data.py
```

**NESO** collects DC/DR/DM auction clearing prices and EAC results (Sep 2021 – present) via the NESO Data Portal CKAN API. **Elexon** collects System Sell/Buy Prices, Market Index Prices, and half-hourly generation by fuel type via the Elexon Insights Solution API.

### 2. Run Analysis

Analyze market trends (coming soon):
```bash
python src/analysis/market_analysis.py
```

### 3. Generate Forecasts

Train and evaluate forecasting models (coming soon):
```bash
python src/forecasting/train_models.py
```

### 4. Launch Dashboard

Start the interactive Streamlit app (market dashboard + revenue backtester):
```bash
streamlit run app.py
```

Or visit the live deployment: [uk-bess-exploration.streamlit.app](https://uk-bess-exploration.streamlit.app)

## Key Markets Analyzed

- **Dynamic Containment (DC)**: Fast-acting frequency response (High and Low)
- **Dynamic Regulation (DR)**: Continuous frequency regulation (High and Low)
- **Dynamic Moderation (DM)**: Slower frequency response (High and Low)

Additional markets — including the Balancing Mechanism, legacy Frequency Response services, and the Day-Ahead energy market — are planned for future data collection.

## Forecasting Models

The project will implement and compare multiple forecasting approaches:

1. **Statistical Models**:
   - ARIMA (AutoRegressive Integrated Moving Average)
   - SARIMA (Seasonal ARIMA)
   - Prophet (Facebook's time series forecasting)

2. **Machine Learning Models**:
   - Random Forest
   - Gradient Boosting
   - Feature engineering with lag and rolling statistics

3. **Model Evaluation**:
   - RMSE, MAE, MAPE metrics
   - Cross-validation
   - Residual analysis

## Data Sources

- **NESO Data Portal**: DC/DR/DM auction clearing prices and volumes, EAC results (Nov 2023 – present)
  - API: CKAN datastore (`https://api.neso.energy/api/3/action`) — no key required
  - Website: https://www.neso.energy/

- **Elexon Insights Solution**: System Sell/Buy Prices, Market Index Prices, half-hourly generation by fuel type
  - API: `https://data.elexon.co.uk/bmrs/api/v1` — no key required
  - Website: https://www.elexon.co.uk/

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## Important Notes

### Data Availability
- Dynamic Containment data is typically available from 2020 onwards
- Some historical data may have different formats
- Check data availability windows for each market

### Rate Limits
- NESO Data Portal (CKAN datastore): 2 requests/minute
- Elexon Insights Solution: 60 requests/minute
- Adjust in `config.yaml` if needed




## 📧 Contact

Finbar Rhodes - [LinkedIn](https://www.linkedin.com/in/finbar-rhodes-637650210/)

---

**Note**: This project is for analysis and educational purposes only. Not financial or trading advice.
