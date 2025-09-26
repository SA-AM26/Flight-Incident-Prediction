# requirements.txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
scikit-learn>=1.3.0
joblib>=1.3.0

# ================================
# DEPLOYMENT INSTRUCTIONS
# ================================

## Quick Setup (5 minutes)

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **File Structure** 
   Make sure you have these files in your project:
   ```
   your_project/
   ├── aviation_dashboard.py          # The dashboard code
   ├── requirements.txt               # Dependencies  
   ├── artifacts/
   │   ├── binary_classifier_final.pkl
   │   ├── delay_minutes_Poisson_final.pkl
   │   └── aviation_predictions_PRODUCTION_[timestamp].csv
   └── final_unified_dataset.csv
   ```

3. **Launch Dashboard**
   ```bash
   streamlit run aviation_dashboard.py
   ```

4. **Access Dashboard**
   - Local: http://localhost:8501
   - Network: Your IP:8501

## Dashboard Features

### 🛫 Live Operations Tab
- **Real-time simulation**: Time slider to replay historical data
- **Flight status tracking**: Scheduled → In-Flight → Completed  
- **Filter by airline and status**
- **Color-coded risk levels**
- **Delay predictions with minutes**

### ⚠️ Risk Dashboard Tab  
- **Critical safety alerts**: High-risk flights with incident warnings
- **Historical incident context**: "Similar to Mangalore IX-812 - 158 fatalities"
- **Risk distribution charts**
- **Risk breakdown by airline**

### 📊 Analytics Tab
- **Delay pattern analysis**: By time of day, airline, route
- **Primary delay causes**: Weather, Mechanical, Crew, ATC
- **Route performance metrics**
- **Hourly delay trends**

### 🔍 Flight Search Tab
- **Search by flight number, tail number, or route**
- **Detailed flight information**
- **Risk factor analysis** 
- **Incident warnings for high-risk flights**

## Key Capabilities

### Real-Time Simulation
- Replays your historical data as if happening live
- Shows flight progression: Scheduled → In-Flight → Completed
- Time window controls (1-24 hours around current time)
- Auto-advance option with speed control

### Incident-Aware Predictions
- Links risk factors to real aviation incidents
- Shows potential consequences: "158 fatalities", "21 fatalities"
- Provides prevention measures
- Historical context for operational decisions

### Operational Intelligence
- Uses your production pipeline results automatically
- Falls back to real-time prediction if needed
- Comprehensive risk scoring (weather, mechanical, crew, ATC)
- Realistic delay minute calculations (15-120 minutes)

### Production Integration
- Auto-detects your production results files
- Uses pre-computed predictions for speed
- Handles missing data gracefully
- Caches results for performance

## Deployment Options

### Option 1: Local Development
```bash
streamlit run aviation_dashboard.py
```

### Option 2: Network Deployment
```bash
streamlit run aviation_dashboard.py --server.address 0.0.0.0
```

### Option 3: Streamlit Cloud
1. Push code to GitHub
2. Connect to streamlit.io
3. Deploy from repository
4. Add requirements.txt

### Option 4: Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "aviation_dashboard.py", "--server.address", "0.0.0.0"]
```

## Performance Notes

- **Sample Limit**: Adjustable 1K-50K flights for performance
- **Caching**: 5-minute cache for data loading
- **Production Results**: Uses pre-computed predictions when available
- **Memory**: ~2GB RAM recommended for full dataset

## Troubleshooting

### Common Issues:

1. **ModuleNotFoundError: joblib**
   ```bash
   pip install joblib
   ```

2. **Files not found**
   - Check file paths in sidebar
   - Ensure artifacts/ folder exists
   - Verify pickle files are present

3. **Slow performance**
   - Reduce sample limit in sidebar
   - Use production results instead of real-time prediction
   - Clear cache if needed

4. **Time simulation not working**
   - Check if scheduled_dep_utc column exists
   - Dashboard will generate timestamps if missing

### Support Files:
- Check sidebar for file detection status
- Green = Found, Red = Missing
- Auto-refresh available in sidebar

## Security Notes

- Dashboard runs locally by default
- No external API calls
- Uses only your local data files
- Safe for internal operations use

## Next Steps

1. **Test with sample data**: Start with small sample_limit
2. **Verify production results**: Check if pre-computed results load
3. **Customize time window**: Adjust for your operational needs  
4. **Train operations team**: Show key features and alerts
5. **Deploy to operations center**: Use network deployment option

The dashboard transforms your 1.8M flight predictions into actionable operational intelligence with real incident context.
