import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np

print("🔧 Creating sample AQI data...\n")

# Cities
cities = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", 
          "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]

# Generate data for years 2015–2025
all_data = []

for year in [2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025]:
    # 100 records per city per year
    dates = pd.date_range(
        start=f'{year}-01-01', 
        end=f'{year}-12-31' if year < 2025 else '2025-11-26',
        periods=100
    )
    
    for city in cities:
        # Base AQI varies by city
        base_aqi = {
            "Delhi": 250, "Mumbai": 180, "Bangalore": 120,
            "Chennai": 100, "Kolkata": 200, "Hyderabad": 140,
            "Pune": 110, "Ahmedabad": 160, "Jaipur": 190, "Lucknow": 220
        }
        
        for date in dates:
            aqi_base = base_aqi.get(city, 150)
            aqi = max(50, min(500, int(np.random.normal(aqi_base, 50))))
            
            all_data.append({
                'city': city,
                'aqi': aqi,
                'dominant_pollutant': np.random.choice(['pm25', 'pm10', 'o3', 'no2'], 
                                                       p=[0.5, 0.3, 0.15, 0.05]),
                'timestamp': date.strftime('%Y-%m-%d %H:%M:%S'),
                'station': f'{city} Central',
                'lat': np.random.uniform(10, 30),
                'lon': np.random.uniform(70, 90),
                'pm25': max(0, int(np.random.normal(aqi * 0.6, 20))),
                'pm10': max(0, int(np.random.normal(aqi * 0.8, 30))),
                'o3': max(0, int(np.random.normal(50, 20))),
                'no2': max(0, int(np.random.normal(40, 15))),
                'so2': max(0, int(np.random.normal(20, 10))),
                'co': max(0, int(np.random.normal(10, 5)))
            })

df = pd.DataFrame(all_data)

# Create data folder
os.makedirs('data', exist_ok=True)

# Save to CSV
csv_path = 'data/aqi_data_waqi.csv'
df.to_csv(csv_path, index=False)

print(f"✅ Created {len(df)} records")
print(f"📊 Cities: {', '.join(cities)}")
print(f"📅 Years: 2015–2025")
print(f"💾 Saved to: {csv_path}\n")

# Show sample
print("Sample data:")
print(df.head(10))
print(f"\nRecords per city per year: ~100")
print(f"Total cities: {len(cities)}")
