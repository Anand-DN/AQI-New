import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys from environment
WAQI_TOKEN = os.getenv('WAQI_API_TOKEN')
DATA_GOV_KEY = os.getenv('DATA_GOV_API_KEY')

def collect_openaq_data():
    """
    Collect AQI data from OpenAQ API (No API key needed)
    """
    print("🌍 Starting OpenAQ data collection...\n")
    
    cities = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai", 
              "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]
    
    all_data = []
    date_to = datetime.now().strftime('%Y-%m-%d')
    date_from = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    for city in cities:
        url = "https://api.openaq.org/v2/measurements"
        params = {
            'city': city,
            'country': 'IN',
            'date_from': date_from,
            'date_to': date_to,
            'limit': 1000,
            'parameter': 'pm25'
        }
        
        try:
            print(f"📍 Fetching {city}...")
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and len(data['results']) > 0:
                    for record in data['results']:
                        all_data.append({
                            'city': city,
                            'location': record.get('location', ''),
                            'parameter': record.get('parameter', ''),
                            'value': record.get('value', 0),
                            'unit': record.get('unit', ''),
                            'date': record.get('date', {}).get('utc', ''),
                            'lat': record.get('coordinates', {}).get('latitude', ''),
                            'lon': record.get('coordinates', {}).get('longitude', '')
                        })
                    print(f"   ✓ Got {len(data['results'])} records")
                else:
                    print(f"   ⚠ No data available")
            else:
                print(f"   ✗ HTTP {response.status_code}")
                
            time.sleep(1)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    if all_data:
        df = pd.DataFrame(all_data)
        os.makedirs('data', exist_ok=True)
        csv_path = 'data/aqi_data_openaq.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Success! Collected {len(df)} total records")
        print(f"💾 Saved to: {csv_path}\n")
        return df
    else:
        print("\n❌ No data collected")
        return None


def collect_waqi_data():
    """
    Collect AQI data from WAQI API (Needs API token)
    """
    if not WAQI_TOKEN:
        print("❌ WAQI_API_TOKEN not found in .env file!")
        print("Get your token from: https://aqicn.org/data-platform/token/")
        return None
    
    print("🌍 Starting WAQI data collection...\n")
    
    cities = ["delhi", "mumbai", "bangalore", "kolkata", "chennai", 
              "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow"]
    
    all_data = []
    
    for city in cities:
        url = f"https://api.waqi.info/feed/{city}/?token={WAQI_TOKEN}"
        
        try:
            print(f"📍 Fetching {city.title()}...")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'ok':
                    aqi_data = data['data']
                    
                    record = {
                        'city': city.title(),
                        'aqi': aqi_data.get('aqi', 0),
                        'dominant_pollutant': aqi_data.get('dominentpol', ''),
                        'timestamp': aqi_data.get('time', {}).get('s', ''),
                        'station': aqi_data.get('city', {}).get('name', ''),
                        'lat': aqi_data.get('city', {}).get('geo', [0])[0],
                        'lon': aqi_data.get('city', {}).get('geo', [0])[1] if len(aqi_data.get('city', {}).get('geo', [])) > 1 else 0,
                    }
                    
                    # Individual pollutants
                    iaqi = aqi_data.get('iaqi', {})
                    record['pm25'] = iaqi.get('pm25', {}).get('v', None)
                    record['pm10'] = iaqi.get('pm10', {}).get('v', None)
                    record['o3'] = iaqi.get('o3', {}).get('v', None)
                    record['no2'] = iaqi.get('no2', {}).get('v', None)
                    record['so2'] = iaqi.get('so2', {}).get('v', None)
                    record['co'] = iaqi.get('co', {}).get('v', None)
                    
                    all_data.append(record)
                    print(f"   ✓ AQI: {record['aqi']}, Dominant: {record['dominant_pollutant']}")
                else:
                    print(f"   ⚠ API Error: {data.get('data', 'Unknown')}")
            else:
                print(f"   ✗ HTTP {response.status_code}")
                
            time.sleep(1)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    if all_data:
        df = pd.DataFrame(all_data)
        os.makedirs('data', exist_ok=True)
        csv_path = 'data/aqi_data_waqi.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Success! Collected {len(df)} records")
        print(f"💾 Saved to: {csv_path}\n")
        return df
    else:
        print("\n❌ No data collected")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("        AQI Data Collection Script")
    print("=" * 60)
    print("\nChoose data source:")
    print("1. OpenAQ (No API key needed)")
    print("2. WAQI (Needs API token)")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        collect_openaq_data()
    elif choice == '2':
        collect_waqi_data()
    elif choice == '3':
        collect_openaq_data()
        print("\n" + "-" * 60 + "\n")
        collect_waqi_data()
    else:
        print("Invalid choice!")
