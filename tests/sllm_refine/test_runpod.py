import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()

ip = os.getenv("RUNPOD_IP")
port = os.getenv("RUNPOD_PORT")
key = os.getenv("RUNPOD_API_KEY")

print(f"IP: {ip}")
print(f"PORT: {port}")

url = f"http://{ip}:{port}/v1/models"
headers = {"Authorization": f"Bearer {key}"}

try:
    resp = requests.get(url, headers=headers, timeout=10)
    print(f"\nStatus: {resp.status_code}")
    data = resp.json()
    for model in data.get("data", []):
        print(f"Model: {model.get('id')}")
except Exception as e:
    print(f"Error: {e}")
