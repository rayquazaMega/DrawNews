import requests
client = requests.Session()
client.headers["User-Agent"] = "Mozilla/5.0"
resp = client.get("http://localhost:8000/")
print(resp.text)
