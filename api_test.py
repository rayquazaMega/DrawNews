import requests
import json

# Define the API endpoint
api_url = "http://localhost:8000//generate"

# Define the request payload
payload = {
    "prompt": "A futuristic cityscape at sunset with flying cars",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "style": {}
}

# Set the headers
headers = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
    "Accept": "application/json",  # 让服务器知道我们期望 JSON 响应
    "Origin": "http://localhost",  # 有时需要设置正确的 Origin
    "Referer": "http://localhost",  # 根据需要设置
}

# Make the POST request
response = requests.post(api_url, headers=headers, json=payload)#data=json.dumps(payload))

# Check the response
if response.status_code == 200:
    data = response.json()
    print("Image generated successfully!")
    # Save the base64 image to a file
else:
    print(f"Failed to generate image. Status code: {response.status_code}")
    print(f"Error details: {response.text}")
