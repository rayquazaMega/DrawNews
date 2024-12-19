import http.client
import json
import base64

# Define the API endpoint
host = "127.0.0.1"
port = 8000
endpoint = "/generate"

# Define the request payload
payload = {
    "prompt": "A futuristic cityscape at sunset with flying cars",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "style": {}
}

# Prepare headers
headers = {
    "Content-Type": "application/json"
}

# Initialize connection
conn = http.client.HTTPConnection(host, port)

# Make the POST request
try:
    conn.request("POST", endpoint, body=json.dumps(payload), headers=headers)
    response = conn.getresponse()

    # Read the response
    if response.status == 200:
        data = json.loads(response.read().decode("utf-8"))
        print("Image generated successfully!")
        # Save the base64 image to a file
        with open("generated_image.png", "wb") as f:
            f.write(base64.b64decode(data["image_base64"]))
    else:
        print(f"Failed to generate image. Status code: {response.status}")
        print(f"Error details: {response.read().decode('utf-8')}")
finally:
    conn.close()
