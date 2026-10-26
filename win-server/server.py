from flask import Flask, request, jsonify
import os
import requests
import threading

app = Flask(__name__)

# 簡易驗證金鑰 (在學校網路建議加上)
API_KEY = "112550191"

# WSL API configuration
WSL_API_URL = "http://127.0.0.1:9000/infer"
WSL_TIMEOUT = 120  # Timeout for WSL API call in seconds

# Global job state
job_lock = threading.Lock()
job_state = {
    "status": "idle",  # "idle", "processing", "success", "error"
    "result": None,
    "error_message": None
}

def normalize_result_schema(result_json):
    images = result_json.get("images") or {}
    return {
        "images": {
            "preview_base64": images.get("preview_base64", ""),
            "foreground_base64": images.get(
                "foreground_base64",
                result_json.get("foreground_base64", "")
            ),
            "gameplay_base64": images.get(
                "gameplay_base64",
                result_json.get("gameplay_base64", "")
            ),
            "background_base64": images.get(
                "background_base64",
                result_json.get("background_base64", "")
            ),
        },
        "metadata": result_json.get("metadata", {})
    }

def check_auth():
    return request.headers.get("X-API-KEY") == API_KEY

def process_image_background(image_bytes, filename, content_type):
    global job_state
    try:
        print(f"Background worker: Sending image to WSL API at {WSL_API_URL}")
        files = {
            'photo': (filename, image_bytes, content_type)
        }
        # Post to the WSL inference API
        response = requests.post(WSL_API_URL, files=files, timeout=WSL_TIMEOUT)
        
        if response.status_code == 200:
            result_json = response.json()
            with job_lock:
                job_state["status"] = "success"
                job_state["result"] = normalize_result_schema(result_json)
                job_state["error_message"] = None
            print("Background worker: Processing completed successfully.")
        else:
            with job_lock:
                job_state["status"] = "error"
                job_state["error_message"] = f"WSL API error: {response.status_code} - {response.text}"
            print(f"Background worker: Processing failed. {job_state['error_message']}")
            
    except requests.exceptions.RequestException as e:
        with job_lock:
            job_state["status"] = "error"
            job_state["error_message"] = f"WSL API request failed: {str(e)}"
        print(f"Background worker: Exception occurred: {str(e)}")

@app.route('/upload', methods=['POST'])
def upload():
    global job_state
    
    if not check_auth():
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    
    print("Received /upload request")
    
    if 'photo' not in request.files:
        return jsonify({"status": "error", "message": "No photo field in request"}), 400
        
    photo = request.files['photo']
    image_bytes = photo.read()
    filename = photo.filename or 'latest.jpg'
    content_type = photo.content_type or 'image/jpeg'
    
    # Save a local copy for debugging purposes (optional, keeps old behavior)
    if not os.path.exists('captures'):
        os.makedirs('captures')
    with open(os.path.join('captures', 'latest.jpg'), 'wb') as f:
        f.write(image_bytes)
        
    # Reset job state and start background processing
    with job_lock:
        job_state["status"] = "processing"
        job_state["result"] = None
        job_state["error_message"] = None
        
    # Start background thread to forward request to WSL API
    worker = threading.Thread(
        target=process_image_background,
        args=(image_bytes, filename, content_type)
    )
    worker.daemon = True
    worker.start()
    
    print("Started background processing thread.")
    
    # Immediately return HTTP 200 acceptance back to Unity
    return jsonify({"status": "accepted"}), 200

@app.route('/download', methods=['GET'])
def download():
    if not check_auth():
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
    with job_lock:
        current_status = job_state["status"]
        result = job_state["result"]
        error_message = job_state["error_message"]
        
    if current_status == "idle":
        # Returning processing when idle keeps Unity polling nicely if it somehow calls early
        return jsonify({"status": "processing"})
        
    elif current_status == "processing":
        return jsonify({"status": "processing"})
        
    elif current_status == "success":
        response_data = {
            "status": "success",
            "images": result.get("images", {}),
            "metadata": result.get("metadata", {})
        }
        print("Returning success payload to Unity.")
        return jsonify(response_data)
        
    elif current_status == "error":
        print(f"Returning error payload to Unity: {error_message}")
        return jsonify({
            "status": "error",
            "message": error_message or "Unknown processing error"
        })

    return jsonify({"status": "error", "message": "Invalid internal state"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
