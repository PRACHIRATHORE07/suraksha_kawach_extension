import re
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

def extract_mobile_numbers(html_content):
    # Define a regular expression to match potential mobile numbers
    mobile_number_pattern = re.compile(r'(?:\+91\s?)?(\d{10})|(?:\b\d{4}‐\d{3}‐\d{4}\b)')

    # Find all matches in the HTML content
    mobile_numbers = re.findall(mobile_number_pattern, html_content)

    # Return the extracted numbers without leading zeros
    return [number.lstrip('0') for number in mobile_numbers]

@app.route('/extract_mobile_numbers', methods=['POST'])
def api_extract_mobile_numbers():
    try:
        data = request.json
        if 'url' not in data:
            return jsonify({'error': 'URL is missing in the request'}), 400

        # Fetch HTML content from the provided URL
        response = requests.get(data['url'])
        html_content = response.text

        # Extract mobile numbers from the HTML content that start with +91
        mobile_numbers = extract_mobile_numbers(html_content)

        return jsonify({'status': 'success', 'mobile_numbers': mobile_numbers})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5002, debug=True)