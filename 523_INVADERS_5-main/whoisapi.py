import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/whois', methods=['GET'])
def whois():
    url = "https://whois-by-api-ninjas.p.rapidapi.com/v1/whois"

    # Get the domain parameter from the request
    domain_to_query = request.args.get('domain', '')

    querystring = {"domain": domain_to_query}

    headers = {
        "X-RapidAPI-Key": "private_key", //private keys can't be shared
        "X-RapidAPI-Host": "private_key" //private keys can't be shared
    }

    response = requests.get(url, headers=headers, params=querystring)

    return jsonify(response.json())

if __name__ == '__main__':
    app.run(debug=True, port=5001)
