from flask import Flask, request, jsonify
import joblib
import numpy as np
from urllib.parse import urlparse
from tld import get_tld
import numpy as np
import pandas as pd
import re
from urllib.parse import urlparse
import os.path
from flask_cors import CORS  # Import the CORS module

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

file_path = 'malicious_phish.csv'
dataset = pd.read_csv(file_path)




n_samples, n_features = dataset.shape

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
dataset["class_url"] = lb_make.fit_transform(dataset["type"])
dataset["class_url"].value_counts()

def url_length(url):
    return len(str(url))

dataset['url_length'] = dataset['url'].apply(lambda i: url_length(i))

from urllib.parse import urlparse

def hostname_length(url):
    return len(urlparse(url).netloc)

dataset['hostname_length'] = dataset['url'].apply(lambda i: hostname_length(i))

def count_www(url):
    url.count('www')
    return url.count('www')

dataset['count-www'] = dataset['url'].apply(lambda i: count_www(i))

def count_https(url):
    return url.count('https')

dataset['count-https'] = dataset['url'].apply(lambda i : count_https(i))

def count_http(url):
    return url.count('http')

dataset['count-http'] = dataset['url'].apply(lambda i : count_http(i))

def count_dot(url):
    count_dot = url.count('.')
    return count_dot

dataset['count.'] = dataset['url'].apply(lambda i: count_dot(i))

def count_per(url):
    return url.count('%')

dataset['count%'] = dataset['url'].apply(lambda i : count_per(i))

def count_ques(url):
    return url.count('?')

dataset['count?'] = dataset['url'].apply(lambda i: count_ques(i))

def count_hyphen(url):
    return url.count('-')

dataset['count-'] = dataset['url'].apply(lambda i: count_hyphen(i))

def count_equal(url):
    return url.count('=')

dataset['count='] = dataset['url'].apply(lambda i: count_equal(i))

def count_atrate(url):

    return url.count('@')

dataset['count@'] = dataset['url'].apply(lambda i: count_atrate(i))

def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

dataset['count_dir'] = dataset['url'].apply(lambda i: no_of_dir(i))

def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')

dataset['count_embed_domian'] = dataset['url'].apply(lambda i: no_of_embed(i))

def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adataset\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0

dataset['short_url'] = dataset['url'].apply(lambda i: shortening_service(i))

from tld import get_tld

#First Directory Length
def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

dataset['fd_length'] = dataset['url'].apply(lambda i: fd_length(i))

#Length of Top Level Domain
dataset['tld'] = dataset['url'].apply(lambda i: get_tld(i,fail_silently=True))

def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

dataset['tld_length'] = dataset['tld'].apply(lambda i: tld_length(i))

def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    if match:
        return 1
    else:
        return 0

dataset['sus_url'] = dataset['url'].apply(lambda i: suspicious_words(i))


def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits

dataset['count-digits']= dataset['url'].apply(lambda i: digit_count(i))


def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters


dataset['count-letters']= dataset['url'].apply(lambda i: letter_count(i))

def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0
dataset['abnormal_url'] = dataset['url'].apply(lambda i: abnormal_url(i))

#Use of IP or not in domain
def having_ip_address(url: str) -> int:
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0
dataset['use_of_ip_address'] = dataset['url'].apply(lambda i: having_ip_address(i))

from googlesearch import search
def google_index(url):
    site = search(url, 5)
    return 1 if site else 0
dataset['google_index'] = dataset['url'].apply(lambda i: google_index(i))

def get_url(url):

    status = []

    status.append(having_ip_address(url))
    status.append(abnormal_url(url))
    status.append(count_dot(url))
    status.append(count_www(url))
    status.append(count_atrate(url))
    status.append(no_of_dir(url))
    status.append(no_of_embed(url))

    status.append(shortening_service(url))
    status.append(count_https(url))
    status.append(count_http(url))

    status.append(count_per(url))
    status.append(count_ques(url))
    status.append(count_hyphen(url))
    status.append(count_equal(url))

    status.append(url_length(url))
    status.append(hostname_length(url))
    status.append(suspicious_words(url))
    status.append(digit_count(url))
    status.append(letter_count(url))
    status.append(fd_length(url))
    tld = get_tld(url,fail_silently=True)

    status.append(tld_length(tld))

    return status




# Load the pre-trained model
model_path = 'classifier5.joblib'
classifier5 = joblib.load(model_path)

def model_predict5(test_url):
    features_test = get_url(test_url)
    features_test = np.array(features_test).reshape((1, -1))
    pred = classifier5.predict(features_test)

    if int(pred[0]) == 0:
        res = "Benign"
    elif int(pred[0]) == 1:
        res = "Defacement"
    elif int(pred[0]) == 2:
        res = "Phishing"
    elif int(pred[0]) == 3:
        res = "Malware"

    return res

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        url = data['url']

        # Perform prediction
        prediction = model_predict5(url)

        # Return the result
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)