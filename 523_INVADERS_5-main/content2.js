async function whoisapi(currentURL) {
    try {
        const response = await fetch(`http://localhost:5000/whois?domain=${currentURL}`);
        const data = await response.json();
        console.log(data);

        const creationDateEpoch = data.creation_date[0];
        const creationDate = new Date(creationDateEpoch * 1000);

        const expirationDateEpoch = data.expiration_date[0];
        const expirationDate = new Date(expirationDateEpoch * 1000);

        const differenceInMilliseconds = Date.now() - creationDate.getTime();
        const differenceInDays = differenceInMilliseconds / (1000 * 3600 * 24);

        if (differenceInDays <= 90) {
            alert("Domain is registered within 3 months!");
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

function extractLogos() {
    var logoImages = document.querySelectorAll('img[class*="logo"], img[src*="logo"]');
    var logos = Array.from(logoImages).map(img => img.src);
    chrome.runtime.sendMessage({ action: 'logosDetected', logos: logos });
}

async function sendAPIRequest(processedURL) {
    try {
        const response = await fetch('http://127.0.0.1:5001/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: processedURL }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        console.log('API Response:', data);
        // Handle the API response as needed
    } catch (error) {
        console.error('Error:', error);
    }
}

async function scanphonenumber(currentURL) {
    try {
        const apiUrl = 'http://127.0.0.1:5002/extract_mobile_numbers';
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: currentURL }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        console.log(data);
        // Process the data returned from the Flask API
    } catch (error) {
        console.error('Error fetching data:', error);
    }
}

function removeProtocolAndWWW(url) {
    // Remove "http://" or "https://"
    let processedURL = url.replace(/^(https?:)?\/\//, '');

    // Remove "www."
    processedURL = processedURL.replace(/^www\./, '');

    return processedURL;
  }

chrome.storage.sync.get(['extensionEnabled'], function (result) {
    if (result.extensionEnabled) {
      // Your content script logic when the extension is enabled
      console.log('Extension is enabled.');
    } else {
      // Your content script logic when the extension is disabled
      console.log('Extension is disabled.');
    }
  });

  // Add listener for messages from popup.js
  chrome.runtime.onMessage.addListener(
    function (request, sender, sendResponse) {
      if (request.extensionEnabled !== undefined) {
        // Handle the state change sent from popup.js
        if (request.extensionEnabled) {
          console.log('Extension is enabled.');
        } else {
          console.log('Extension is disabled.');
        }
      }
    }
  );

async function main() {
    const currentURL = window.location.href;
    const processedURL = removeProtocolAndWWW(currentURL);

    // Uncomment and call the functions as needed
    await whoisapi(currentURL);
    // extractLogos();
    await sendAPIRequest(processedURL);
    scanphonenumber(currentURL);
}

if (!window.location.href.endsWith('.pdf')) {
    // Ensure the entire block is asynchronous
    (async () => {
        try {
            await main();
        } catch (error) {
            console.error('Error:', error);
        }
    })();
}
