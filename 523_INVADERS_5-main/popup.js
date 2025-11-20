console.log('Popup Script - Loaded');
document.addEventListener('DOMContentLoaded', function () {
  var toggleButton = document.getElementById('toggleButton');

  // Set the initial state based on the stored value
  chrome.storage.sync.get(['extensionEnabled'], function (result) {
    if (result.extensionEnabled) {
      toggleButton.textContent = 'Turn Off';
    } else {
      toggleButton.textContent = 'Turn On';
    }
  });

  // Toggle the extension state when the button is clicked
  toggleButton.addEventListener('click', function () {
    chrome.storage.sync.get(['extensionEnabled'], function (result) {
      var newEnabledState = !result.extensionEnabled;

      // Save the new state
      chrome.storage.sync.set({ 'extensionEnabled': newEnabledState });

      // Update the button text
      toggleButton.textContent = newEnabledState ? 'Turn Off' : 'Turn On';

      // Send a message to content.js to handle the state change
      chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        var activeTab = tabs[0];
        chrome.tabs.sendMessage(activeTab.id, { extensionEnabled: newEnabledState });
      });
    });
  });
});

document.addEventListener('DOMContentLoaded', function() {
  chrome.storage.local.get(['logos'], function(result) {
    displayLogos(result.logos);
  });

  function displayLogos(logos) {
    var logoContainer = document.getElementById('logoContainer');
    logos.forEach(function(logo, index) {
      var imgElement = document.createElement('img');
      imgElement.src = logo;
      imgElement.alt = 'Logo ' + index;
      logoContainer.appendChild(imgElement);
    });
  }
});
