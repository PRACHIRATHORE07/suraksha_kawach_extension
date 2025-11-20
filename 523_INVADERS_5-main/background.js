// background.js
chrome.runtime.onMessage.addListener(
  function(request, sender, sendResponse) {
      console.log('Background Script - Received Message:', request);
      if (request && request.apiResponse === "Safe") {
          console.log('Background Script - Opening Popup');
          window.open("popup.html", "extension_popup", "width=300,height=400,status=no,scrollbars=yes,resizable=no");
      }
  }
);


chrome.runtime.onInstalled.addListener(function () {
  // Set the initial state when the extension is installed
  chrome.storage.sync.set({ 'extensionEnabled': true });
});

chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === 'logosDetected') {
    chrome.browserAction.setBadgeText({ text: request.logos.length.toString() });
    chrome.browserAction.setBadgeBackgroundColor({ color: '#FF0000' });
    chrome.storage.local.set({ logos: request.logos });
  }
});
