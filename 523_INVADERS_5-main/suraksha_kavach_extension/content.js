// This script will be injected into every webpage

// You can manipulate the webpage here if needed
console.log('Content  loaded on', window.location.href);

  // content.js or popup.js

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.message === 'update_icon') {
    chrome.action.setIcon({ path: { "48": message.iconPath } });
  }
});

