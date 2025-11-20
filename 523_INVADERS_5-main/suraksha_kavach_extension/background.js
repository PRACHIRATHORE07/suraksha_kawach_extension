// background.js

console.log("Background script is running.");

let currentIconPath = "lock.png";

// Listen for messages from contentScript.js
chrome.runtime.onInstalled.addListener(() => {
  console.log('Extension installed!');
});
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.message === 'page_load') {
    console.log('Page loaded message received');
    // Handle the message as needed
  }

  // Update the icon when a tab is activated
  if (message.message === 'update_icon') {
    chrome.action.setIcon({ path: { "48": message.iconPath } });
  }
});
// Listen for tab change
chrome.tabs.onActivated.addListener((activeInfo) => {
  // Send a message to the content script or popup script to update the icon
  chrome.tabs.sendMessage(activeInfo.tabId, { message: "update_icon", iconPath: currentIconPath });
});




