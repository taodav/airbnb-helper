

const AIRBNB_API_URL = "https://api.airbnb.com/v2/listings/";
const CLIENT_ID = "?client_id=3092nxybyb0otqw18e8nh5nty&locale=en-US&currency=CAD&_format=v1_legacy_for_p3&_source=mobile_p3";

console.log("ejbjebfjebfjk");


function getCurrentTabUrl(callback) {
    // Query filter to be passed to chrome.tabs.query - see
    // https://developer.chrome.com/extensions/tabs#method-query
    var queryInfo = {
      active: true,
      currentWindow: true
    };
  
    chrome.tabs.query(queryInfo, (tabs) => {
      var tab = tabs[0];
  
      var url = tab.url;
      console.assert(typeof url == 'string', 'tab.url should be a string');
      console.log(url);
  
      callback(url);
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    alert("hbfjkebfjbefj");
    getCurrentTabUrl((url) => {
      var dropdown = document.getElementById('dropdown');
  
      url = AIRBNB_API_URL + id + CLIENT_ID;
      $.ajax({
          type: 'GET',
          url: url,
          success: function(data){
              console.log(data)
          }
      })
    })
  });