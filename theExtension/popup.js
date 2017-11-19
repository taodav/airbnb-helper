// Search the bookmarks when entering the search keyword.

const API_URL = "https://8c3a54cd.ngrok.io/pred?id=";

$(function() {
  $('#search').change(function() {
     $('#bookmarks').empty();
     dumpBookmarks($('#search').val());
  });
});
// Traverse the bookmark tree, and print the folder and nodes.
function dumpBookmarks(query) {
  var bookmarkTreeNodes = chrome.bookmarks.getTree(
    function(bookmarkTreeNodes) {
      priceStr = (typeof query.price === "number")? "$" + Math.ceil(query.price) + " USD" : query.price;
      $('#price-estimate').html(priceStr);
      $('#type-estimate').html(query.type);
    });
}

function getUrlAndMakeRequest(){
  chrome.tabs.query({'active': true, 'lastFocusedWindow': true}, function (tabs) {
      var url = tabs[0].url;

      var parts = url.split('/');
      var keys = parts[parts.length - 1];
      var id = keys.split('?')[0];

      url = API_URL + id;

      $.ajax({
        type: "GET",
        url: url,
        success: function(data){
            dumpBookmarks(data);
        },
        error: function(err){
            console.log("RIP");
        }
    })
  });
}

document.addEventListener('DOMContentLoaded', function () {
  dumpBookmarks({price: "Fetching price...", type: "Fetching type....."})
  getUrlAndMakeRequest();
});