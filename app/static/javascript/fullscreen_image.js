// Function to open the explanation plots in full screen for different browsers
function openFullscreen(imageId) {

      var elem = document.getElementById(imageId);

      if (elem.requestFullscreen) {

        elem.requestFullscreen();

      } else if (elem.mozRequestFullScreen) { // Firefox
        elem.mozRequestFullScreen();

      } else if (elem.webkitRequestFullscreen) { //  Chrome, Safari and Opera
        elem.webkitRequestFullscreen();

      } else if (elem.msRequestFullscreen) { // Edge
        elem.msRequestFullscreen();
      }
    }