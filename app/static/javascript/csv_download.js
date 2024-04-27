// Function to download the CSV file with the historic data
document.addEventListener('DOMContentLoaded', function() {

    document.getElementById('download-button').addEventListener('click', function(event) {

        event.preventDefault();
        window.location.href = '../static/data.csv';

    });
});
