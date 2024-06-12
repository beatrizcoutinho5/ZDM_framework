// Functions to update and format the current date for the Dashboard Pages

function formatDate(date) {

    const day = date.getDate();
    const month = date.toLocaleString('en-GB', { month: 'long' });
    const year = date.getFullYear();
    const suffix = getNumberSuffix(day);

    return `${day}${suffix} ${month} ${year}`;
}

// Function to get the suffix for a number
function getNumberSuffix(number) {

    if (number >= 11 && number <= 13) {
        return 'th';
    }

    const lastDigit = number % 10;

    switch (lastDigit) {

        case 1: return 'st';
        case 2: return 'nd';
        case 3: return 'rd';

        default: return 'th';

    }
}

// Function to format the time
function formatTime(date) {

    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');

    return `${hours}:${minutes}:${seconds}`;
}

// Function to update the display
function updateDateTimeDisplay() {

    const currentDate = new Date();
    const formattedDate = formatDate(currentDate);
    const formattedTime = formatTime(currentDate);
    const dateTimeDisplay = document.getElementById('dateTimeDisplay');

    dateTimeDisplay.innerHTML = `${formattedDate} <br/> ${formattedTime}`;
}

updateDateTimeDisplay();

// Update the  display every second
setInterval(updateDateTimeDisplay, 1000);