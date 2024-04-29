// Function to allow the users to select a time period for the Analytics and Historic Data pages
// and for the interface values to update

const dateInput1 = document.getElementById('dateInput1');
const timeInput1 = document.getElementById('timeInput1');

const dateInput2 = document.getElementById('dateInput2');
const timeInput2 = document.getElementById('timeInput2');

const delay = 1000; // 1 second delay

// Function to handle date and time change event

let timeoutId;
function handleDateTimeChangeWithDelay() {

    clearTimeout(timeoutId);

    timeoutId = setTimeout(async () => {

        // Gets the values entered by the user
        const fromDate = dateInput1.value + ' ' + timeInput1.value;
        const toDate = dateInput2.value + ' ' + timeInput2.value;

        // Sends to the function that will return the updated values, regarding the dates that were selected
        const response = await fetch(`/update-analytics?fromDate=${fromDate}&toDate=${toDate}`);

        // Store the values of date and time inputs in local storage so it doesnt go back to the current
        // date and time when the page is reloaded to show the new interface values
        localStorage.setItem('fromDate', fromDate);
        localStorage.setItem('toDate', toDate);

        location.reload();

    }, delay);
}

dateInput1.addEventListener('change', handleDateTimeChangeWithDelay);
timeInput1.addEventListener('change', handleDateTimeChangeWithDelay);

dateInput2.addEventListener('change', handleDateTimeChangeWithDelay);
timeInput2.addEventListener('change', handleDateTimeChangeWithDelay);

// Set the values of date and time inputs from the local storage after the page reloads
document.addEventListener('DOMContentLoaded', function() {

    const fromDate = localStorage.getItem('fromDate');
    const toDate = localStorage.getItem('toDate');

    if (fromDate && toDate) {

        dateInput1.value = fromDate.split(' ')[0];
        timeInput1.value = fromDate.split(' ')[1];

        dateInput2.value = toDate.split(' ')[0];
        timeInput2.value = toDate.split(' ')[1];

    }
});

