// Get the current date and time to display on the pages

const currentDate = new Date();
const currentYear = currentDate.getFullYear();
const currentMonth = String(currentDate.getMonth() + 1).padStart(2, "0");

const currentDay = String(currentDate.getDate()).padStart(2, "0");
const currentHours = String(currentDate.getHours()).padStart(2, "0");
const currentMinutes = String(currentDate.getMinutes()).padStart(2, "0");

// From
document.getElementById(
  "dateInput1"
).value = `${currentYear}-${currentMonth}-${currentDay}`;

document.getElementById(
  "timeInput1"
).value = `${currentHours}:${currentMinutes}`;

// To
document.getElementById(
  "dateInput2"
).value = `${currentYear}-${currentMonth}-${currentDay}`;

document.getElementById(
  "timeInput2"
).value = `${currentHours}:${currentMinutes}`;

