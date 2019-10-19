const API_URL = "https://wt-13aebf4eeaa9913542725d4a90e4d49e-0.sandbox.auth0-extend.com/auth0-ably";
const loginBtn = document.querySelector("#loginBtn");
const logoutBtn = document.querySelector("#logoutBtn");
const sendBtn = document.querySelector("#sendMessage");

logoutBtn.addEventListener("click", (event) => {
  auth.logout();
  UIUpdate.loggedOut();
});

loginBtn.addEventListener("click", (event) => {
  auth.login();
});

sendBtn.addEventListener("click", (event) => {
  ably.publish(document.querySelector("input").value);
});