let UIUpdate = {};

const chatBox = document.querySelector("#chatBox");
const chatMessages = document.querySelector("#chatMessages");

UIUpdate.loggedIn = function(tokens) {
  loginBtn.classList.add("d-none");
  logoutBtn.classList.remove("d-none");
  chatBox.classList.remove("d-none");
};

UIUpdate.loggedOut = function() {
  localStorage.removeItem("access_token");
  loginBtn.classList.remove("d-none");
  logoutBtn.classList.add("d-none");
  chatBox.classList.add("d-none");
};

UIUpdate.alertBox = function(message) {
  const alertBox = document.querySelector(".alert");
  alertBox.innerHTML = message;
};

UIUpdate.updateChatBox = function(messages) {
  chatMessages.innerHTML = messages.map(m => {
    return `[${(new Date(m.timestamp)).toLocaleTimeString("en-CA")}] <em>${m.clientId} said:</em> ${m.data}`;
  }).join("<br>");
};