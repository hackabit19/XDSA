let auth = {};
auth.tokens = {
  ACCESS_TOKEN: "",
  ID_TOKEN: "",
  idPayload: {}
};

auth.login = () => {
  return webAuth.authorize();
};

auth.logout = () => {
  auth.tokens.ACCESS_TOKEN = "";
  auth.tokens.ID_TOKEN = "";
};

auth.isLoggedIn = () => {
  return !!auth.tokens.ACCESS_TOKEN;
};

let webAuth = new auth0.WebAuth(AUTH0_CONFIG);

// Parses the hash info on redirect and extracts the
auth.parseHash = () => {
  webAuth.parseHash((err, authResult) => {
    if (err) {
      console.error(err);
    } else if (authResult && authResult.accessToken && authResult.idToken) {
      window.location.hash = '';
      auth.tokens.ACCESS_TOKEN = authResult.accessToken;
      auth.tokens.ID_TOKEN = authResult.idToken;
      auth.tokens.idPayload = authResult.idTokenPayload;
      UIUpdate.loggedIn();
      UIUpdate.alertBox(`Welcome ${auth.tokens.idPayload.name}`);
      ably.initialize({authUrl: `${API_URL}/auth`, accessToken: auth.tokens.ACCESS_TOKEN});
    }
  });
};

window.addEventListener("DOMContentLoaded", auth.parseHash);