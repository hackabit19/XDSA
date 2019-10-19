const express = require("express");
const expressjwt = require("express-jwt");
const jwt = require("jsonwebtoken");
const jwks = require("jwks-rsa");
const https = require("https");
const cors = require("cors");
const config = require("./config");
const auth0Config = config.Auth0;
const ABLY_KEY = config.Ably.key;

const appId = ABLY_KEY.split(".")[0];
const keyId = ABLY_KEY.split(".")[1].split(":")[0];
const keySecret = ABLY_KEY.split(".")[1].split(":")[1];

const app = express();

const jwtCheck = expressjwt({
  secret: jwks.expressJwtSecret({
    cache: true,
    rateLimit: true,
    jwksRequestsPerMinute: 5,
    jwksUri: `https://${auth0Config.domain}/.well-known/jwks.json`
  }),
  audience: auth0Config.audience,
  issuer: `https://${auth0Config.domain}/`,
  algorithms: ['RS256']
});

app.use(cors());

app.get("/auth", jwtCheck, (req, res) => {
  // Get the ID Token from Auth0
  https.get(`https://${auth0Config.domain}/userinfo`, {headers: {Authorization: req.headers.authorization}}, (resp) => {
    let data = '';

    // A chunk of data has been recieved.
    resp.on('data', (chunk) => {
      data += chunk;
    });

    // The whole response has been received. Create the Ably JWT
    resp.on('end', () => {
      let userInfo = JSON.parse(data);

      const payload = Object.assign({}, {
        "x-ably-capability": JSON.stringify({
          "*": ["publish", "subscribe"]
        }),
        "x-ably-clientId": userInfo.name
      });
      const options = {
        expiresIn: "120 minutes",
        keyid: `${appId}.${keyId}`
      };

      jwt.sign(payload, keySecret, options, (err, tokenId) => {
        if (err) console.log(err);

        res.header("Cache-Control", "private, no-cache, no-store, must-revalidate");
        res.setHeader("Content-Type", "application/json");
        res.status(200).send(JSON.stringify(tokenId));
      });
    });

  }).on("error", (err) => {
    console.log("Error fetching ID Token: " + err.message);
  });
});

app.get('*', function (req, res) {
  res.sendStatus(404);
});

app.listen(8888, () => console.log("API started on port 8888"));