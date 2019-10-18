let realtime;
let channel;
let ably = {};
let messages = [];

ably.initialize = function(options) {
  let Ably = window.Ably;
  realtime = new Ably.Realtime({
    authUrl: options.authUrl,
    authHeaders: {
      "Authorization": `Bearer ${options.accessToken}`
    }
  });
  realtime.connection.once('connected', function () {
    console.log("Client connected to Ably using JWT");
    channel = realtime.channels.get('auth0-ably');

    UIUpdate.updateChatBox([]);

    channel.subscribe("main", (msg) => {
      messages.push(msg);
      UIUpdate.updateChatBox(messages);
    });
  });
};

ably.publish = function(msg) {
  if (!channel) return console.log("Channel is not opened yet");
  channel.publish("main", msg);
};

