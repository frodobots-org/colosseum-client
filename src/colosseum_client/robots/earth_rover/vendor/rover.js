// The Earth Rovers SDK's browser-side rover connection, trimmed for the Colosseum link:
// Agora RTC for the cameras (uid 1000 front, 1001 rear) and Agora RTM 1.x for commands
// and telemetry. The Python side injects window.ROVER_CONFIG and drives the window.* API.
(function () {
  const cfg = window.ROVER_CONFIG || {};
  window.rtmReady = false;
  window.rtmAborted = false;
  window.rtcState = "idle";
  window.rtcError = null;
  window.remoteUsers = {};
  window.rtmStats = { dispatched: 0, delivered: 0, failed: 0, lastError: null, lastDeliveredAt: null };

  // --- RTC: subscribe to every publisher, one at a time (WebRTC negotiation), and play video tracks.
  const client = AgoraRTC.createClient({ mode: "rtc", codec: cfg.codec || "vp8" });
  const queue = [];
  let processing = false;
  async function processQueue() {
    if (processing) return;
    processing = true;
    while (queue.length) {
      const { user, mediaType } = queue.shift();
      try {
        await client.subscribe(user, mediaType);
        if (mediaType === "video") {
          let holder = document.getElementById("player-" + user.uid);
          if (!holder) {
            holder = document.createElement("div");
            holder.id = "player-" + user.uid;
            holder.className = "player";
            document.getElementById("players").appendChild(holder);
          }
          user.videoTrack.play("player-" + user.uid);
          user.videoTrack.captureEnabled = true;
        }
        // Audio is subscribed but not played: the link has no speaker.
      } catch (err) {
        window.rtcError = String((err && err.message) || err);
      }
      await new Promise((r) => setTimeout(r, 100));
    }
    processing = false;
  }
  client.on("user-published", (user, mediaType) => {
    window.remoteUsers[user.uid] = user;
    queue.push({ user, mediaType });
    processQueue();
  });
  client.on("user-unpublished", (user, mediaType) => {
    if (mediaType === "video") delete window.remoteUsers[user.uid];
  });
  client.on("connection-state-change", (current) => {
    window.rtcState = current;
  });
  window.joinRtc = async function () {
    await client.join(cfg.appid, cfg.channel, cfg.rtcToken || null, cfg.uid ? Number(cfg.uid) : null);
    window.rtcState = "joined";
  };

  const surfaces = new WeakMap();
  async function captureFrame(track, format, quality) {
    const frame = await track.getCurrentFrameData();
    if (!frame) return null;
    let surface = surfaces.get(track);
    if (!surface) {
      const canvas = document.createElement("canvas");
      surface = { canvas, context: canvas.getContext("2d") };
      surfaces.set(track, surface);
    }
    if (surface.canvas.width !== frame.width) surface.canvas.width = frame.width;
    if (surface.canvas.height !== frame.height) surface.canvas.height = frame.height;
    surface.context.putImageData(frame, 0, 0);
    return surface.canvas.toDataURL("image/" + (format || "jpeg"), quality === undefined ? 0.8 : quality);
  }
  // Same contract as the SDK's getFramePacket: {data_url, timestamp} | {error} | null.
  window.getFramePacket = async function (uid, format, quality) {
    const user = window.remoteUsers[uid];
    if (!user || !user.videoTrack || !user.videoTrack.captureEnabled) return null;
    const capturedAt = Date.now() / 1000;
    try {
      const dataUrl = await captureFrame(user.videoTrack, format, quality);
      return dataUrl ? { data_url: dataUrl, timestamp: capturedAt } : null;
    } catch (err) {
      const text = String((err && err.message) || err);
      if (/IndexSizeError|source (width|height) is 0/i.test(text)) {
        return { error: "video track is publishing but no frames decode (0x0): the browser lacks the H.264 codec; install Google Chrome" };
      }
      return { error: text };
    }
  };

  // --- RTM 1.x: login as the ride user; the rover's reports arrive as peer messages from BOT_UID.
  const rtm = AgoraRTM.createInstance(cfg.appid);
  const rtmChannel = rtm.createChannel(cfg.channel);
  rtm.on("ConnectionStateChanged", (state) => {
    window.rtmReady = state === "CONNECTED";
    window.rtmAborted = state === "ABORTED";
  });
  rtm.on("MessageFromPeer", (message, peerId) => {
    if (String(peerId) !== String(cfg.botUid)) return;
    let text;
    try {
      text = message.rawMessage ? new TextDecoder().decode(message.rawMessage) : message.text;
    } catch (err) {
      return;
    }
    if (text && window.roverReport) window.roverReport(text);
  });
  window.joinRtm = async function () {
    await rtm.login({ token: cfg.rtmToken, uid: String(cfg.uid) });
    try { await rtmChannel.join(); } catch (err) { /* channel membership is optional */ }
    window.rtmReady = true;
  };

  const SEND_TIMEOUT_MS = 4000;
  function dispatch(json) {
    if (!window.rtmReady) {
      window.rtmStats.failed += 1;
      window.rtmStats.lastError = "RTM client is not connected";
      throw new Error("RTM client is not connected");
    }
    window.rtmStats.dispatched += 1;
    const result = rtm
      .sendMessageToPeer({ text: JSON.stringify(json) }, String(cfg.botUid))
      .then((sendResult) => {
        if (sendResult && sendResult.hasPeerReceived === false) throw new Error("rover did not receive the message (offline?)");
        window.rtmStats.delivered += 1;
        window.rtmStats.lastError = null;
        window.rtmStats.lastDeliveredAt = Date.now() / 1000;
        return true;
      })
      .catch((err) => {
        window.rtmStats.failed += 1;
        window.rtmStats.lastError = String((err && err.message) || err);
        throw err;
      });
    result.catch(() => undefined);
    return result;
  }
  window.sendMessage = function (json) {
    dispatch(json);
    return true;
  };
  window.sendMessageAwait = function (json) {
    const send = dispatch(json);
    let timer;
    const deadline = new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error("rover unreachable: no receipt within " + SEND_TIMEOUT_MS + " ms")), SEND_TIMEOUT_MS);
    });
    return Promise.race([send, deadline]).finally(() => clearTimeout(timer));
  };
  window.rtmHealth = function () {
    return {
      ready: window.rtmReady === true, aborted: window.rtmAborted === true, rtc: window.rtcState, rtc_error: window.rtcError,
      dispatched: window.rtmStats.dispatched, delivered: window.rtmStats.delivered, failed: window.rtmStats.failed,
      last_error: window.rtmStats.lastError, last_delivered_at: window.rtmStats.lastDeliveredAt,
      publishers: Object.keys(window.remoteUsers),
    };
  };
  window.leaveAll = async function () {
    try { await client.leave(); } catch (err) { /* already gone */ }
    try { await rtm.logout(); } catch (err) { /* already gone */ }
  };
  window.roverPageReady = true;
})();
