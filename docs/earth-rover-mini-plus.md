# Earth Rover Mini+

The FrodoBots Earth Rover Mini+ is a mobile base evaluated with the same
`colosseum-robot` flow as the arms: the Router assigns a navigation task, the
client drives the rover from the policy's commands, the operator scores the
trial, and the recording is exported and uploaded as a LeRobot dataset.

## Contract

| | |
|---|---|
| `robot_type` | `earth_rover_mini_plus`, 0 joints, no gripper |
| Action space | `base_velocity`: `[linear, angular]` normalized to `[-1, 1]`; `+angular` turns left; 10 Hz |
| Cameras | `front_image` (required), `rear_image` (optional, must be delivered every step when configured) |
| Policy state | `navigation_state` (float64): `latitude, longitude, heading_deg, speed, battery, rpm_1..rpm_4` |
| GPS goal | `goal_gps = [latitude, longitude]` in the state |
| Photo goal | a `goal_image` sensor (the reference photo); no GPS is sent |
| Completion | the policy sends a zero chunk with `done=True`; the trial also ends at `navigation.timeout_s` (default 300 s) or when the operator presses Enter |

Goals come from the bot's FrodoBots missions, the way the SDK works: the
client lists the bot's SDK-enabled active missions, the operator picks one
(or passes `--mission SLUG`), and the destination is one of its checkpoints
(the last, or `--checkpoint N`). The ride is opened on that mission, so it is
a normal recorded ride. Fine-tuning tasks are registered from a mission
(`goal.mission_slug` ties them together; the client matches the task to the
chosen mission); Open Track trials send the checkpoint goal directly.
`--goal-gps LAT,LON` / `--goal-image PATH` remain for destinations that are
not a mission checkpoint, such as photo goals, which are kept on the operator
laptop (`adapter_config.goal_dir`) and checked against the task's sha256.

## Safety

The rover keeps executing the last command it received, so the adapter owns a
10 Hz sender thread that re-sends the latest action and replaces it with a stop
once no new action arrived within `command_ttl_s` (1 s). A stalled policy
therefore produces stop-and-go motion rather than a runaway. Observations are
rejected when the front frame is older than 2 s or telemetry older than 3 s;
the trial then ends with an error and the adapter closes, which sends a stop and
waits for the rover's receipt. An unconfirmed stop is reported as an error and
the ride is left open for the operator to check the rover.

## Install and run

```bash
uv sync --extra earth-rover            # Agora RTC/RTM client; Linux x86_64 for real runs, macOS 13+ for development
cp configs/robot.earth_rover_mini_plus.yaml.example configs/robot.yaml
export FRODOBOTS_SDK_TOKEN=...         # organizer sdk_access_token of a non-admin account
uv run colosseum-robot configs/robot.yaml --no-execute-action      # first physical run: observe only; lists the bot's missions
uv run colosseum-robot configs/robot.yaml --mission courtyard-loop --checkpoint 3
uv run colosseum-robot configs/robot.yaml --track open --goal-image goals/courtyard-fountain.jpg
```

The ride is acquired through frodobots_web's programmatic endpoints
(`/sdk/missions`, `/sdk/checkpoints_list`, `/sdk/start_ride`, `/sdk/end_ride`)
with the configured `bot_slug` and the chosen mission, and released when the
adapter closes; a trial without a mission falls back to the SDK's token-only
session (`/sdk/token`), which has no ride and no platform-side recording.

The rover itself is reached the way the Earth Rovers SDK does it: a headless
Google Chrome page (`robots/earth_rover/vendor/rover.html`, driven by
Playwright) joins the ride's Agora RTC channel for the cameras and logs into
Agora RTM 1.x for commands and telemetry (`transport: browser`, the default).
Measured on a Mini: ~10 fps per camera with 50 ms median frame age, telemetry
every 0.5 s, commands dispatched in ~4 ms and the stop acknowledged in ~0.5 s.
Google Chrome is preferred because it decodes the rover's H.264 stream; set
`chrome_executable_path` for another codec-capable browser. `transport: native`
uses the Agora Python SDK instead, but its RTM client is 2.x and cannot receive
the rover's 1.x telemetry. Video arrives over the ride's Agora channel (front camera uid 1000,
rear 1001); commands and telemetry are RTM peer messages to and from the rover.
`scripts/earth_rover_link_check.py` exercises the link without the Router
(frame age, telemetry rate, command round trip, confirmed stop) and is the
first thing to run on a new rover or operator machine.

## Operator page

`colosseum-rover-ui configs/robot.yaml` opens http://127.0.0.1:8770 and works
like the SDK dashboard: enter the bot and press **Connect** (a token-only
session: live front camera and telemetry, no ride yet), pick one of its
missions and the destination checkpoint, choose the track and whether the
rover may move, and **Start session**. A trial switches that same connection
to a ride on the mission and returns to the preview when it ends. The page
runs exactly the terminal flow, so every question (start trial, score 0–4,
preference) appears as a form; it also shows whether the Router is reachable,
an abort button for the running trial, a stop button for the session, and the
submitted results with links to the Router admin and review API. It listens on
localhost only; the policy server still runs separately.

The live view is the SDK dashboard's: the page joins the ride's Agora channel
in your browser with the **spectator** identity the backend issues alongside
the ride (`SPECTATOR_USERID` / `SPECTATOR_RTC_TOKEN`) and plays the front and
rear tracks directly, at the rover's frame rate, while the link's headless page
keeps `USERID` for commands, telemetry and the policy's frames. Audio is
subscribed but silent until you press Unmute. A link without spectator
credentials (the simulator) falls back to snapshots.

The page also carries the SDK dashboard's map and drive controls (Leaflet 1.9.4
is vendored in `rover_ui_static/`; the tiles come from OpenStreetMap):

- **Map**: the rover's position and heading from telemetry, a breadcrumb trail,
  the selected mission's checkpoints (the destination pulses; 15 m circles),
  the trial's GPS goal, and the distance left to the destination. "Follow
  rover" keeps it centred; dragging the map turns that off.
- **Drive**: d-pad or WASD/arrows while the rover is connected and no trial is
  running (also between the trials of a session, to bring the rover back to
  the start). A held key streams the command at 10 Hz and release sends one
  zero, like the SDK; the operator process also zeros the rover if the page
  goes silent for 0.5 s (tab closed, browser frozen). The speed slider scales
  the ±1 command, **Lamp** toggles the headlamp bit carried by every command,
  and **STOP** (or Space) sends a confirmed stop. When a trial adapter holds
  the connection the controls are disabled and STOP aborts the trial instead.
  `+angular` turns left, as in the FrodoBots command.

### Diagnostics

The Diagnostics card shows what the link sees: whether it is streaming and on which channel (preview or ride),
frames received per camera with the age of the newest one and the rate reaching the policy, telemetry age,
which identities the rover is publishing (1000 front, 1001 rear), RTM command delivery counts, what this
page's own video is playing, and the headless page's recent errors. Below it, timestamped events record
connects, ride switches, trial start and end, and failures with a stream summary. **Copy diagnostics** puts the
full dump on the clipboard; paste it into the issue when something fails.

Two `adapter_config` knobs bound how old the newest data may be before a trial ends: `front_max_age_s`
(default 2) and `telemetry_max_age_s` (default 3). Raise them a little on a weak 4G link; a policy must not act
on a view seconds behind the rover, and the command watchdog stops the rover `command_ttl_s` after the last
policy action regardless.

When a session starts, the shared link leaves the preview channel and joins the ride's channel; the first
observation waits for a frame from the new channel. When the trial ends, the ride is released, the link waits
for the platform to clear the rover's channel, then requests the preview again (and retries once); if the rover
does not come back, Disconnect and Connect.

## Scoring

At the end of each trial the operator enters the V4 score (0–4). It maps to the
Router's fields as `partial_success = score / 4` and `success = score == 4`:
4 means the policy declared done, held still for `stopped_s`, arrived, and had
no collision, boundary violation or human help.

## Test mode and dry runs

- `test: true` uses a synthetic rover (no link, goals ignored): it exercises the
  full Router flow and the export with a 0-joint, 2-D action recording.
- `adapter_config.link: simulated` runs the real adapter, watchdog and recording
  against an in-memory rover.

## Writing a policy

```python
front = obs.image("front_image").as_rgb()
state = obs.raw_state["navigation_state"]      # [lat, lon, heading_deg, speed, battery, rpm_1..4]
goal = obs.raw_state.get("goal_gps")           # GPS goals
photo = obs.image("goal_image")                # photo goals
sdk.send_action(np.array([[0.3, 0.0]] * 10), observation=obs)       # drive
sdk.send_action(np.zeros((1, 2)), observation=obs, done=True)       # arrived
```

Register the policy with `action_spaces: [base_velocity]`, `control_hz: 10` and
a horizon of at least a few steps so the watchdog does not stop the rover
between chunks. The policy server's `examples/rover_policy.py` is a working
minimal policy.
