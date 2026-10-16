# Run Earth Rover evaluations

The operator runs the Earth Rovers SDK sidecar and `colosseum-rover`; staff run Astra in the policy server. Researchers use the normal policy SDK and never receive rover credentials.

1. Prepare the intended bot: a non-admin organizer SDK account, recording enabled, an SDK-enabled open mission with zero checkpoints, and a marked start/destination. Keep an operator beside the rover.
2. Start the updated SDK on localhost with the intended `BOT_SLUG`, `MISSION_SLUG` and `CONTROL_WATCHDOG_S=0.5`. The FrodoBots API must provide exact `ROVER_IDENTITY` fields. A legacy API/SDK is rejected before motion.
3. Copy `configs/earth-rover.yaml.example`. Set the URLs, institution matching the client token, intended bot slug and local SDK URL. Use HTTPS/WSS outside localhost. Supply the token through `COLOSSEUM_CLIENT_TOKEN`. Set qualified speed/turn limits after the site's stop and network-loss checks; a bot ID alone is not sufficient setup.
4. Install this package and ffmpeg. Run `colosseum-rover --config /private/path/rover.yaml --arm`. Without `--arm`, the runner can connect and display status but cannot execute a trial. A physical session also requires `qualified: true`.
5. In `/admin/earth-rover`, select rover, photo mission and Astra. Confirm the start pose/operator, keep Test enabled, then Start. Stop requests a confirmed stop and records the termination reason. Enter the 0–4 rubric after reviewing arrival and heading.
6. For Open comparisons, register and connect a second compatible policy. Reset the same rover to the documented start before run B. Fine-tuning uses an eligible policy bound to a registered task and its matching navigation contract. Astra itself is the Open photo baseline.
7. Choose End rover session after the final run. Keep the runner open until uploads finish. Several runs share one FrodoBots ride; each result stores its own UTC interval.

The control loop runs independently at 10 Hz. Stale camera/telemetry, missing admin heartbeats, invalid actions or expired chunks result in zero commands. The SDK adds a transport watchdog. Completion requires a confirmed stop receipt and fresh speed/RPM evidence over the stationary interval. A disconnect disarms the runner; inspect the rover and restart explicitly before another trial. A local watchdog cannot guarantee physical stopping after rover network loss; qualification is required.

Per-run files include exact policy image observations, policy inputs/outputs, bounded commands, private full telemetry, front/rear review videos, identity, configuration and health. Review uploads are checksum verified. Native DRV/GPS/IMU/HLS data is exported after the ride closes and linked to the same run. Set the optional native exporter configuration and AWS read profile/role. GPS and judge data stay in private evidence.

If the laptop restarts, retry uploads for **ended rides only** without opening another ride:

```sh
colosseum-rover --config /private/path/rover.yaml --archive-only
```

Never delete local evidence until its server-side checks pass. Failed model inference remains a scoreable policy outcome; missing/failed infrastructure is marked invalid and kept for investigation. Upload failure leaves the result and local evidence intact for retry.

For software-only operation use `--simulate --arm`; it never contacts the configured SDK. All simulated runs are forced to Test. The router's `scripts/demo_earth_rover.py --auto-run` provides a complete local demonstration with fixed Astra responses.
