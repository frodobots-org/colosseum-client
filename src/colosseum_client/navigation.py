"""Earth Rover Mini+ contract shared by the adapter, the recorder export and tests. No hardware imports."""

ROVER_TYPE = 'earth_rover_mini_plus'
ACTION_SPACE = 'base_velocity'
# Normalized commands in [-1, 1]; positive angular turns left (counter-clockwise).
ACTION_NAMES = ('linear', 'angular')
# Policy state vector, in this order, as float64 (latitude/longitude need the precision).
NAV_STATE_NAMES = ('latitude', 'longitude', 'heading_deg', 'speed', 'battery', 'rpm_1', 'rpm_2', 'rpm_3', 'rpm_4')
NAV_STATE_KEY = 'navigation_state'
GOAL_GPS_KEY = 'goal_gps'
FRONT_CAMERA = 'front_image'
REAR_CAMERA = 'rear_image'
GOAL_IMAGE = 'goal_image'
ROVER_CAMERAS = (FRONT_CAMERA, REAR_CAMERA)
