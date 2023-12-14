## This file includes default values for sampling frequencies, the speed of
## sound, and algorithm parameters

# Sampling frequency of the audio data
FS=250_000;

# Sampling frequency of the video data
FS_VIDEO=360;

# Speed of sound
SPEED_OF_SOUND=344.69;

# Default plotting dimensions, in pixels
DEFAULT_PLOT_DIM=(1100, 300);

###############################################################################
## Algorithm Parameters
###############################################################################

# Upper bound on the length of a chirp sequence, in samples
# approx 2 * room diameter in meters / speed of sound * sampling frequency
MAX_SEQUENCE_LENGTH = Int64(ceil(12 / SPEED_OF_SOUND * FS));