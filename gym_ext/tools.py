import numpy as np

def warning_clip_obs(o, ranges):
    if np.all(ranges[0] <= o) and np.all(o <= ranges[1]):
        return o
    else:
        print "WARNING: Clipped observation %s." % o
        return np.clip(o, ranges[0], ranges[1])


# Note: A functor!
def apply_to_snd(map_obs, o):
    return (o[0], map_obs(o[1]))
