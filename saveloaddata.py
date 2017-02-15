import contextlib
import glob
from os import path
import pickle

import numpy as np

@contextlib.contextmanager
def load_res(algo, interr, data_dir_p):
    glob_s = "{}-{}-xe-*.pickle".format(algo, interr)
    all_episode_lengths = []
    all_xss             = []

    for p in glob.glob(path.join(data_dir_p, glob_s)):
        with open(p, 'rb') as f:
            episode_lengths, xss = pickle.load(f)

        all_episode_lengths.append(episode_lengths)
        all_xss += xss

    yield np.vstack(all_episode_lengths), all_xss
