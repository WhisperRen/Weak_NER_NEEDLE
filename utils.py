import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence


SEED = 99

IGNORE_CHARACTERS = ('', ' ', '\n', '\r\n', '\t', '\u2003', '\u2002', '\u3000')


def setup_seed():
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # some cudnn methods can be random even after fixing the seed unless force it to be deterministic
    torch.backends.cudnn.deterministic = True
