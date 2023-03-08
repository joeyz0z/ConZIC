import numpy as np
import os
import colorlog
import random
import torch


def create_logger(folder, filename):
    log_colors = {
        'DEBUG': 'blue',
        'INFO': 'white',
        'WARNING': 'green',
        'ERROR': 'red',
        'CRITICAL': 'yellow',
    }

    import logging
    logger = logging.getLogger('ConZIC')
    # %(filename)s$RESET:%(lineno)d
    # LOGFORMAT = "%(log_color)s%(asctime)s [%(log_color)s%(filename)s:%(lineno)d] | %(log_color)s%(message)s%(reset)s |"
    LOGFORMAT = ""
    LOG_LEVEL = logging.DEBUG
    logging.root.setLevel(LOG_LEVEL)
    stream = logging.StreamHandler()
    stream.setLevel(LOG_LEVEL)
    stream.setFormatter(colorlog.ColoredFormatter(LOGFORMAT, datefmt='%d %H:%M', log_colors=log_colors))

    # print to log file
    hdlr = logging.FileHandler(os.path.join(folder, filename))
    hdlr.setLevel(LOG_LEVEL)
    # hdlr.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    hdlr.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(hdlr)
    logger.addHandler(stream)
    return logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_init_text(tokenizer, seed_text, max_len, batch_size=1):
    """ Get initial sentence by padding seed_text with [mask] words to max_len """
    text = seed_text + tokenizer.mask_token * max_len
    ids = tokenizer.encode(text)
    batch = [ids] * batch_size
    return batch

def update_token_mask(tokenizer, token_mask, max_len, index):
    """ '.'(full stop) is only allowed in the last token position """
    if index == max_len - 1:
        token_mask[:, tokenizer.vocab['.']] = 1
    else:
        token_mask[:, tokenizer.vocab['.']] = 0
    return token_mask