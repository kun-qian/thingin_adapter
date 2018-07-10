import time
import re

from gensim.utils import simple_preprocess

import logging


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        # print('%r (%r, %r) %2.2f sec' % (method.__name__, args, kw, te-ts))
        logging.info('%r %2.2f sec' % (method.__name__, te - ts))
        return result

    return timed


def des_sort(scores):
    des_score = sorted(enumerate(scores), key=lambda item: -item[1])
    # logging.info(des_score)
    return des_score


def split_phase(phase):
    splits = []

    if '-' in phase:
        splits = phase.split('-')
    elif '_' in phase:
        splits = phase.split('_')
    elif phase.isupper():
        splits = [phase]
    else:
        # split by capital letter
        pre_splits = re.findall('[A-Z][^A-Z]*', phase)
        pre_splits = [phase[0:phase.find(pre_splits[0])]] + pre_splits if len(pre_splits) > 0 else [phase]

        '''
        #gensim.utils.simple_preprocess will do the split by " ", "-", remove sysmbols and a little stopwords
        #**but not "_", the above codes should be added if need to split by "_"
        '''
        splits.clear()
        for one_unit in pre_splits:
            splits = splits + simple_preprocess(one_unit)

        # and remove '_|-| '
        splits = [re.sub('_|-| ', '', word) for word in splits]

    return splits


if __name__ == '__main__':
    str = "it is time for WorldCup! Isn't it great? I want a-cup-of-coffee, do you want a hello_Kitty!"
    # str = split_doc(str)
    str = split_phase(str)
    print(str)
