GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

from argparse import ArgumentParser

class MyParser(ArgumentParser):
    def str2bool(self, v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
