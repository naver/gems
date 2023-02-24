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


class MainParser(MyParser):
    def __init__(self):
        ArgumentParser.__init__(self)

        #   ---- General parameters ----   #
        self.add_argument(
            "--exp_name", type=str, default="test_exp", help="Experiment name."
        )
        self.add_argument(
            "--run_name", type=str, default="test_run", help="Run name."
        )
        self.add_argument(
            "--data_dir", type=str, default="data/GeMS", help="Path to data/results parent directory."
        )
        self.add_argument(
            "--device", type=str, default="cpu", help="PyTorch device."
        )
        self.add_argument(
            "--seed", type=int, default=2021, help="Seed for reproducibility."
        )
        self.add_argument(
            "--progress_bar", type=self.str2bool, default=True, help="Toggle progress bar."
        )

        #   ---- Training parameters ----   #
        self.add_argument(
            "--batch_size", type=int, default=256, help="Minibatch size for ranker pretraining."
        )
        self.add_argument(
            "--max_epochs", type=int, default = 300, help = "Maximum number of training epochs."
        )
