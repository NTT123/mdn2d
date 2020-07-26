"""Logger class."""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


class Logger:
    """Logger class."""

    def __init__(self, root_dirname):
        self.step = 0
        self.log = defaultdict(list)
        self.root_dirname = Path(root_dirname)
        if not self.root_dirname.exists():
            self.root_dirname.mkdir(parents=True, exist_ok=True)

    def add_scalar(self, name, value):
        self.log[name].append(value)

    def plot(self, output_file):
        plt.clf()
        log = list(self.log.items())
        L = len(log)
        plt.figure(figsize=(15, 2 * L))
        for i, (k, v) in enumerate(log):
            plt.subplot(L, 1, i + 1)
            plt.plot(v)
            plt.title(k)
        plt.subplots_adjust(hspace=0.5)

        plt.savefig(str(self.root_dirname / output_file))
        plt.close()
