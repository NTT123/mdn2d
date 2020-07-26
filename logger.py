"""Logger class."""

import matplotlib.pyplot as plt

from pathlib import Path


class Logger:
    """Logger class."""

    def __init__(self, root_dirname):
        self.step = 0
        self.log = {}
        self.root_dirname = Path(root_dirname)
        if not self.root_dirname.exists():
            self.root_dirname.mkdir(parents=True, exist_ok=True)

    def add_scalar(self, name, value):
        if not name in self.log:
            self.log[name] = []

        self.log[name].append(value)

    def plot(self, output_file):
        plt.clf()
        plt.close()
        log = list(self.log.items())
        L = len(log)
        plt.figure(figsize=(10, 4*L))
        for i, (k, v) in enumerate(log):
            plt.subplot(L, 1, i+1)
            plt.plot(v)
            plt.title(k)
        plt.savefig(str(self.root_dirname/output_file))
