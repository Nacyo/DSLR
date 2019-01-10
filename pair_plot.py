import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pair_plot(data):
    data.dropna(inplace=True)
    sns.set(style="ticks")
    g = sns.pairplot(data, hue="Hogwarts House", kind='scatter',
                     diag_kind='hist', plot_kws=dict(alpha=0.5), size=3)
    plt.show()


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("file",
                            help="data file", type=argparse.FileType('r'))
        args = parser.parse_args()
        data = pd.read_csv(args.file)
        data.drop(['Index', 'First Name', 'Last Name',
                   'Birthday'], axis=1, inplace=True)
        pair_plot(data)


if __name__ == '__main__':
    main()
