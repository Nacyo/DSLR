import sys
import argparse
import pandas as pd
import os.path
import matplotlib.pyplot as plt


def histograms(data):
    df1 = data[data['Hogwarts House'] == 'Ravenclaw']
    df2 = data[data['Hogwarts House'] == 'Slytherin']
    df3 = data[data['Hogwarts House'] == 'Gryffindor']
    df4 = data[data['Hogwarts House'] == 'Hufflepuff']

    data.drop(['Hogwarts House'], axis=1, inplace=True)
    labels = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    i = 1
    for col in data:
        plt.subplot(4, 4, i)
        plt.title(col)
        ax1 = plt.hist(df1[col].dropna(), alpha=0.5)
        ax2 = plt.hist(df2[col].dropna(), alpha=0.5)
        ax3 = plt.hist(df3[col].dropna(), alpha=0.5)
        ax4 = plt.hist(df4[col].dropna(), alpha=0.5)
        i += 1
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("file",
                            help="data file", type=argparse.FileType('r'))
        args = parser.parse_args()

        data = pd.read_csv(args.file)

        data.drop(['Index', 'First Name', 'Last Name',
                   'Birthday'], axis=1, inplace=True)
        histograms(data)


if __name__ == '__main__':
    main()
