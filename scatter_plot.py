import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def scatter_plot(data):
    data.dropna(inplace=True)

    df1 = data[data['Hogwarts House'] == 'Ravenclaw']
    df2 = data[data['Hogwarts House'] == 'Slytherin']
    df3 = data[data['Hogwarts House'] == 'Gryffindor']
    df4 = data[data['Hogwarts House'] == 'Hufflepuff']

    data.drop(['Hogwarts House'], axis=1, inplace=True)
    labels = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    fig, ax = plt.subplots()
    plt.title('Astronomy / Defense Against the Dark Arts')
    ax1 = plt.scatter(df1['Defense Against the Dark Arts'], df1['Astronomy'], alpha=0.5)
    ax2 = plt.scatter(df2['Defense Against the Dark Arts'], df2['Astronomy'], alpha=0.5)
    ax3 = plt.scatter(df3['Defense Against the Dark Arts'], df3['Astronomy'], alpha=0.5)
    ax4 = plt.scatter(df4['Defense Against the Dark Arts'], df4['Astronomy'], alpha=0.5)
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
        scatter_plot(data)


if __name__ == '__main__':
    main()
