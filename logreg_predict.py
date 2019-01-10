import sys
import argparse
import pandas as pd
import numpy as np
import os.path
import pickle
import csv
from logistic_regression import *


def main():
        parser = argparse.ArgumentParser()

        parser.add_argument("-e", "--erase", action='store_true',
                            help="reinit trained model")
        parser.add_argument("file",
                            help="data file", type=argparse.FileType('r'))
        args = parser.parse_args()

        filename = 'finalized_model.sav'

        if args.erase is False and os.path.isfile(filename):
            loaded_model = pickle.load(open(filename, 'rb'))
            print("Model loaded")
        elif args.erase and os.path.isfile(filename) is False:
            print("Model does not exist cannot erase")
            sys.exit()
        else:
            if args.erase:
                os.remove(filename)
            print("Model erased or does not exist")
            print("Please train your model before continuing")
            sys.exit()

        data = pd.read_csv(args.file)
        pd.set_option('expand_frame_repr', False)

        df, _ = prepare_data(data)

        print("Prediction in process")
        prediction = loaded_model.one_vs_all(df.values)

        prediction_class = convert_result(prediction)

        with open('houses.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["Index", "Hogwarts House"])
            for row in range(0, len(prediction_class)):
                myList = [row]
                myList.append(prediction_class[row])
                writer.writerow(myList)

        print("DONE")


if __name__ == '__main__':
    main()
