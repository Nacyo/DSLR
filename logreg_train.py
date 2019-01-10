import sys
import argparse
import pandas as pd
from logistic_regression import *


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--function",
                            default='GD',
                            choices=['GD', 'MBGD'],
                            help='Optimizaion function choice, gradient decent, or mini batch GD')
        parser.add_argument("-i", "--iterations",
                            help="set number of iterations", type=int)
        parser.add_argument("-a", "--alpha",
                            help="set step size", type=float)
        parser.add_argument("-v", "--verbose",
                            action='store_true',
                            help="print loss info during training")
        parser.add_argument("-bs", "--batch_size",
                            help="set batch size for MBGD", type=int)
        parser.add_argument("file",
                            help="data file", type=argparse.FileType('r'))

        args = parser.parse_args()

        model = LogisticRegression()

        if args.iterations:
            model.nb_iter = args.iterations
            print("Number of iteration set at {:d}".format(model.nb_iter))
        if args.alpha:
            model.alpha = args.alpha
            print("Alpha set at {:f}".format(model.alpha))
        if args.function:
            model.opti_func = args.function
            print("Optimizaion function choice: {:s}".format(model.opti_func))
        if args.verbose:
            model.verbose = args.verbose
            print("Verbose option selected")
        if args.batch_size:
            model.batch_size = args.batch_size
            print("Batch size choice: {:d}".format(model.batch_size))
            print("Be sure to add -f MBGD for any effect on training")

        data = pd.read_csv(args.file)

        df, y = prepare_data(data)
        model.train(df, y)


if __name__ == '__main__':
    main()
