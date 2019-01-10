import sys
import argparse
import pandas as pd
import os.path
import utils_stats


def apply_funcs(arr_val):
    funcs = [utils_stats.count_c, utils_stats.mean_c, utils_stats.std_c,
             utils_stats.min_c, utils_stats.quartiles_c, utils_stats.max_c]

    result = []
    for func in funcs:
        x = func(arr_val[1:])
        if isinstance(x, tuple):  # tuple if quartiles function called
            for val in x:
                result.append(val)
        else:
            result.append(x)
    return result


def describe_data(data):
    values = []
    to_print = []
    col_names = []

    for col in data:
        col_names.append(col)
        nb_null = 0
        values = []
        for row in data[col]:
            x = row
            if not utils_stats.isnan(x):
                values.append(x)
            else:
                nb_null += 1
        values = apply_funcs(values) + [nb_null]
        to_print.append(values)
    to_print = list(map(list, zip(*to_print)))
    return to_print, col_names


def print_descrition(data, col_names):
    pd.set_option('display.expand_frame_repr', False)
    dat1 = pd.DataFrame({'Functions': ['Count', 'Mean', 'Std', 'Min',
                                       '25%', '50%', '75%', 'Max', 'NullSum']})
    dat2 = pd.DataFrame(data, columns=col_names)
    print(dat1.join(dat2))


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("file",
                            help="data file", type=argparse.FileType('r'))
        args = parser.parse_args()
        data = pd.read_csv(args.file)
        data.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name',
                   'Birthday', 'Best Hand'], axis=1, inplace=True)
        result, col_names = describe_data(data[1:])
        print_descrition(result, col_names)


if __name__ == '__main__':
    main()
