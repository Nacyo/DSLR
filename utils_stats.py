def sum_c(X):
    result = 0
    for x in X:
        if isinstance(x, tuple):
            print(x)
        result += float(x)
    return result


def min_c(X):
    min = X[1]
    for x in X:
        if x < min:
            min = x
    return min


def max_c(X):
    max = X[0]
    for x in X:
        if x > max:
            max = x
    return max


def mean_c(X):
    result = sum_c(X) / len(X)
    return result


def count_c(X):
    result = 0
    for x in X:
        if x is None:
            continue
        result += 1
    return result


def squared_c(X):
    result = []
    for x in X:
        result.append(x * x)
    return result


def std_c(X):
    esp = mean_c(X)
    result = []
    for x in X:
        result.append(x - esp)
    return (sum_c(squared_c(result)) / len(X)) ** (1 / 2)


def median_c(X, lenght):
    if lenght % 2 == 0:
        median = X[lenght // 2 - 1] / 2 + X[lenght // 2 + 1] / 2
    else:
        median = X[(lenght - 1)//2]
    return median


def quartiles_c(X):
    X.sort()
    lenght = len(X)
    q2 = median_c(X, lenght)
    first_half = X[:lenght // 2 + 1]
    second_half = X[(lenght + 1) // 2:]
    q1 = median_c(first_half, len(first_half))
    q3 = median_c(second_half, len(second_half))
    return q1, q2, q3


def isnan(x):
    return x != x
