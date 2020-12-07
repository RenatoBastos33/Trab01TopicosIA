import statistics


def Average(list):
    sum = 0
    for num in list:
        sum += num
    return sum / len(list)


def standardDeviation(list):
    return statistics.stdev(list)


