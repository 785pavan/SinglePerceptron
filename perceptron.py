import csv
import numpy as np

Address1 = "Example.tsv"

with open(Address1) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    i = 0
    rdata = []
    for n in csv_reader:
        if len(n) != 0:
            n3 = []
            for n2 in n:
                if len(n2) != 0:
                    n3.append(n2)
            rdata.append(n3)
number_of_column = len(rdata[0])
number_of_rows = len(rdata)
print(number_of_rows, number_of_column, rdata)


def alphabet_to_output(column):
    class1 = []
    for ato1 in column:
        if ato1 == "A":
            class1.append(1)
        elif ato1 == "B":
            class1.append(0)

    return class1


def row_to_column(data):
    number_of_column1 = len(data[0])
    number_of_rows1 = len(data)
    col_array = []
    col_array_class = []
    for rtc2 in range(number_of_rows1):
        col_array_class.append(data[rtc2][0])
    # col_array.append(col_array2)

    # temp = 0
    for rtc1 in range(1, number_of_column1):
        col_array2 = []
        for rtc2 in range(number_of_rows1):
            temp = data[rtc2][rtc1]
            col_array2.append(float(temp))
        col_array.append(col_array2)

    return col_array, col_array_class


# print(row_to_column(rdata[1]))
[data_col, class_col] = row_to_column(rdata)
# class_col = data_col[0]
print(alphabet_to_output(class_col))
class_col_binary = alphabet_to_output(class_col)
print(data_col)


def error_func(output, data_column_class):
    sum_err = 0
    err_array_ef = []
    # print(len(output), len(data_column_class))
    for ef1 in range(len(output)):
        if output[ef1] != data_column_class[ef1]:
            sum_err += 1
        err_array_ef.append(data_column_class[ef1] - output[ef1])
    return sum_err, err_array_ef


o_test = []
for o in range(0, number_of_rows):
    o_test.append(0)
print(o_test)

print(error_func(o_test, class_col_binary))
print(error_func([0, 0, 1, 1], [0, 1, 0, 1]))
[sum_error, err_array] = error_func(o_test, class_col_binary)


def update_weights_const(data_column, previous_weight_array, error):
    print("error:", error)
    number_of_column2 = len(data_column)
    number_of_rows2 = len(data_column[0])
    # new_weight0 = []
    s_err = sum(error)
    # error_func(o_test, class_col_binary)
    new_weight0 = previous_weight_array[0] + s_err

    new_weights = list()
    new_weights.append(new_weight0)

    for uw1 in range(number_of_column2):
        err_sum1 = 0
        for uw2 in range(number_of_rows2):
            err_sum1 += error[uw2] * data_col[uw1][uw2]
        new_weight = previous_weight_array[uw1 + 1] + err_sum1
        new_weights.append(new_weight)

    return new_weights


print(update_weights_const(data_col, [0, 0, 0], err_array))


# print(update_weights_const(data_col, [200, 0, 0], err_array))


def update_weights_annealing(data_column, previous_weight_array, error, iter_up_weights):
    print("error:", error)
    number_of_column2 = len(data_column)
    number_of_rows2 = len(data_column[0])
    # new_weight0 = []
    s_err = sum(error)
    # error_func(o_test, class_col_binary)
    new_weight0 = previous_weight_array[0] + (1 / iter_up_weights) * s_err

    new_weights = list()
    new_weights.append(new_weight0)

    for uw1 in range(number_of_column2):
        err_sum1 = 0
        for uw2 in range(number_of_rows2):
            err_sum1 += (1 / iter_up_weights) * (error[uw2] * data_col[uw1][uw2])
        new_weight = previous_weight_array[uw1 + 1] + err_sum1
        new_weights.append(new_weight)

    return new_weights


def activation_perceptron(data_column, weights):
    number_of_column3 = len(data_column)
    number_of_rows3 = len(data_column[0])
    out_array = []
    for ap1 in range(number_of_rows3):
        check1 = 0
        for ap2 in range(number_of_column3):
            next_sum = data_column[ap2][ap1] * weights[ap2 + 1]
            check1 += next_sum
        check1 += weights[0]
        if check1 > 0:
            out_array.append(1)
        else:
            out_array.append(0)

    return out_array


print(activation_perceptron(data_col, [200, 191.36959200000007, 236.86229100000003]))
print(sum(activation_perceptron(data_col, [200, 191.36959200000007, 236.86229100000003])))

out_arr1 = activation_perceptron(data_col, [200, 191.36959200000007, 236.86229100000003])
[sum_error, err_array] = error_func(out_arr1, class_col_binary)
print(sum_error)
print(update_weights_const(data_col, [200, 191.36959200000007, 236.86229100000003], err_array))
out_arr1 = activation_perceptron(data_col, [190, -163.3488430000001, 227.67655900000005])
[sum_error, err_array] = error_func(out_arr1, class_col_binary)
print(sum_error)


# class PerceptronGradientDecent(object):
#
#     def __init__(self, iteration1, learning_rate):
#
#         self.iteration1 = iteration1
#         self.learning_rate = learning_rate
#
#     def predictions(self, rows, weights):
#
#         return 1 + rows + weights


def iteration_final(data, iterations):
    number_of_rows4 = len(rdata)
    out_error_array = []
    iter1 = 0
    [data_col_f, class_col_f] = row_to_column(data)
    class_col_binary_f = alphabet_to_output(class_col_f)

    o_zeros = []
    for o2 in range(0, number_of_rows4):
        o_zeros.append(0)
    # print(o_zeros)

    [sum_error_f, err_array_f] = error_func(o_zeros, class_col_binary_f)
    updated_ws0 = update_weights_const(data_col_f, [0, 0, 0], err_array_f)
    out_error_array.append(sum_error_f)
    updated_ws = updated_ws0
    for itf1 in range(iterations):
        new_out = activation_perceptron(data_col_f, updated_ws)
        [sum_error_f, err_array_f] = error_func(new_out, class_col_binary_f)
        updated_ws = update_weights_const(data_col_f, updated_ws, err_array_f)
        out_error_array.append(sum_error_f)
    # [sum_error, err_array] = error_func(out_arr1, class_col_binary)

    out_error_array2 = []
    iter1 = 0
    [data_col_f, class_col_f] = row_to_column(data)
    class_col_binary_f = alphabet_to_output(class_col_f)

    o_zeros = []
    for o2 in range(0, number_of_rows4):
        o_zeros.append(0)
    # print(o_zeros)

    [sum_error_f, err_array_f] = error_func(o_zeros, class_col_binary_f)
    updated_ws0 = update_weights_annealing(data_col_f, [0, 0, 0], err_array_f, 1)
    out_error_array2.append(sum_error_f)
    updated_ws = updated_ws0
    for itf1 in range(iterations):
        new_out = activation_perceptron(data_col_f, updated_ws)
        [sum_error_f, err_array_f] = error_func(new_out, class_col_binary_f)
        updated_ws = update_weights_annealing(data_col_f, updated_ws, err_array_f, itf1 + 2)
        out_error_array2.append(sum_error_f)

    return out_error_array, out_error_array2


[ot1, ot2] = iteration_final(rdata, 100)
print(ot1)
print(ot2)

with open('sol_ML_assignment03.tsv', 'w', newline='') as outfile:
    sol_file = csv.writer(outfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    A1 = []
    A2 = []
    for fi in range(len(ot1)):
        A1.append((ot1[fi]))
    sol_file.writerow(A1)
    A2.append(A1)
    A1 = []
    for fi in range(len(ot1)):
        A1.append((ot2[fi]))
    A2.append(A1)
    sol_file.writerow(A1)
