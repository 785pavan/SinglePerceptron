import csv
import getopt
import sys


def get_data(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        i = 0
        data = []
        for line in csv_reader:
            if line.__len__() != 0:
                row = []
                for col in line:
                    if col.__len__() != 0:
                        row.append(col)
                data.append(row)

    return data, get_col(data), get_row(data)


def get_col(data):
    return data[0].__len__()


def get_row(data):
    return data.__len__()


def classify(column):
    classification = []
    for value in column:
        if value == "A":
            classification.append(1)
        elif value == "B":
            classification.append(0)
    return classification


def transpose(data):
    len_col = data[0].__len__()
    len_row = data.__len__()
    col_array = []
    col_array_class = []
    for i in range(len_row):
        col_array_class.append(data[i][0])
    for j in range(1, len_col):
        col_array2 = []
        for i in range(len_row):
            temp = data[i][j]
            col_array2.append(float(temp))
        col_array.append(col_array2)
    return col_array, col_array_class


def error_func(output, data_column_class):
    sum_err = 0
    err_array = []
    for i in range(len(output)):
        if output[i] != data_column_class[i]:
            sum_err += 1
        err_array.append(data_column_class[i] - output[i])
    return sum_err, err_array


def update_weights_const(data_column, previous_w, error):
    print("error:", error)
    len_col = len(data_column)
    len_row = len(data_column[0])
    sum_error = sum(error)
    w0 = previous_w[0] + sum_error

    new_ws = list()
    new_ws.append(w0)
    for i in range(len_col):
        err_sum = 0
        for j in range(len_row):
            err_sum += error[j] * data_col[i][j]
        new_weight = previous_w[i + 1] + err_sum
        new_ws.append(new_weight)

    return new_ws


def update_weights_annealing(data_column, previous_weight_array, error, iter_up_weights):
    print("error:", error)
    len_col = data_column.__len__()
    len_row = data_column[0].__len__()
    sum_error = sum(error)
    w0 = previous_weight_array[0] + (1 / iter_up_weights) * sum_error
    new_ws = list()
    new_ws.append(w0)
    for i in range(len_col):
        sum_error = 0
        for j in range(len_row):
            sum_error += (1 / iter_up_weights) * (error[j] * data_col[i][j])
        new_weight = previous_weight_array[i + 1] + sum_error
        new_ws.append(new_weight)
    return new_ws


def activation_function(data_column, weights):
    len_col = data_column.__len__()
    len_row = data_column[0].__len__()
    out_array = []
    for i in range(len_row):
        sigma_w = 0
        for j in range(len_col):
            next_sum = data_column[j][i] * weights[j + 1]
            sigma_w += next_sum
        sigma_w += weights[0]
        if sigma_w > 0:
            out_array.append(1)
        else:
            out_array.append(0)
    return out_array


def single_perceptron(data, iterations):
    len_row = data.__len__()
    output_error_const = []
    [col_data, data_class] = transpose(data)
    class_col = classify(data_class)
    o_zeros = []
    for o2 in range(0, len_row):
        o_zeros.append(0)
    [sum_error, err_array] = error_func(o_zeros, class_col)
    print("----------------------------------error values for constant learning rate----------------------------------")
    updated_ws0 = update_weights_const(col_data, [0, 0, 0], err_array)
    output_error_const.append(sum_error)
    updated_ws = updated_ws0
    for i in range(iterations):
        new_out = activation_function(col_data, updated_ws)
        [sum_error, err_array] = error_func(new_out, class_col)
        updated_ws = update_weights_const(col_data, updated_ws, err_array)
        output_error_const.append(sum_error)
    output_error_anneal = []
    [col_data, data_class] = transpose(data)
    class_col = classify(data_class)
    o_zeros = []
    for o2 in range(0, len_row):
        o_zeros.append(0)
    [sum_error, err_array] = error_func(o_zeros, class_col)
    print("---------------------------------error values for annealing learning rate----------------------------------")
    updated_ws0 = update_weights_annealing(col_data, [0, 0, 0], err_array, 1)
    output_error_anneal.append(sum_error)
    updated_ws = updated_ws0
    for i in range(iterations):
        new_out = activation_function(col_data, updated_ws)
        [sum_error, err_array] = error_func(new_out, class_col)
        updated_ws = update_weights_annealing(col_data, updated_ws, err_array, i + 2)
        output_error_anneal.append(sum_error)
    return output_error_const, output_error_anneal


def write_data(row1, row2, filename):
    with open(filename, 'w', newline='') as outfile:
        output_file = csv.writer(outfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        line = []
        for value in row1:
            line.append(value)
        output_file.writerow(line)
        line = []
        for value in row2:
            line.append(value)
        output_file.writerow(line)
        print("Output file written in " + filename)


if __name__ == '__main__':
    argv = sys.argv[1:]
    iter_req = 0
    outputfile = ""
    inputfile = ""
    if argv.__len__() != 0:
        try:
            opts, args = getopt.getopt(argv, "hit:o:", ["data=", "output=", "iter="])
        except getopt.GetoptError:
            print('usage: perceptron.py -i||--data <inputfile> -o||--output <outputfile>')
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print('test.py -i <inputfile> -o <outputfile>')
                sys.exit()
            elif opt in ("-i", "--data"):
                inputfile = arg
                if '.tsv' not in inputfile:
                    inputfile = inputfile + '.tsv'
            elif opt in ("-o", "--output"):
                outputfile = arg
                if '.tsv' not in outputfile:
                    outputfile = outputfile + '.tsv'
            elif opt in ('-t', "--iter"):
                iter_req = int(arg)
    else:
        inputfile = input("Enter data file name: ")
        if '.tsv' not in inputfile:
            inputfile = inputfile + '.tsv'
        outputfile = input("Enter Output file name: ")
        if '.tsv' not in outputfile:
            outputfile = outputfile + '.tsv'
    if inputfile == "":
        inputfile = input("Enter data file name: ")
        if '.tsv' not in inputfile:
            inputfile = inputfile + '.tsv'
    if outputfile == "":
        outputfile = input("Enter Output file name: ")
        if '.tsv' not in outputfile:
            outputfile = outputfile + '.tsv'
    data, col_len, row_len = get_data(inputfile)
    [data_col, data_class] = transpose(data)
    if iter_req == 0:
        iter_req = 100
    [out_const_learning, out_anneal_learning] = single_perceptron(data, iter_req)
    print("Error of each iteration with the constant learning rate:")
    print(out_const_learning)
    print("Error of each iteration with the annealing learning rate:")
    print(out_anneal_learning)
    write_data(out_const_learning, out_anneal_learning, outputfile)
