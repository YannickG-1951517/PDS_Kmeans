# ==============================================================================
# Imports
# ==============================================================================
import sys
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Do stuff
# ==============================================================================
def main():
    functions_data = [data_pds_kmeans_omp_exectime, data_pds_kmeans_omp_speedup]
    functions_graph = [graph_line_error_bars, mult_graph_line_error_bars]

    if (len(sys.argv) < 4 or sys.argv[1] == '-h' or sys.argv[1] == '--help' or sys.argv[1] == 'help'):
        print('Usage: python3 graph.py <file> <data function index> <graph function index> [graph title] [x label] [y label]\n<> = required, [] = optional')
        return
    
    file = sys.argv[1]
    data_func_index = int(sys.argv[2])
    graph_func_index = int(sys.argv[3])
    graph_title = sys.argv[4] if len(sys.argv) > 4 else None
    x_label = sys.argv[5] if len(sys.argv) > 5 else None
    y_label = sys.argv[6] if len(sys.argv) > 6 else None

    (x, y) = get_data_from_file(file, functions_data[data_func_index])
    draw_graph(x, y, functions_graph[graph_func_index], graph_title, x_label, y_label)






# ==============================================================================
# Get data
# ==============================================================================
def get_data(input, func):
    return func(input)

def get_data_from_file(file, func):
    with open(file, 'r') as f:
        input = f.read()
    return get_data(input, func)

def data_pds_kmeans_omp_exectime(input):
    input = input.split('------------------------------------------------------------------------------')
    input = input[1]
    input = input.split('\n')
    input = input[1:-1]
    x = []
    y = []
    for i, line in enumerate(input):
        if (line.startswith('omp')):
            if i+1 < len(input):
                x.append(int(line.split(' try')[0].split(' ')[-1]))
                y.append(float(input[i+1].split(',')[-1]))
    return x, y

def data_pds_kmeans_omp_speedup(input):
    input = input.split('------------------------------------------------------------------------------')
    input = input[1]
    input = input.split('\n')
    input = input[1:-1]
    t_serial = float(input[0].split(' ')[0])
    s = float(input[1].split(' ')[0])
    p = float(input[2].split(' ')[0])
    x = []
    y1 = []
    y2 = []
    y3 = []
    for i, line in enumerate(input):
        if (line.startswith('omp')):
            if i+1 < len(input):
                # convert to int
                n = int(line.split(' try')[0].split(' ')[-1])
                x.append(n)
                y1.append(t_serial/float(input[i+1].split(',')[-1]))
                y2.append((t_serial*n)/t_serial)
                y3.append(t_serial/(s+(p/n)))
    return x, [['Gemeten speedup', True, y1], ['Maximum speedup', False, y2], ['Theoretische speedup', False, y3]]



# ==============================================================================
# Make graph
# ==============================================================================

def draw_graph(x, y, func, title=None, x_label=None, y_label=None):
    func(x, y, title, x_label, y_label)


def graph_line_error_bars(x_values, y_values, title=None, x_label=None, y_label=None):

    unique_x = np.unique(x_values)
    avg_y = []
    std_dev = []
    for x in unique_x:
        y_subset = []
        for x_val, y_val in zip(x_values, y_values):
            if x == x_val:
                y_subset.append(y_val)
        avg_y.append(np.mean(y_subset))
        std_dev.append(np.std(y_subset))

    print(np.unique(x_values))
    print(avg_y)
    print(std_dev)
    plt.errorbar(unique_x, avg_y, yerr=std_dev, label='Data', capsize=5,  markeredgewidth=2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()

def mult_graph_line_error_bars(x_values, y_values, title=None, x_label=None, y_label=None):

    unique_x = np.unique(x_values)
    avg_y = []
    std_dev = []
    for y_vals in y_values:
        avg_y = []
        std_dev = []
        for x in unique_x:
            y_subset = []
            for x_val, y_val in zip(x_values, y_vals[2]):
                if x == x_val:
                    y_subset.append(y_val)
            avg_y.append(np.mean(y_subset))
            std_dev.append(np.std(y_subset))
        if y_vals[1]:
            plt.errorbar(unique_x, avg_y, yerr=std_dev, label=y_vals[0], capsize=5,  markeredgewidth=2)
        else:
            plt.errorbar(unique_x, avg_y, label=y_vals[0])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()










main()