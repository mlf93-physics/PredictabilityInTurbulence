from pathlib import Path
import numpy as np

def import_header(folder="", file_name=None, old_header=False):
    path = Path(folder, file_name)

    # Import header
    header = ""
    header_size = 1 if not old_header else 3

    with open(path, 'r') as file:
        for i in range(header_size):
            header += file.readline().rstrip().lstrip().strip('#').strip().replace(' ', '')
        header = header.split(',')
    # print('header', header)
    header_dict = {}
    for item in header:
        splitted_item = item.split("=")
        if splitted_item[0] == "f":
            header_dict[splitted_item[0]] = np.complex(splitted_item[1])
        else:
            header_dict[splitted_item[0]] = float(splitted_item[1])

    print('header_dict', header_dict)

    return header_dict

def import_data(file_name, old_header=False, skip_lines=0, max_rows=None):
    
    # Import header
    header_dict = import_header(file_name=file_name, old_header=old_header)

    # Import data
    data_in = np.genfromtxt(file_name,
        dtype=np.complex, delimiter=',', skip_header=skip_lines,
        max_rows=max_rows)

    return data_in, header_dict