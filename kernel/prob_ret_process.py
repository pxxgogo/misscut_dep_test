import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('--output_path', type=str, default="log_filtered.txt")
    parser.add_argument('--threshold', type=int, default=50)

    args = parser.parse_args()
    threshold = args.threshold
    input_path = args.input_path
    output_path = args.output_path

    output_handle = open(output_path, 'w')
    input_handle = open(input_path)
    line = input_handle.readline()
    buffer = []
    write_flag = False
    while line:
        if line[0] == '#':
            if len(buffer) > 0:
                if write_flag:
                    write_flag = False
                    output_handle.writelines(buffer)
                buffer = []
            output_handle.write(line)
        # line = line.strip()
        elif line[0:2] == 'B:' or line[0:2] == 'D:':
            if len(buffer) > 0:
                if write_flag:
                    write_flag = False
                    output_handle.writelines(buffer)
                buffer = []
            buffer.append(line)
        else:
            if line == "\n":
                buffer.append(line)
            else:
                items = line.strip().split("\t")
                score = int(items[3][1:-2])
                if score < threshold:
                    write_flag = True
                buffer.append(line)
        line = input_handle.readline()
    if len(buffer) > 0:
        if write_flag:
            write_flag = False
            output_handle.writelines(buffer)
        buffer = []
    output_handle.close()
    input_handle.close()








