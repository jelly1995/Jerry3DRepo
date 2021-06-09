import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Read pickle file.')
    parser.add_argument('--file_path', help='path to pickle file', required=False, default='/home/jerry/Desktop/NuScenesDataset/mini/infos_train.pkl')

    args = parser.parse_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


args = parse_args()
filename = args.file_path

pickle_file = open(filename, 'rb')
new_dict = pickle.load(pickle_file)
pickle_file.close()

print(new_dict)