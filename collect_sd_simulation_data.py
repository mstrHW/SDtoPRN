import pysd
import argparse

from definitions import path_join, make_directory, VENSIM_MODELS_DIR, DATA_DIR
from module.data_loader import np_preproc_for_rnn2d
from module.fd_model.vensim_fd_converter import get_fd, KNOWN_MODEL, UNKNOWN_MODEL


def generate_sd_output(vensim_model_file):
    model = pysd.read_vensim(vensim_model_file)
    data = model.run()
    return data


def get_sd_components(data):
    fields = data.columns
    general_stopwords = ['INITIAL TIME', 'FINAL TIME', 'SAVEPER', 'TIME STEP', 'Time', 'TIME']
    stopwords = ['predator births', 'predator deaths', 'prey births', 'prey deaths', 'Heat Loss to Room']
    fields = [key for key in fields if key not in general_stopwords]
    fields = [key for key in fields if key not in stopwords]
    return fields


def generate_train_data(fields, data):
    dataset = data[fields].as_matrix()
    # dataset = dataset/abs(dataset).max()
    # dataset = preprocessing.normalize(dataset)
    # dataset = preprocessing.scale(dataset)
    return dataset, np_preproc_for_rnn2d(dataset)


def main(args):
    model_name = args.model_name
    dataset_file_name = args.dataset_file_name
    mode = KNOWN_MODEL

    dataset_dir = path_join(DATA_DIR, model_name)
    make_directory(dataset_dir)

    dataset_path = path_join(dataset_dir, dataset_file_name)
    vensim_model_file = path_join(VENSIM_MODELS_DIR, '{}.mdl'.format(model_name))

    FD = get_fd(vensim_model_file, mode=mode)
    data = generate_sd_output(vensim_model_file)

    fields = [level for level in FD.names_units_map.keys()]
    fields.insert(0, 'TIME')

    dataset = data[fields]

    dataset.to_csv(dataset_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="",
    )

    parser.add_argument(
        "--dataset_file_name",
        type=str,
        help="",
    )

    args = parser.parse_args()
    main(args)
