import pysd
import numpy as np

from arch.Loader import Loader
import definitions


def generate_data(vensim_model_file):
    model = pysd.read_vensim(vensim_model_file)
    data = model.run()
    fields = data.columns
    general_stopwords = ['INITIAL TIME', 'FINAL TIME', 'SAVEPER', 'Time', 'TIME']
    # stopwords = ['predator births', 'predator deaths', 'prey births', 'prey deaths', 'Heat Loss to Room']
    fields = [key for key in fields if key not in general_stopwords]
    # fields = [key for key in fields if key not in stopwords]
    return data[fields], fields


def get_pysd_simulation(vensim_model_file, test_X, names):
    # model_name='model5'
    # models_directory = '../vensim_models/'
    # vensim_model_file = models_directory + '{}.mdl'.format(model_name)
    model = pysd.read_vensim(vensim_model_file)

    answers = []
    for x in test_X:
        if len(x) > 0:
            x = x.astype(float)
            patient_id = [int(float(x[0, 0]))]
            initial_value = x[0, 1:]
            iterations_count = x.shape[0]+1
            return_timestamps = np.arange(0, iterations_count * 12, 12)
            condition_map = {}
            params = {}
            for i in range(x.shape[1]-1):
                if names[i] == 'condition' or names[i] == 'pressure':
                    params[names[i]] = initial_value[i]
                else:
                    condition_map[names[i]] = initial_value[i]

            answer = model.run(initial_condition=(0, condition_map), return_timestamps=return_timestamps, params=params)
            answer = answer[names]
            answer = answer.values[1:iterations_count]
            tmp = []
            for ans in answer:
                tmp_line = [patient_id[0]]
                for column in ans:
                    tmp_line.append(column)
                tmp_line = np.array(tmp_line)
                tmp.append(tmp_line)
            tmp = np.array(tmp)

            answers.append(tmp)
    answers = np.array(answers)
    return answers


def get_pysd_simulation_v2(vensim_model_file, test_X, names):
    # model_name='model5'
    # models_directory = '../vensim_models/'
    # vensim_model_file = models_directory + '{}.mdl'.format(model_name)
    model = pysd.read_vensim(vensim_model_file)

    answers = []
    for x in test_X:
        if len(x) > 0:
            answers_j = []
            for j in x:
                j = j.astype(float)
                patient_id = [int(j[0])]
                initial_value = j[1:]
                iterations_count = 1+1
                return_timestamps = np.arange(12, iterations_count * 12, 12)
                condition_map = {}
                params = {}
                for i in range(initial_value.shape[0]):
                    if names[i] == 'condition' or names[i] == 'pressure':
                        params[names[i]] = initial_value[i]
                    else:
                        condition_map[names[i]] = initial_value[i]

                answer = model.run(initial_condition=(0, condition_map), return_timestamps=return_timestamps, params=params)
                answer = answer[names].values[0]
                [patient_id.append(out) for out in answer]
                # answer = answer.values[1:iterations_count]
                answers_j.append(np.array(patient_id))
            answers.append(answers_j)
    answers = np.array(answers)
    return answers


def main():
    pass


if __name__ == '__main__':
    main()
    pass

