import pandas as pd
import numpy as np
from string import ascii_letters
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import sys
import os

# Read arguments
N_variables = int(sys.argv[1])
n_sample = int(sys.argv[2])
split = int(sys.argv[3])

directory = './Results'
suffix = "data_reconstruction.csv"
prefix = f"model_HIVAE_inputDropout_ICU_{N_variables}"

target_path = f"./ICU/dataset_{N_variables}_vars/{n_sample}_N_{split}_split/test_target.csv"


def get_results(results_path, target_path, N_variables):

    response_variable = 'hospital_death'
    original_data_path = './data/widsdatathon2020/training_v2.csv'
    original_data = pd.read_csv(original_data_path)

    header = sorted(list(original_data.columns)[:N_variables])

    target_path = "{}".format(target_path)

    reconstructed = pd.read_csv(results_path, names=header)
    targets = pd.read_csv(target_path)

    y_pred = reconstructed[response_variable]
    y_true = targets[response_variable]

    return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred)


def get_results_for_all():
    results = {}
    models = []
    accs = []
    tests = []
    for folder in os.listdir(directory):
        prefix = os.path.join(directory, folder)
        if ('.DS_Store' not in prefix):
            print(os.path.join(directory, folder))
            for filename in os.listdir(os.path.join(directory, folder)):
                if suffix in filename:
                    file_path = os.path.join(prefix, filename)

                    result = get_results(file_path, target_path, N_variables)
                    models.append(folder)
                    accs.append(result[0])
                    tests.append(result[1])

    result_df = pd.DataFrame(
        {'model': models, 'accuracy': accs, 'recall': tests})
    return result_df


results = get_results_for_all()
save_path = './data/test_results.csv'
results.to_csv(save_path, index=False)
