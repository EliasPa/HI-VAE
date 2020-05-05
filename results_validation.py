import pandas as pd
import numpy as np
from string import ascii_letters
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import sys

print('Results')
print('-'*40)
path = sys.argv[1]
target_path = sys.argv[2]
N_variables = int(sys.argv[3])

response_variable = 'hospital_death'
original_data_path = './data/widsdatathon2020/training_v2.csv'
original_data = pd.read_csv(original_data_path)

header = sorted(list(original_data.columns)[:N_variables])

results_path = "./Results/{}/{}_data_reconstruction.csv".format(path, path)
target_path = "{}".format(target_path)

reconstructed = pd.read_csv(results_path, names=header)
targets = pd.read_csv(target_path)

y_pred = reconstructed[response_variable]
y_true = targets[response_variable]
print(f"Accuracy score for `{response_variable}`: {accuracy_score(y_true, y_pred)}\n")
