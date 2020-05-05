from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from string import ascii_letters
import time
import os
import sys

training_data_path = './data/widsdatathon2020/training_v2.csv'
data_types_path = './data/widsdatathon2020/WiDS Datathon 2020 Dictionary.csv'
training_data = pd.read_csv(training_data_path)

print('-'*40)
print('Started generating data_types-file')

N_variables = 40
if(len(sys.argv) > 1):
    N_variables = int(sys.argv[1])
else:
    print('-'*40)
    print('No argument given for number of variables, using default value')


print('-'*40)
print('Number of variables used: {}'.format(N_variables))
print('-'*40)

training_data = training_data.iloc[:, :N_variables]
training_data = training_data.reindex(sorted(training_data.columns), axis=1)


def createMaskCSV(data):
    mask = data.isna()
    data = data.fillna(0)

    mask_dict = []
    L = mask.shape[0]

    rows = []
    cols = []
    for i in range(mask.shape[0]):
        start_time = time.time()
        row = mask.to_numpy()[i]
        for j, val in enumerate(row):
            if val:
                rows.append(i)
                cols.append(j)
        # print("{0} out of {1} took: {2}".format(
        #     i, L, time.time() - start_time))

    dic = {'row': rows, 'column': cols}
    mask_df = pd.DataFrame(dic)

    return mask_df, data


mask_df, training_data = createMaskCSV(training_data)

training_data_preprocessed_path = './data/training_data_preprocessed.csv'

################################################################################
# Replacing object datatype with category in training data
################################################################################

training_data_check = training_data

# Select object columns
cat_columns = training_data_check.select_dtypes(['object']).columns
# Cast to category
training_data_check[cat_columns] = training_data_check[cat_columns].astype(
    'category')
# Replace with integer coding
training_data_check[cat_columns] = training_data_check[cat_columns].apply(
    lambda x: x.cat.codes)

# Overwrite
training_data_preprocessed_path = './data/training_data_preprocessed_cleaned.csv'
# training_data_check.to_csv(
#     training_data_preprocessed_path, index=False, header=True)


############################
# Creating data_types-file
############################

data_types_path = './data/widsdatathon2020/WiDS Datathon 2020 Dictionary.csv'

data_types_raw = pd.read_csv(data_types_path)
data_types_raw = data_types_raw.sort_values('Data Type')
variable_names = data_types_raw['Variable Name']
column_names = training_data.columns

# Numerical values
count = 'count'
positive = 'pos'
real = 'real'

# Nominal values
categorical = 'cat'
ordinal = 'ordin'

variable_names = np.intersect1d(variable_names, column_names)
minimums = training_data.select_dtypes(include='float').describe().iloc[3, :]
only_real = 'pre_icu_los_days'

print('Selected variables')
print(variable_names)
# minimums
size = list(range(len(variable_names)))
data = {'name': variable_names, 'type': size, 'dim': size, 'nclass': size}
data_types_with_name = pd.DataFrame(data)

print('-'*40)
print('Creating data_types from WiDS Datathon 2020 Dictionary.csv')
print('-'*40)

for i in size:
    col_name = variable_names[i]
    col_type = data_types_raw[data_types_raw['Variable Name']
                              == col_name]['Data Type'].iloc[0]

    col_data = training_data[col_name]

    data_type = ''
    if col_type == 'binary':
        data_type = categorical
    elif col_type == 'integer':
        data_type = count
    elif col_type == 'string':
        data_type = categorical
    elif col_type == 'numeric':
        if col_name != only_real:
            data_type = positive
        else:
            data_type = real

    unique = col_data.unique()
    data_nclass = data_dim = 1
    if (data_type == categorical):
        unique = unique[~pd.isnull(unique)]
        data_dim = data_nclass = len(unique)
    if (col_name == 'bmi'):
        data_nclass = data_dim = 1
        data_type = positive

    data_types_with_name['type'].iloc[i] = data_type
    data_types_with_name['dim'].iloc[i] = data_dim
    data_types_with_name['nclass'].iloc[i] = data_nclass

    # print(col_name, data_type, data_dim, data_nclass)

folder_path = './ICU/dataset_{}_vars'.format(N_variables)

print('Creating directory for data_types')


try:
    os.mkdir(folder_path)
except FileExistsError as e:
    print('-'*40)
    print('Folder already exists, proceeding to writing the files')

print('-'*40)
print('Writing files "data_types" and "data_types_with_name" to {}'.format(folder_path))
print('-'*40)

data_types_path = '{}/data_types.csv'.format(folder_path)

data_types = data_types_with_name.iloc[:, 1:]
data_types.to_csv(data_types_path, index=False, header=True)

data_types_path = '{}/data_types_with_name.csv'.format(folder_path)
data_types_with_name.to_csv(data_types_path, index=False, header=True)
print('Done, results in {}'.format(folder_path))
print('-'*40)
