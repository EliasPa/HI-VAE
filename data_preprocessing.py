import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from string import ascii_letters
import time
import os
import sys
from sklearn.model_selection import train_test_split

training_data_path = './data/widsdatathon2020/training_v2.csv'
test_data_path = './data/widsdatathon2020/unlabeled.csv'
data_types_path = './data/widsdatathon2020/WiDS Datathon 2020 Dictionary.csv'

training_data = pd.read_csv(training_data_path)

print('-'*40)
print('Started generating datasets for training and test')

response_variable = training_data['hospital_death']
new_ratio = len(
    response_variable[response_variable == 1]) / len(response_variable)
print("Ratio of deaths/survival in training data",
      new_ratio, training_data.shape)


# Default values
N = 8000
N_variables = 40
test_train_split = 50

if(len(sys.argv) == 4):
    N_variables = int(sys.argv[1])
    N = int(sys.argv[2])
    test_train_split = int(sys.argv[3])
elif(len(sys.argv) == 3):
    N_variables = int(sys.argv[1])
    N = int(sys.argv[2])
elif(len(sys.argv) == 2):
    N_variables = int(sys.argv[1])
else:
    print('-'*40)
    print('No arguments given, using default values')

print('-'*40)
print('Number of variables used: {}'.format(N_variables))
print('Sample size per class used: {}'.format(N))
print('Test-train- split: {} %'.format(test_train_split))
print('-'*40)


death = training_data[training_data['hospital_death'] == 1]
survive = training_data[training_data['hospital_death'] == 0]

data_folder = './ICU/dataset_{}_vars/{}_N_{}_split'.format(
    N_variables, N, int(test_train_split))

try:
    os.mkdir(data_folder)
except FileExistsError as e:
    print('-'*40)
    print('Folder already exists, continue writing the files')


training_data = pd.concat(
    [death.iloc[:N, :N_variables], survive.iloc[:N*2, :N_variables]])

print('-'*40)
print('Shape of the training data')
print(training_data.shape)
print('-'*40)
# training_data = training_data.iloc[N:, :N_variables]
data_split = test_train_split / 100.0
print(data_split)
training_data, test_data = train_test_split(
    training_data, test_size=data_split, stratify=training_data['hospital_death'])

test_data = test_data.iloc[:, :N_variables]

print('-'*40)
print('Shape of the training data')
print(test_data.shape)
print('-'*40)


print('Testi Ennen')
print(test_data['hospital_death'].sum())

####################################
# Save targets for test data
####################################
print('-'*40)
print('Saving targets for test data')
print('-'*40)
test_target_y = test_data['hospital_death']

test_target_path = '{}/test_target.csv'.format(data_folder)
test_target_y.to_csv(test_target_path, index=False, header=True)


# Set hospital_death-column to None in test
test_data['hospital_death'] = None
# print('Testi Jälkeen')
# print(test_data['hospital_death'].sum())
training_data = training_data.reindex(sorted(training_data.columns), axis=1)


def createMaskCSV(data):
    mask = data.isna()
    # data = data.fillna(0)

    mask_dict = []
    L = mask.shape[0]

    rows = []
    cols = []
    for i in range(mask.shape[0]):
        start_time = time.time()
        row = mask.to_numpy()[i]
        for j, val in enumerate(row):
            if val:
                rows.append(i+1)
                cols.append(j+1)
        # print("{0} out of {1} took: {2}".format(
            # i, L, time.time() - start_time))

    dic = {'row': rows, 'column': cols}
    mask_df = pd.DataFrame(dic)

    return mask_df, data


print('-'*40)
print('Creting mask for the training data')
print('-'*40)
mask_df, training_data = createMaskCSV(training_data)

mask_df_path = '{}/Missingxx_y.csv'.format(data_folder)
mask_df.to_csv(mask_df_path, index=False, header=False)

training_data_preprocessed_path = '{}/training_data_preprocessed.csv'.format(
    data_folder)
training_data.to_csv(training_data_preprocessed_path, index=False, header=True)


# Sanity check
training_data_check = pd.read_csv(training_data_preprocessed_path)

# Just to make sure that the data is feasible
# print('death_count', training_data_check['hospital_death'].sum())
# print(training_data_check.dtypes)

# Select object columns
cat_columns = training_data_check.select_dtypes(['object']).columns
# Cast to category
training_data_check[cat_columns] = training_data_check[cat_columns].astype(
    'category')
# Replace with integer coding
training_data_check[cat_columns] = training_data_check[cat_columns].apply(
    lambda x: x.cat.codes + 1)

# Overwrite
training_data_preprocessed_path = '{}/training_data_preprocessed_cleaned.csv'.format(
    data_folder)
training_data_check = training_data_check.fillna(0)


print('-'*40)
print('Saving preprocessed training data')
print('-'*40)
training_data_check.to_csv(
    training_data_preprocessed_path, index=False, header=False)
print(training_data['hospital_death'].sum())


# Create mask for test data
print('-'*40)
print('Creting mask for the test data')
print('-'*40)
test_data = test_data.reindex(sorted(test_data.columns), axis=1)
test_mask, test_data = createMaskCSV(test_data)

test_mask_path = '{}/test_Missingxx_y.csv'.format(data_folder)
test_mask.to_csv(test_mask_path, index=False, header=False)

test_data_preprocessed_path = '{}/test_data_preprocessed.csv'.format(
    data_folder)
test_data.to_csv(test_data_preprocessed_path, index=False, header=True)

# Sanity check
test_data = pd.read_csv(test_data_preprocessed_path)
test_data['hospital_death']

# Select object columns
cat_columns = test_data.select_dtypes(['object']).columns
# Cast to category
test_data[cat_columns] = test_data[cat_columns].astype('category')
# Replace with integer coding
test_data[cat_columns] = test_data[cat_columns].apply(lambda x: x.cat.codes+1)

# Overwrite
test_data_preprocessed_path = '{}/test_data_preprocessed_cleaned.csv'.format(
    data_folder)
test_data = test_data.fillna(0)
print('-'*40)
print('Saving preprocessed test data')
print('-'*40)
test_data.to_csv(test_data_preprocessed_path, index=False, header=False)

index_of_response = training_data.columns.get_loc('hospital_death')
# print("Index of response variable: ", index_of_response)

print('Done, results in {}'.format(data_folder))
print('-'*40)
