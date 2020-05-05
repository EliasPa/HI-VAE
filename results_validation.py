import pandas as pd
import numpy as np
from string import ascii_letters
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import sys

print('Results')
print('-'*40)
header_text = 'age,albumin_apache,apache_2_diagnosis,apache_3j_diagnosis,apache_post_operative,arf_apache,bilirubin_apache,bmi,bun_apache,creatinine_apache,elective_surgery,encounter_id,ethnicity,fio2_apache,gcs_eyes_apache,gcs_motor_apache,gcs_unable_apache,gcs_verbal_apache,gender,glucose_apache,heart_rate_apache,height,hematocrit_apache,hospital_admit_source,hospital_death,hospital_id,icu_admit_source,icu_id,icu_stay_type,icu_type,intubated_apache,map_apache,paco2_apache,paco2_for_ph_apache,pao2_apache,patient_id,ph_apache,pre_icu_los_days,readmission_status,weight'
header = header_text.split(',')

path = sys.argv[1]
target_path = sys.argv[2]
results_path = "./Results/{}/{}_data_reconstruction.csv".format(path, path)
target_path = "{}".format(target_path)


reconstructed = pd.read_csv(results_path, names=header)
targets = pd.read_csv(target_path)

y_pred = reconstructed['hospital_death']
y_true = targets['hospital_death']
print(accuracy_score(y_true, y_pred))
