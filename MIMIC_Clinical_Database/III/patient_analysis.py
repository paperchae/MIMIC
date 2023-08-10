import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import wfdb
import gzip
import shutil
import os

raw_data_path = '/Users/paperc/Desktop/Datasets/mimic-iii-clinical-database-1.4/'


def uom_converter(measure_type, inputs):
    if measure_type == 'height':
        return inputs * 2.54
    elif measure_type == 'weight':
        return inputs * 0.453592
    elif measure_type == 'temperature':
        return (inputs - 32) * 5 / 9


def get_item_id(label):
    items = pd.read_csv(raw_data_path + 'D_ITEMS.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
    # labels = ['Height', 'Weight']
    # for label in labels:
    label_df = items[items['LABEL'].str.contains(label, na=False)][['ITEMID', 'LABEL', 'DBSOURCE', 'UNITNAME']]

    return label_df


# height_id = get_item_id('Weight').tail(2)
# weight_id = get_item_id('Weight').tail(2)


def read_csv(file_name):
    chunk_size = 10 ** 6

    for chunk in pd.read_csv(raw_data_path + file_name, chunksize=chunk_size, compression='gzip', header=0, sep=',',
                             quotechar='"'):
        # height_cm_mv = chunk[chunk['ITEMID'] == 226730]
        # height_inches_mv = chunk[chunk['ITEMID'] == 226707]
        # has_height = chunk[chunk['ITEMID'] == 226730]['SUBJECT_ID'].values
        # daily_weight_kg_mv = chunk[chunk['ITEMID'] == 224639]
        # admission_weight_kg_mv = chunk[chunk['ITEMID'] == 226512]
        # admission_weight_lbs_mv = chunk[chunk['ITEMID'] == 226531]

        print(chunk.head())
        print('break')


def loc(df, column, condition, value):
    if condition == 'eq':
        return df.loc[df[column] == value]
    elif condition == 'in':
        return df.loc[df[column].isin(value)]


def read_patients(subject_list):
    patient_df = pd.read_csv(raw_data_path + 'PATIENTS.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
    patient_df = loc(patient_df, 'SUBJECT_ID', 'in', subject_list)
    admission_df = pd.read_csv(raw_data_path + 'ADMISSIONS.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
    admission_df = loc(admission_df, 'SUBJECT_ID', 'in', subject_list)
    chart_df = pd.read_csv(raw_data_path + 'CHARTEVENTS.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
    chart_df = loc(chart_df, 'SUBJECT_ID', 'in', subject_list)
    # df = df.loc[df['SUBJECT_ID'].isin(subject_list), ['GENDER', 'DOB', 'EXPIRE_FLAG']]
    return patient_df, admission_df


patients = read_patients([249, 700])

read_csv('PATIENTS.csv.gz')
