import pandas as pd
from Visualization.Visualization import *
from Analytics.Statistics.DescriptiveStatistics import Distribution, Outlier
from sklearn.preprocessing import LabelEncoder as le


class DataCleansing:
    def __init__(self, data):
        self.data = data
        gender_le = le()
        ethnicity_le = le()
        diagnosis_le = le()

    def ethnicity(self):
        eth = {'WHITE': 'WHITE',
               'BLACK|AFRICAN': 'BLACK',
               'ASIAN': 'ASIAN',
               'HISPANIC|LATINO': 'HISPANIC',
               'UNKNOWN|PATIENT|OTHER|UNABLE|PORTUGUESE|MULTI|INDIAN|NATIVE': 'OTHERS & UNKNOWN'}
        for e in eth:
            self.data.loc[self.data['ETHNICITY'].str.contains(e), 'ETHNICITY'] = eth[e]

    def diagnosis(self):
        heart_brain_disease = {'CONGESTIVE HEART FAILURE|HEART FAILURE|CARDIAC INSUFFICIENCY': 'HEART FAILURE',  # 심부전
                               'HYPERTENSIVE|HYPERTENSION': 'HYPERTENSION',  # 고혈압
                               'DISSECTION|AORTIC DISSECTION': 'AORTIC DISSECTION',  # 대동맥 박리
                               'AORTIC STENOSIS': 'AORTIC STENOSIS',  # 대동맥 협착
                               'ANGINA PECTORIS|ANGINA': 'ANGINA',  # 협심증 : 심장에 혈액을 공급하는 혈관인 관상 동맥이 동맥 경화증으로 좁아져서 생기는 질환
                               'ARRHYTHMIA': 'ARRHYTHMIA',  # 부정맥 : 심장이 정상적으로 뛰지 않는 것(빈맥증, 서맥증)
                               'HYPOTENSION': 'HYPOTENSION',  # 고혈압
                               'STROKE|CEREBRAL INFARCTION|CEREBRAL HEMORRHAGE': 'STROKE',  # 뇌졸중
                               'STEMI|MYOCARDIAC INFARCT|MYOCARDIAL INFARCT|MYOCARDIAL INFARCTION|CARDIAL INFARCTION|CARDIAC INFARCTION|MYOCARDIAL INFARCTION': 'CARDIAC INFARCTION',
                               # 심근경색
                               'ARTERIOSCLEROSIS': 'ARTERIOSCLEROSIS',  # 동맥 경화증
                               'VALVE|VALVULAR|VALVE DISEASE|VALVULAR DISEASE|VALVULAR HEART': 'VALVULAR DISEASE', # 판막증
                               'HEART ATTACK|CARDIAC ARREST': 'CARDIAC ARREST',  # 심장마비
                               'CORONARY|CORONARY ARTERY|CORONARY ARTERY BYPASS': 'CORONARY ARTERY DISEASE',  # 관상동맥 질병
                               'TACHYCARDIA|VENTRICULAR TACHYCARDIA': 'VENTRICULAR TACHYCARDIA',  # 심실빠른맥
                               'CARDIOMYOPATHY|CMS': 'CARDIOMYOPATHY',  # 심근증
                               'CHEST PAIN': 'CHEST PAIN',  # 흉통
                               # 'STEMI': 'STEMI'  # ST분절 상승 심근경색
                               }
        sorted_hb_disease = sorted(list(heart_brain_disease.keys()), key=len, reverse=True)
        for h in sorted_hb_disease:
            self.data.loc[self.data['DIAGNOSIS'].str.contains(h), 'reDIAGNOSIS'] = heart_brain_disease[h]  # 1650
        self.data.loc[~self.data['DIAGNOSIS'].str.contains(
            '|'.join(list(heart_brain_disease.keys()))), 'reDIAGNOSIS'] = 'NON CARDIAC DISEASE'

    def create_bmi(self):
        self.data['BMI'] = self.data['WEIGHT'] / (self.data['HEIGHT'] / 100) ** 2

    def remove_age_mask(self):
        self.data.loc[self.data['AGE'] >= 300, 'AGE'] = self.data.loc[self.data['AGE'] >= 300, 'AGE'] - 210


if __name__ == "__main__":
    patient_df = pd.read_csv('/home/paperc/PycharmProjects/MIMIC/MIMIC_Clinical_Database/III/result/patients.csv')

    DataCleansing(patient_df).ethnicity()
    DataCleansing(patient_df).diagnosis()
    DataCleansing(patient_df).create_bmi()
    DataCleansing(patient_df).remove_age_mask()
    # patient_df = patient_df.loc[patient_df['SBP'].isna() == False]
    # patient_df = patient_df.loc[patient_df['DBP'].isna() == False]

    # sbp_test = np.random.randn(1990) / 3 * 10 + 80
    # dbp_test = np.random.randn(1990) / 3 * 20 + 60
    sbp_test = patient_df['SBP'].values
    dbp_test = patient_df['DBP'].values

    Distribution(patient_df.loc[patient_df['AGE'].isna() == False]['AGE'].values).hist(plot_normal=False)
    Distribution(patient_df.loc[patient_df['HEIGHT'].isna() == False]['HEIGHT'].values).hist()
    Distribution(patient_df.loc[patient_df['WEIGHT'].isna() == False]['WEIGHT'].values).hist()
    Distribution(patient_df.loc[patient_df['BMI'].isna() == False]['BMI'].values).hist()

    heatmap(patient_df[['SBP', 'DBP', 'AGE', 'ETHNICITY', 'HEIGHT', 'WEIGHT', 'BMI']])

    out_test = Outlier(sbp_test).iqr_test()
    print(out_test)
    std_test = Outlier(sbp_test).std_test()
    print(std_test)
    maha_test = Outlier(sbp_test).get_mahalanobis_dis(2)
    print(maha_test)
