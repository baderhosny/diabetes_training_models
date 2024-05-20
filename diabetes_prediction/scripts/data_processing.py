import pandas as pd
from sklearn.model_selection import train_test_split

file_path = 'diabetes_prediction/data/Diabetes.csv'


def load_data(file_path):
    return pd.read_csv(file_path)


def clean_data(df):
    df = df[df['Pregnant'] <= 10]
    nan_count = df.isna().sum(axis=1)
    df.dropna(thresh=df.shape[1]-2, inplace=True)
    df.dropna(subset=['Glucose', 'Serum_Insulin'], how='all', inplace=True)
    return df


def approach_one(df):
    df_approach_one = df.dropna(axis=0, how='any')
    df_approach_one.to_csv('Diabetes_approach_one.csv', index=False)
    return df_approach_one


def calculate_glucose_means(df):
    normal_glucose_level_mean = df.loc[df['Glucose']
                                       <= 99, 'Serum_Insulin'].mean()
    prediabetes_glucose_level_mean = df.loc[(df['Glucose'] >= 100) & (
        df['Glucose'] <= 125), 'Serum_Insulin'].mean()
    diabetes_glucose_level_mean = df.loc[df['Glucose']
                                         >= 126, 'Serum_Insulin'].mean()
    return normal_glucose_level_mean, prediabetes_glucose_level_mean, diabetes_glucose_level_mean


def approach_two(df, means):
    df_approach_two = df.copy()
    normal_glucose_level_mean, prediabetes_glucose_level_mean, diabetes_glucose_level_mean = means
    df_approach_two.loc[(df['Glucose'] <= 99) & df['Serum_Insulin'].isna(
    ), 'Serum_Insulin'] = normal_glucose_level_mean
    df_approach_two.loc[(df['Glucose'] >= 100) & (df['Glucose'] <= 125) &
                        df['Serum_Insulin'].isna(), 'Serum_Insulin'] = prediabetes_glucose_level_mean
    df_approach_two.loc[(df['Glucose'] >= 126) & df['Serum_Insulin'].isna(
    ), 'Serum_Insulin'] = diabetes_glucose_level_mean
    df_approach_two.to_csv('Diabetes_approach_two.csv', index=False)
    return pd.read_csv('Diabetes_approach_two.csv')


def prepare_data(df):
    df = df.drop('Skin_Fold', axis=1)
    return df.dropna(subset=['Glucose', 'Diastolic_BP', 'BMI'])


def data_splitting(df_approach_one, df_approach_two):
    X1 = df_approach_one.drop('Class', axis=1)
    Y1 = df_approach_one['Class']
    X2 = df_approach_two.drop('Class', axis=1)
    Y2 = df_approach_two['Class']
    return train_test_split(X1, Y1, test_size=0.33, random_state=42), train_test_split(X2, Y2, test_size=0.33, random_state=42)
