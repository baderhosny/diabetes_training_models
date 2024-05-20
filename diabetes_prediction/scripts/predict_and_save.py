import os
import pandas as pd


def predict_and_save_results(stacking_classifier):
    file_path = input("Enter the path to the CSV file: ")
    data = pd.read_csv(file_path)

    if data.isnull().values.any():
        data.dropna(inplace=True)

    features = ['Pregnant', 'Glucose', 'Diastolic_BP', 'Skin_Fold',
                'Serum_Insulin', 'BMI', 'Diabetes_Pedigree', 'Age']
    X_new = data[features]

    predictions = stacking_classifier.predict(X_new)
    data['result'] = predictions
    data['result'] = data['result'].map({1: True, 0: False})

    output_file_name = input(
        "Enter the name of the CSV file to save (without extension): ")
    output_directory = 'diabetes_prediction/output'
    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(
        output_directory, output_file_name + ".csv")

    data.to_csv(output_file_path, index=False)
    print("Prediction results saved successfully as {}".format(output_file_path))
