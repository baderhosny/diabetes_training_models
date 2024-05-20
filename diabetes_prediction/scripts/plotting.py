import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def accuracy_report(lg_acc_one, lg_acc_two, dtree_acc_one, dtree_acc_two, rfc_acc_one, rfc_acc_two, xgb_acc_one, stacking_acc):
    data = {
        'Logistic Regression One': lg_acc_one*100,
        'Logistic Regression Two': lg_acc_two*100,
        'Decision Tree One': dtree_acc_one*100,
        'Decision Tree Two': dtree_acc_two*100,
        'Random Forest One': rfc_acc_one*100,
        'Random Forest Two': rfc_acc_two*100,
        'XGBoost One': xgb_acc_one*100,
        'Stacking Classifier': stacking_acc*100
    }

    data_df = pd.DataFrame(data, index=['Class 1'])
    data_df

    plt.figure(figsize=(14, 6))
    sns.barplot(data=data_df)
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.title("Model Metrics for Diabetic Prediction")
    plt.show()
