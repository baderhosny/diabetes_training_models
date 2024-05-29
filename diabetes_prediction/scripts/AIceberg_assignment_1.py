import os
import pandas as pd
import pickle
from data_processing import load_data, clean_data, approach_one, calculate_glucose_means, approach_two, prepare_data, data_splitting
from model_training import train_logistic_regression, train_decision_tree, train_random_forest, evaluate_model
from xgboost_training import train_xgboost, evaluate_xgboost
from ensemble_learning import train_stacking_classifier
from plotting import accuracy_report
from predict_and_save import predict_and_save_results

if __name__ == "__main__":
    input_file = 'diabetes_training_models/diabetes_prediction/data/Diabetes.csv'
    df = load_data(input_file)
    df = clean_data(df)
    df_approach_one = approach_one(df)
    means = calculate_glucose_means(df)
    df_approach_two = approach_two(df, means)
    df_approach_two = prepare_data(df_approach_two)

    (X1_train, X1_test, y1_train, y1_test), (X2_train, X2_test, y2_train,
                                             y2_test) = data_splitting(df_approach_one, df_approach_two)

    # Train models and get accuracy
    log_model_one = train_logistic_regression(X1_train, y1_train)
    log_model_two = train_logistic_regression(X2_train, y2_train)
    lg_acc_one, _ = evaluate_model(log_model_one, X1_test, y1_test)
    lg_acc_two, _ = evaluate_model(log_model_two, X2_test, y2_test)

    dtree_model_one = train_decision_tree(X1_train, y1_train)
    dtree_model_two = train_decision_tree(X2_train, y2_train)
    dtree_acc_one, _ = evaluate_model(dtree_model_one, X1_test, y1_test)
    dtree_acc_two, _ = evaluate_model(dtree_model_two, X2_test, y2_test)

    rfc_model_one = train_random_forest(X1_train, y1_train)
    rfc_model_two = train_random_forest(X2_train, y2_train)
    rfc_acc_one, _ = evaluate_model(rfc_model_one, X1_test, y1_test)
    rfc_acc_two, _ = evaluate_model(rfc_model_two, X2_test, y2_test)

    xgb_model_one = train_xgboost(X1_train, y1_train)
    xgb_acc_one = evaluate_xgboost(xgb_model_one, X1_test, y1_test)

    stacking_classifier = train_stacking_classifier(X1_train, y1_train)
    stacking_acc, _ = evaluate_model(stacking_classifier, X1_test, y1_test)

    # Save the stacking classifier
    os.makedirs('models', exist_ok=True)
    with open('diabetes_training_models/diabetes_prediction/models/stacking_classifier.pkl', 'wb') as f:
        pickle.dump(stacking_classifier, f)

    accuracy_report(lg_acc_one, lg_acc_two, dtree_acc_one, dtree_acc_two,
                    rfc_acc_one, rfc_acc_two, xgb_acc_one, stacking_acc)
    predict_and_save_results(stacking_classifier)
