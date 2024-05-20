import xgboost as xgb
from sklearn.metrics import accuracy_score


def train_xgboost(X_train, y_train):
    xgb_classifier = xgb.XGBClassifier(
        n_estimators=150, learning_rate=0.27, random_state=100)
    xgb_classifier.fit(X_train, y_train)
    return xgb_classifier


def evaluate_xgboost(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy
