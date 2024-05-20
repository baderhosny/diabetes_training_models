from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def train_stacking_classifier(X_train, y_train):
    base_learners = [
        ('rfc', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('dtree', DecisionTreeClassifier(random_state=42)),
        ('lg', LogisticRegression(max_iter=900))
    ]
    stacking_classifier = StackingClassifier(
        estimators=base_learners, final_estimator=LogisticRegression())
    stacking_classifier.fit(X_train, y_train)
    return stacking_classifier
