import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

        
class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
        # print(self.pima.head()) to be used in case of issues
        self.X_test = None
        self.y_test = None
        

    def define_feature(self):
        feature_cols = ['pregnant','glucose', 'bp', 'bmi']
        X = self.pima[feature_cols]
        y = self.pima.label
        return X, y
    
    def logistic_regression(self):
        logreg = LogisticRegression()
        return self.train_and_predict(logreg)

    def sgd_classifier(self):
        sgd_clf = SGDClassifier(max_iter=1000,tol=1e-3,random_state=0)
        return self.train_and_predict(sgd_clf)

    def decision_tree_classifier(self):
        dt_clf = DecisionTreeClassifier(random_state = 0, max_depth = 4)
        return self.train_and_predict(dt_clf)

    def kneighbors_classifier(self):
        kn_clf = KNeighborsClassifier(n_neighbors = 8)
        return self.train_and_predict(kn_clf)

    def train_and_predict(self,model):
        # split X and y into training and testing sets
        X, y = self.define_feature()
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, random_state=0)
        model.fit(X_train, y_train)
        y_pred_class = model.predict(self.X_test)
        return y_pred_class


    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)


    def examine(self):
        dist = self.y_test.value_counts()
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()
    
    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)
    
if __name__ == "__main__":
    classifer = DiabetesClassifier()
    lr_result = classifer.logistic_regression()
    # print(f"Predicition={lr_result}") to be used in case of issues
    lr_score = classifer.calculate_accuracy(lr_result)
    # print(f"score={lr_score}")
    lr_con_matrix = classifer.confusion_matrix(lr_result)
    # print(f"confusion_matrix=${lr_con_matrix}") to be used in case of issues
    
    sgd_result = classifer.sgd_classifier()
    # print(f"Predicition={sgd_result}") to be used in case of issues
    sgd_score = classifer.calculate_accuracy(sgd_result)
    # print(f"score={sgd_score}")
    sgd_con_matrix = classifer.confusion_matrix(sgd_result)
    # print(f"confusion_matrix=${sgd_con_matrix}") to be used in case of issues

    dt_result = classifer.decision_tree_classifier()
    # print(f"Predicition={dt_result}") to be used in case of issues
    dt_score = classifer.calculate_accuracy(dt_result)
    # print(f"score={dt_score}") to be used in case of issues
    dt_con_matrix = classifer.confusion_matrix(dt_result)
    # print(f"confusion_matrix=${dt_con_matrix}") to be used in case of issues
    
    kn_result = classifer.kneighbors_classifier()
    # print(f"Predicition={kn_result}") to be used in case of issues
    kn_score = classifer.calculate_accuracy(kn_result)
    # print(f"score={kn_score}") to be used in case of issues
    kn_con_matrix = classifer.confusion_matrix(kn_result)
    # print(f"confusion_matrix=${kn_con_matrix}") to be used in case of issues

    output = pd.DataFrame(np.array([
        ['logistic_regression' , lr_score , lr_con_matrix , 'My Logistic Regression solution and the labels are pregnant, glucose, bp, bmi' ],
        ['SGDClssifier', sgd_score, sgd_con_matrix, 'solution with SGDC and the labels are pregnant, glucose, bp, bmi'],
        ['DecisionTreeClassier',dt_score, dt_con_matrix, 'solution with DTC and the labels are pregnant, glucose, bp, bmi'],
        ['KNeighborClassofier', kn_score, kn_con_matrix, 'solution with KNC and the labels are pregnant, glucose, bp, bmi']
        ]), columns = ['Experiment','Accuracy','Confusion Matrix','Comment'])

    print(output.to_markdown())
