#!/usr/bin/env python
# coding: utf-8

# In[55]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

n_samples, h, w = lfw_people.images.shape

X = lfw_people.data

n_features = X.shape[1]

y = lfw_people.target
target_names = lfw_people.target_names

n_classes = target_names.shape[0]

pca = RandomizedPCA(n_components=150, whiten=True,svd_solver='randomized', random_state=42)

svc = SVC(kernel='rbf', class_weight='balanced')

model = make_pipeline(pca,svc)
print(model)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

param_grid = {'svc__C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(model, param_grid)
clf = clf.fit(X_train,y_train)
clf.best_estimator_

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=target_names))

def plot_gallery(images, titles, h, w, n_row=4, n_col=6):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        pred_name,true_name = titles[i]
        title = 'predicted: %s\ntrue:      %s' % (pred_name, true_name)
        col = 'k' if pred_name == true_name else 'r'
        title_obj = plt.title(title, size=12)
        plt.setp(title_obj, color = col)
        plt.xticks(())
        plt.yticks(())

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return (pred_name,true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

df = pd.DataFrame({'y_test':y_test,'y_pred':y_pred})
df['result'] = np.where(df['y_test'] == df['y_pred'], 1, 0)

confusion_matrix = pd.crosstab(df['y_pred'], df['y_test'], rownames=['Predicted'], colnames=['Actual'])
print(confusion_matrix)

heatmap_data = pd.pivot_table(df, values='result', 
                     index=['y_pred'], 
                     columns='y_test')

sbn.heatmap(heatmap_data, cmap="YlGnBu")
