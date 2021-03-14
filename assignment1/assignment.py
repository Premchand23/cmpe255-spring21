import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures


class Housing:
    def __init__(self):
        self.df = pd.read_csv('housing.csv', delim_whitespace=True,header = None)
        print(self.df)

    def validate(self):
        x = self.df[5]
        x = pd.DataFrame({'RM':x})
        y = self.df[13]
        y = pd.DataFrame({'MEDV':y})


        x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 1)

        mod = LinearRegression()
        mod.fit(x_train,y_train)
        y_pred = mod.predict(x_test)

        print("Linear Regression")

        print("R2 Score:",r2_score(y_pred,y_test))

        print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

        plt.scatter(x_test, y_test,color='g')
        plt.plot(x_test, y_pred,color='k')
        plt.show()

        po_feat = PolynomialFeatures(degree=2)
        x_train_po = po_feat.fit_transform(x_train)
        po_model = LinearRegression()
        po_model.fit(x_train_po, y_train)
        y_pred1 = po_model.predict(po_feat.fit_transform(x_test))

        print("Polyomial Regression")
        
        print("R2 Score:", r2_score(y_test, y_pred1))
        
        print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred1)))

        sns.scatterplot(y_test['MEDV'].values, y_pred1.reshape(-1), alpha=0.4)
        sns.regplot(y_test['MEDV'].values, y_pred1.reshape(-1), scatter_kws={'s': 20, 'alpha':0.3}, line_kws={'color':'green', 'linewidth': 2}, order=2)
        plt.show()

        po_feat = PolynomialFeatures(degree=20)
        x_train_po = po_feat.fit_transform(x_train)
        po_model = LinearRegression()
        po_model.fit(x_train_po, y_train)
        y_pred1 = po_model.predict(po_feat.fit_transform(x_test))

        sns.scatterplot(y_test['MEDV'].values, y_pred1.reshape(-1), alpha=0.4)
        sns.regplot(y_test['MEDV'].values, y_pred1.reshape(-1), scatter_kws={'s': 20, 'alpha':0.3}, line_kws={'color':'green', 'linewidth': 2}, order=2)
        plt.show()

        X = pd.DataFrame(np.c_[self.df[1], self.df[5],self.df[11]], columns=['LSTAT','RM','B'])
        Y = self.df[13]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=9)

        mmod = LinearRegression()

        mmod.fit(X_train, y_train)

        y_pred2 = mmod.predict(X_test)

        print("Multiple Regression")

        print("R2 Score:", r2_score(y_test, y_pred2))

        print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred2)))

        adj_r2 = 1 - (1-mmod.score(X, y))*(len(y)-1)/(len(y)-X.shape[1]-1)
        print(adj_r2)

housing_price  = Housing()
housing_price.validate()