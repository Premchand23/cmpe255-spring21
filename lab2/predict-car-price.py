import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        print(f'${len(self.df)} lines loaded')

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    def validate(self):
        np.random.seed(2)
        n = len(self.df)
        n_val = int(0.2 * n)
        n_test = int(0.2 * n)
        n_train = n - (n_val + n_test)
        idx = np.arange(n)
        np.random.shuffle(idx)
        df_shuffled = self.df.iloc[idx]
        df_train = df_shuffled.iloc[:n_train].copy()
        df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
        df_test = df_shuffled.iloc[n_train+n_val:].copy()
        y_train_orig = df_train.msrp.values
        y_val_orig = df_val.msrp.values
        y_test_orig = df_test.msrp.values
        y_train = np.log1p(df_train.msrp.values)
        y_val = np.log1p(df_val.msrp.values)
        y_test = np.log1p(df_test.msrp.values)

        del df_train['msrp']
        del df_val['msrp']
        del df_test['msrp']
        

        def linear_regression(X, y):
            ones = np.ones(X.shape[0])
            X = np.column_stack([ones, X])

            XTX = X.T.dot(X)
            XTX_inv = np.linalg.inv(XTX)
            w = XTX_inv.dot(X.T).dot(y)
        
            return w[0], w[1:]

        base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']


        def prepare_X(df):
                df_num = df[base]
                df_num = df_num.fillna(0)
                X = df_num.values
                return X

        X_train = prepare_X(df_train)
        w_0, w = linear_regression(X_train, y_train)
        y_pred = w_0 + X_train.dot(w)
        plt.figure(figsize=(6, 4))

        sns.distplot(y_train, label='target', kde=False,
                hist_kws=dict(color='#222222', alpha=0.6))
        sns.distplot(y_pred, label='prediction', kde=False,
                hist_kws=dict(color='#aaaaaa', alpha=0.8))
        plt.legend()
        plt.ylabel('Frequency')
        plt.xlabel('Log(Price + 1)')
        plt.title('Predictions vs actual distribution')
        plt.show()

        def rmse(y, y_pred):
                error = y_pred - y
                mse = (error ** 2).mean()
                return np.sqrt(mse)

        rmse(y_train, y_pred)

        X_val = prepare_X(df_val)
        y_pred = w_0 + X_val.dot(w)

        rmse(y_val, y_pred)

        def prepare_X_1(df):
                df = df.copy()
                features = base.copy()

                df['age'] = 2017 - df.year
                features.append('age')

                df_num = df[features]
                df_num = df_num.fillna(0)
                X = df_num.values
                return X

        X_train = prepare_X_1(df_train)
        w_0, w = linear_regression(X_train, y_train)

        y_pred = w_0 + X_train.dot(w)
        print('train', rmse(y_train, y_pred))

        X_val = prepare_X_1(df_val)
        y_pred = w_0 + X_val.dot(w)
        print('validation', rmse(y_val, y_pred))

        plt.figure(figsize=(6, 4))


        sns.distplot(y_val, label='target', kde=False,
                    hist_kws=dict(color='#222222', alpha=0.6))
        sns.distplot(y_pred, label='prediction', kde=False,
                    hist_kws=dict(color='#aaaaaa', alpha=0.8))

        plt.legend()

        plt.ylabel('Frequency')
        plt.xlabel('Log(Price + 1)')
        plt.title('Predictions vs actual distribution')

        plt.show()

        self.df['make'].value_counts().head(5)

        def prepare_X_2(df):
                df = df.copy()
                features = base.copy()

                df['age'] = 2017 - df.year
                features.append('age')

                for v in [2, 3, 4]:
                    feature = 'num_doors_%s' % v
                    df[feature] = (df['number_of_doors'] == v).astype(int)
                    features.append(feature)

                for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
                    feature = 'is_make_%s' % v
                    df[feature] = (df['make'] == v).astype(int)
                    features.append(feature)

                df_num = df[features]
                df_num = df_num.fillna(0)
                X = df_num.values
                return X

        X_train = prepare_X_2(df_train)
        w_0, w = linear_regression(X_train, y_train)

        y_pred = w_0 + X_train.dot(w)
        print('train:', rmse(y_train, y_pred))

        X_val = prepare_X_2(df_val)
        y_pred = w_0 + X_val.dot(w)
        print('validation:', rmse(y_val, y_pred))

        self.df['engine_fuel_type'].value_counts()

        def prepare_X_3(df):
                df = df.copy()
                features = base.copy()

                df['age'] = 2017 - df.year
                features.append('age')
                
                for v in [2, 3, 4]:
                    feature = 'num_doors_%s' % v
                    df[feature] = (df['number_of_doors'] == v).astype(int)
                    features.append(feature)

                for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
                    feature = 'is_make_%s' % v
                    df[feature] = (df['make'] == v).astype(int)
                    features.append(feature)

                for v in ['regular_unleaded', 'premium_unleaded_(required)', 
                        'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)']:
                    feature = 'is_type_%s' % v
                    df[feature] = (df['engine_fuel_type'] == v).astype(int)
                    features.append(feature)
                    
                df_num = df[features]
                df_num = df_num.fillna(0)
                X = df_num.values
                return X
        
        X_train = prepare_X_3(df_train)
        w_0, w = linear_regression(X_train, y_train)

        y_pred = w_0 + X_train.dot(w)
        print('train:', rmse(y_train, y_pred))

        X_val = prepare_X_3(df_val)
        y_pred = w_0 + X_val.dot(w)
        print('validation:', rmse(y_val, y_pred))


        self.df['driven_wheels'].value_counts()

        self.df['market_category'].value_counts().head(5)

        self.df['vehicle_size'].value_counts().head(5)

        self.df['vehicle_style'].value_counts().head(5)

        def prepare_X_4(df):
                df = df.copy()
                features = base.copy()

                df['age'] = 2017 - df.year
                features.append('age')
                
                for v in [2, 3, 4]:
                    feature = 'num_doors_%s' % v
                    df[feature] = (df['number_of_doors'] == v).astype(int)
                    features.append(feature)

                for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
                    feature = 'is_make_%s' % v
                    df[feature] = (df['make'] == v).astype(int)
                    features.append(feature)

                for v in ['regular_unleaded', 'premium_unleaded_(required)', 
                        'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)']:
                    feature = 'is_type_%s' % v
                    df[feature] = (df['engine_fuel_type'] == v).astype(int)
                    features.append(feature)

                for v in ['automatic', 'manual', 'automated_manual']:
                    feature = 'is_transmission_%s' % v
                    df[feature] = (df['transmission_type'] == v).astype(int)
                    features.append(feature)

                for v in ['front_wheel_drive', 'rear_wheel_drive', 'all_wheel_drive', 'four_wheel_drive']:
                    feature = 'is_driven_wheens_%s' % v
                    df[feature] = (df['driven_wheels'] == v).astype(int)
                    features.append(feature)

                for v in ['crossover', 'flex_fuel', 'luxury', 'luxury,performance', 'hatchback']:
                    feature = 'is_mc_%s' % v
                    df[feature] = (df['market_category'] == v).astype(int)
                    features.append(feature)

                for v in ['compact', 'midsize', 'large']:
                    feature = 'is_size_%s' % v
                    df[feature] = (df['vehicle_size'] == v).astype(int)
                    features.append(feature)

                for v in ['sedan', '4dr_suv', 'coupe', 'convertible', '4dr_hatchback']:
                    feature = 'is_style_%s' % v
                    df[feature] = (df['vehicle_style'] == v).astype(int)
                    features.append(feature)

                df_num = df[features]
                df_num = df_num.fillna(0)
                X = df_num.values
                return X
        
        X_train = prepare_X_4(df_train)
        w_0, w = linear_regression(X_train, y_train)

        y_pred = w_0 + X_train.dot(w)
        print('train:', rmse(y_train, y_pred))

        X_val = prepare_X_4(df_val)
        y_pred = w_0 + X_val.dot(w)
        print('validation:', rmse(y_val, y_pred))

        w_0

        def linear_regression_reg(X, y, r=0.0):
                ones = np.ones(X.shape[0])
                X = np.column_stack([ones, X])

                XTX = X.T.dot(X)
                reg = r * np.eye(XTX.shape[0])
                XTX = XTX + reg

                XTX_inv = np.linalg.inv(XTX)
                w = XTX_inv.dot(X.T).dot(y)
                
                return w[0], w[1:]

        X_train = prepare_X_4(df_train)

        for r in [0, 0.001, 0.01, 0.1, 1, 10]:
            w_0, w = linear_regression_reg(X_train, y_train, r=r)
            print('%5s, %.2f, %.2f, %.2f' % (r, w_0, w[13], w[21]))
        

        X_train = prepare_X_4(df_train)
        w_0, w = linear_regression_reg(X_train, y_train, r=0)

        y_pred = w_0 + X_train.dot(w)
        print('train', rmse(y_train, y_pred))

        X_val = prepare_X_4(df_val)
        y_pred = w_0 + X_val.dot(w)
        print('val', rmse(y_val, y_pred))


        X_train = prepare_X_4(df_train)
        w_0, w = linear_regression_reg(X_train, y_train, r=0.01)

        y_pred = w_0 + X_train.dot(w)
        print('train', rmse(y_train, y_pred))

        X_val = prepare_X_4(df_val)
        y_pred = w_0 + X_val.dot(w)
        print('val', rmse(y_val, y_pred))

        X_train = prepare_X_4(df_train)
        X_val = prepare_X_4(df_val)

        for r in [0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
            w_0, w = linear_regression_reg(X_train, y_train, r=r)
            y_pred = w_0 + X_val.dot(w)
            print('%6s' %r, rmse(y_val, y_pred))
        
        X_train = prepare_X_4(df_train)
        w_0, w = linear_regression_reg(X_train, y_train, r=0.01)

        X_val = prepare_X_4(df_val)
        y_pred = w_0 + X_val.dot(w)
        print('validation:', rmse(y_val, y_pred))

        X_test = prepare_X_4(df_test)
        y_pred = w_0 + X_test.dot(w)
        print('test:', rmse(y_test, y_pred))

        q =[]
        for i in range(5):
            ad = df_test.iloc[i].to_dict()

            X_test = prepare_X_4(pd.DataFrame([ad]))[0]
            y_pred = w_0 + X_test.dot(w)
            suggestion = np.expm1(y_pred)
            ad['msrp.pred'] = suggestion
            ad['msrp'] = y_test_orig[i]
            q.append(ad)
        print(q)
        

predict_car_price = CarPrice()
predict_car_price.trim()
predict_car_price.validate()

    



    