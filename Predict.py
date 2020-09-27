import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model
import numpy as np
import concurrent.futures
import math
from threading import Thread
from statsmodels.tsa.arima_model import ARIMAResults
from datetime import datetime as dt
import pytz
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',FutureWarning)


# monkey patch around bug in ARIMA class
def __getnewargs__(self):
    return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))


ARIMA.__getnewargs__ = __getnewargs__


class Model:
    def __init__(self, data):
        self.data = data
        self.london = pytz.timezone('Europe/London')
        self.max_rmse = 10
        self.describe = {}
        self.model_path = 'models'
        self.units = ('temp', 'hum', 'heat')

    @property
    def df(self):
        return pd.DataFrame(self.data)

    @df.setter
    def df(self, data):
        self.data = data

    def status(self, unit):
        if self.describe[unit]['rmse'] > self.max_rmse:
            return True
        return False


class GetARIMA(Model):
    """:key
    data is a dictionary that can be converted into dataframe
    """
    def __init__(self, data):
        super().__init__(data)
        self.describe = {'hum': {'rmse': 1.118, 'next': 400, 'date': '24-09-2020 19:09:36', 'arrow': 'up'},
                         'temp': {'rmse': 0.0, 'next': 400, 'date': '24-09-2020 19:09:36', 'arrow': 'down'},
                         'heat': {'rmse': 0.0251, 'next': 400, 'date': '24-09-2020 19:09:36', 'arrow': 'up'}}
        self.arima_vars = None
        self.model_path = 'models/arima'
        self.load_models()

    def load_models(self):
        for i in self.describe:
            self.describe[i]['model'] = ARIMAResults.load(f'{self.model_path}/model_{i}.pkl')

    def get_stat(self):
        return {i: {j: self.describe[i][j] for j in ['rmse', 'date', 'arrow']} for i in self.describe}

    def data_prep(self):
        self.df.drop('id', axis='columns', inplace=True)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.df.set_index('datetime', inplace=True)

        arima_vars = {'data': self.df[-1000:]}
        arima_vars['training_data_len'] = math.ceil(len(arima_vars['data']) * .9)
        arima_vars['temp'] = {'values': arima_vars['data']['temperature'].values}
        arima_vars['hum'] = {'values': arima_vars['data']['humidity'].values}
        arima_vars['heat'] = {'values': arima_vars['data']['heat_index'].values}
        for i in self.units:
            arima_vars[i]['train'] = arima_vars[i]['values'][:arima_vars['training_data_len']]
            arima_vars[i]['test'] = arima_vars[i]['values'][arima_vars['training_data_len']:]

        return arima_vars

    @staticmethod
    def percentage(new_value, old_value):
        return round(((new_value-old_value)/old_value) * 100, 2)

    def train_model(self, units=None):
        if not units:
            units = self.units
        self.arima_vars = self.data_prep()
        for i in units:
            model = self.best_model(self.arima_vars[i]['train'], self.arima_vars[i]['test'])
            if model['model']:
                arrow = 'down'
                if model['rmse'] > self.describe[i]['rmse']:
                    arrow = 'up'
                self.describe[i] = model
                self.describe[i]['arrow'] = arrow

    def data_forecast(self, unit):
        full = {'hum': 'humidity', 'temp': 'temperature', 'heat': 'heat_index'}
        self.describe[unit]['next'] += 1
        pred = self.describe[unit]['model'].forecast(steps=self.describe[unit]['next'])[0]
        test = self.df[full[unit]].values[-self.describe[unit]['next']:]
        error = np.sqrt(mean_squared_error(test, pred))
        arrow = 'down'
        if error > self.describe[unit]['rmse']:
            arrow = 'up'
        self.describe[unit]['arrow'] = arrow
        self.describe[unit]['rmse'] = error
        return pred[-1]

    def predict(self):
        result = {}

        for i in self.describe.keys():
            self.describe[i]['next'] += 1
            result[i] = self.describe[i]['model'].forecast(steps=self.describe[i]['next'])[0][-1]

        units = [i for i in self.units if self.status(i)]
        if len(units) > 0:
            t1 = Thread(target=self.train_model, args=(units,))
            t1.start()
        return result

    def get_model(self, train_data, my_order, test_data):
        try:
            model_arima = ARIMA(train_data, order=my_order)
            model_arima_fit = model_arima.fit()
            predict_data = model_arima_fit.data_forecast(steps=10)[0]
            error = np.sqrt(mean_squared_error(test_data[:9], predict_data[1:]))
            return {'model': model_arima_fit, 'rmse': error, 'next': len(test_data),
                    'date':  "{:%d-%m-%Y %H:%M:%S}".format(dt.now().astimezone(self.london))}
        except Exception as e:
            return {'model': None, 'rmse': 10 ** 10, 'next': len(test_data),
                    'date':  "{:%d-%m-%Y %H:%M:%S}".format(dt.now().astimezone(self.london))}

    def best_model(self, train_data, test_data):
        orders = [(2, 1, 2), (2, 0, 2), (2, 1, 1), (1, 1, 1), (0, 2, 1), (0, 1, 0)]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [executor.submit(self.get_model, train_data, i, test_data) for i in orders]

        # runs process and returns result as they complete
        completed = []
        for f in concurrent.futures.as_completed(results):
            ans = f.result()
            print(ans)
            completed.append(ans)
        return sorted(completed, key=lambda k: k['rmse'])[0]


class GetLSTM(Model):
    def __init__(self, data, kind):
        super().__init__(data)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.kind = kind
        self.kind_dict = {'hum': 'humidity', 'temp': 'temperature', 'heat': 'heat_index'}
        self.describe = {'rmse': 1.118, 'date': '24-09-2020 19:09:36', 'model': None, 'arrow': 'down'}
        self.step = 60
        self.lstm = None
        self.last_trained = None
        self.train_time_increment = 60*60*3
        self.model_path = 'models/lstm'
        self.load_model()

    def pre_processing(self, size):
        # self.df.drop('id', axis='columns', inplace=True)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.df.set_index('datetime', inplace=True)

        lstm = {'data': self.df[-size:]}
        lstm['training_data_len'] = math.ceil(len(lstm['data']) * .9)
        lstm['values'] = lstm['data'].filter([self.kind_dict[self.kind]]).values

        return lstm

    def load_model(self):
        rmse = {'hum': {'rmse': 1.3491, 'accuracy': 59.82, 'loss': 40.18},
                'temp': {'rmse': 0.01, 'accuracy': 99.1, 'loss': 0.9},
                'heat': {'rmse': 0.0301, 'accuracy': 71.53, 'loss': 28.47}}
        self.describe['model'] = load_model(fr"{self.model_path}/{self.kind}.h5")
        self.describe.update(rmse[self.kind])

    def data_prep(self):
        lstm = self.pre_processing(2040)

        lstm['scaled_data'] = self.scaler.fit_transform(lstm['values'])
        lstm['train_data'] = lstm['scaled_data'][0:lstm['training_data_len'], :]
        # Split the data into x_train and y_train data sets
        lstm['x_train'] = []
        lstm['y_train'] = []
        for j in range(self.step, len(lstm['train_data'])):
            lstm['x_train'].append(lstm['train_data'][j - self.step:j, 0])
            lstm['y_train'].append(lstm['train_data'][j, 0])

        # Convert x_train and y_train to numpy arrays
        lstm['x_train'], lstm['y_train'] = np.array(lstm['x_train']), np.array(lstm['y_train'])

        # Reshape the data into the shape accepted by the LSTM
        # LSTM accepts only 3 dimentionaly shape
        lstm['x_train'] = np.reshape(lstm['x_train'], (lstm['x_train'].shape[0], lstm['x_train'].shape[1], 1))
        lstm['model'] = self.build_model(input_shape=(lstm['x_train'].shape[1], 1))

        # Test data set
        lstm['test_data'] = lstm['scaled_data'][lstm['training_data_len'] - self.step:, :]
        # Create the x_test and y_test data sets
        lstm['x_test'] = []
        lstm['y_test'] = lstm['values'][lstm['training_data_len']:,:]
        for k in range(self.step, len(lstm['test_data'])):
            lstm['x_test'].append(lstm['test_data'][k - self.step:k, 0])
        # Convert x_test to a numpy array
        lstm['x_test'] = np.array(lstm['x_test'])
        # Reshape the data into the shape accepted by the LSTM
        lstm['x_test'] = np.reshape(lstm['x_test'], (lstm['x_test'].shape[0], lstm['x_test'].shape[1], 1))

        return lstm

    @staticmethod
    def build_model(input_shape):
        # Build the LSTM network model
        model = Sequential()
        # units=50 => this is the number of neurons
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        return model

    def train_model(self):
        lstm = self.data_prep()
        # Compile the model
        self.lstm = lstm

        lstm['model'].compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        self.describe = self.train()

    @staticmethod
    def percentage(new_value, old_value):
        return round(((new_value-old_value)/old_value) * 100, 2)

    def predict(self):
        lstm = self.pre_processing(60)

        lstm['scaled'] = self.scaler.fit_transform(lstm['values'])
        # Create an empty list
        X_test = [lstm['scaled']]
        # Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
        # Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        # Get the predicted scaled price
        pred = self.describe['model'].predict(X_test)
        # undo the scaling
        pred = self.scaler.inverse_transform(pred)
        result = pred[0][0]

        # if (dt.now().astimezone(self.london) - self.last_trained).seconds > self.train_time_increment:
        #     self.train_model()
        return result

    def train(self):
        # Train the model
        self.lstm['history'] = self.lstm['model'].fit(self.lstm['x_train'], self.lstm['y_train'], batch_size=500, epochs=1)

        # Getting the models predicted price values
        predictions = self.lstm['model'].predict(self.lstm['x_test'])
        predictions = self.scaler.inverse_transform(predictions)  # Undo scaling
        self.lstm['predictions'] = predictions
        # Calculate/Get the value of RMSE
        rmse = np.sqrt(np.mean(((predictions - self.lstm['y_test']) ** 2)))

        score = round(100 - self.lstm['history'].history['loss'][0]*100, 2)
        now = dt.now().astimezone(self.london)
        self.last_trained = now
        arrow = 'down'
        if rmse > self.describe['rmse']:
            arrow = 'up'
        return {'model': self.lstm['model'], 'rmse': rmse, 'accuracy': score, 'loss': 100-score,
                'date': "{:%d-%m-%Y %H:%M:%S}".format(now), 'arrow': arrow}

    def save_model(self, file_name):
        self.describe['model'].save(file_name)


class GroupLSTM:
    def __init__(self, data):
        self.data = data
        self.kind_dict = {'hum': 'humidity', 'temp': 'temperature', 'heat': 'heat_index'}
        self.models = {i: GetLSTM(data, i) for i in self.kind_dict}

    def predict(self, data):
        result = {}
        for i in self.models:
            self.models[i].data = data
            result[i] = self.models[i].predict()
        return result

    def train_models(self):
        for i in self.models:
            self.models[i].train_model()

    def describe(self):
        return {i: {'rmse': self.models[i].describe['rmse'], 'date': self.models[i].describe['date'],
                    'accuracy': self.models[i].describe['accuracy'], 'arrow': self.models[i].describe['arrow'],
                    'loss': self.models[i].describe['loss']} for i in self.models}

    def save_models(self):
        for i in self.models:
            self.models[i].save_model(f'{i}.h5')



# pt = 'new_data.csv'
# df = pd.read_csv(pt)
# dft = df.to_dict()
# ari = GetARIMA(data=dft)
# print(ari.predict())
#
# obj = GroupLSTM(dft)
# # obj.train_models()
# # obj.save_models()
# print(obj.predict(dft))
