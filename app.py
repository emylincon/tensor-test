from flask import Flask, jsonify
from Predict import GroupLSTM
from Predict import GetARIMA
import pandas as pd
import json

app = Flask(__name__)
pt = 'new_data.csv'
df = pd.read_csv(pt)
dft = df.to_dict()
lstm_agent = GroupLSTM(dft)
arima_agent = GetARIMA(dft)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/get-data')
def get_data():
    result = {'arima': arima_agent.predict(), 'lstm': lstm_agent.predict(dft)}

    return json.dumps(str(result)), 200


if __name__ == '__main__':
    app.run()
