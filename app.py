from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request

import pandas as pd

import meter

app = Flask(__name__)

@app.route('/')
def index():

	callback = meter.main(dataset)
	return render_template('index.html', callback=callback	)

@app.route('/json')
def json():

	callback = meter.main(dataset)
	return jsonify(callback)


@app.route('/charts', methods=['POST'])
def gen_chart():

	data = dict(request.form)
	
	callback = meter.data_charts(dataset, data['data'][0])
	return jsonify(callback)
	
if __name__ == '__main__':

	dataset = pd.read_csv('mai-set.csv', sep=';')
	app.run(debug=True, host='0.0.0.0', port=80)