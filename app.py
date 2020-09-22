import os
import sys
if sys.version_info[0] == 2:
	import thread
else:
	import _thread as thread

import json
import argparse
import pickle

import pandas as pd
import numpy as np

from flask import Flask, request


app = Flask(__name__)

parser = argparse.ArgumentParser()

parser.add_argument(
	'--model_dir', type=str, default = 'model',
	help = 'Directory of data to train with and save files')

parser.add_argument(
	'--risk_threshold', type=float, default = 0.5,
	help = 'threshold from 0-1 where above means risky')


FLAGS, unparsed = parser.parse_known_args()

if FLAGS.model_dir == '':
	dir_ = os.path.dirname(os.path.realpath(__file__))
else:
	dir_ = FLAGS.model_dir

global model, cat_dict
with open(dir_ + '/model.pkl', 'rb') as model_in:
	model = pickle.load(model_in)

with open(dir_ + '/cat_dict.pkl', 'rb') as file_in:
	cat_dict = pickle.load(file_in)

@app.route('/heart-beat', methods=['GET'])
def run_heart_beat_check():  
	status_object = {'status':'alive'}
	response = app.response_class(
		response=json.dumps(status_object),
		status=200,
		mimetype='application/json'
	)
	return response

@app.route('/api')
def api():
	global model, cat_dict

	gender = str(request.args.get('gender',None))
	own_car = str(request.args.get('own_car',None))
	own_home = str(request.args.get('own_home',None))
	n_children = int(request.args.get('n_children',None))
	income = int(request.args.get('income',None))

	gender = cat_dict.cat_to_ind([gender], 'CODE_GENDER')[0]
	own_car = cat_dict.cat_to_ind([own_car], 'FLAG_OWN_CAR')[0]
	own_home = cat_dict.cat_to_ind([own_home], 'FLAG_OWN_REALTY')[0]

	# convert string to num

	input_data = pd.DataFrame({'CODE_GENDER':gender,
								'FLAG_OWN_CAR':own_car,
								'FLAG_OWN_REALTY':own_home,
								'CNT_CHILDREN':n_children,
								'AMT_INCOME_TOTAL':income}, index = [0])
	input_data = input_data[['CNT_CHILDREN','AMT_INCOME_TOTAL','CODE_GENDER',
							'FLAG_OWN_CAR','FLAG_OWN_REALTY']] #reorder columns
	print(input_data)
	pred = model.predict(input_data)[0] 
	pred_class = 'risky' if pred > FLAGS.risk_threshold else 'not_risky'
	output = {
				'decision':pred_class,
				'risk_score':round(float(pred),4)
			 }
	return output


if __name__ == '__main__':
	app.run(debug=False, port=8000, host='0.0.0.0')