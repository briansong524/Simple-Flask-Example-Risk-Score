import os
import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from utils import is_risky, categorical_dictionary

parser = argparse.ArgumentParser()

parser.add_argument(
	'--data_dir', type=str, default = '',
	help = 'Directory of data to train with')

parser.add_argument(
	'--model_dir', type=str, default = 'model',
	help = 'Directory to save files')


def main(FLAGS):

	# initialize
	print('begin model training process')
	if FLAGS.data_dir == '':
		work_dir = os.path.dirname(os.path.realpath(__file__))
	else:
		work_dir = FLAGS.data_dir
	
	df_app = pd.read_csv(work_dir + '/application_record.csv')
	df_credit = pd.read_csv(work_dir + '/credit_record.csv')
	print('pulled data')

	# make target variable
	print('cleaning and processing data')
	df_credit = df_credit[df_credit.STATUS != 'X']
	df_credit['is_risky'] = df_credit.STATUS.map(is_risky)
	piv = df_credit.pivot_table(index = 'ID', values = 'is_risky', aggfunc = lambda x: sum(x) / float(len(x)))
	piv.columns = ['binary_status']
	piv.reset_index(inplace = True)
	id_class_dict = dict(zip(piv.ID, piv.binary_status))
	df_app['is_risky'] = df_app['ID'].map(id_class_dict)
	
	# make training set
	df = df_app[~df_app['is_risky'].isna()].copy()
	x_cat_cols = ['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY'] # categorical columns
	x_num_cols = ['CNT_CHILDREN','AMT_INCOME_TOTAL'] # numeric columns

	cat_dict = categorical_dictionary()
	X = df[x_num_cols].copy()
	y = df['is_risky'].copy()

	for col_ in x_cat_cols:
		cat_dict.add_col(df[col_].values, col_)
		X[col_] = cat_dict.cat_to_ind(df[col_], col_)
	print('cleaned/processed data')

	# train model
	print('training model')
	model = xgb.XGBRegressor()
	model.fit(X, y)
	print('trained model')

	# output hard files 
	save_loc = work_dir + '/model'
	if not os.path.exists(save_loc):
		os.makedirs(save_loc)

	with open(save_loc + '/model.pkl','wb') as model_out:
		pickle.dump(model, model_out)

	with open(save_loc + '/cat_dict.pkl','wb') as dict_out:
		pickle.dump(cat_dict, dict_out)

	print('finished exporting files')
	print('done')

if __name__ =='__main__':
	FLAGS, unparsed = parser.parse_known_args()
	main(FLAGS)