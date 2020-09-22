
def is_risky(x):
	# making binary 'is_risky' flag for target variable
	# 0 for good status, 1 for bad status
	if x in ['C']:
		return 0
	else:
		return 1

class categorical_dictionary:
	'class containing all categorical variables indexed and converted'
	
	def __init__(self):
		self.cat_dict = {}
		self.rev_cat_dict = {}
	
	def add_col(self, vals, col_name, verbose = True):
		cat_vals = set(vals) # get classes
		temp_dict = dict(zip(cat_vals, range(1, len(cat_vals)+1)))
		temp_dict[col_name + '_UNK'] = 0 # adding an index for previously non-existant class 
		self.cat_dict[col_name] = temp_dict
		rev_temp_dict = {j:i for i,j in temp_dict.items()}
		self.rev_cat_dict[col_name] = rev_temp_dict
		if verbose:
			print('Added ' + col_name)
			
	def cat_to_ind(self, vals, col_name):
		def failsafe_mapper(val, col_name):
			'make mapping robust by handling previously unseen classes'
			try:
				mapped_val = self.cat_dict[col_name][val]
			except:
				print('Unknown value: "' + str(val) + '", appending as index 0 (general unknown class index)')
				mapped_val = 0
			return mapped_val
		
		mapped_list = list(map(lambda x: failsafe_mapper(x,col_name), vals))
		return mapped_list
	
	def ind_to_cat(self, vals, col_name):
		return list(map(lambda x: self.rev_cat_dict[col_name][x], vals))

