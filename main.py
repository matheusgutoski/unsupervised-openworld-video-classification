import numpy as np
import generator as gen
np.random.seed(2021)


if __name__ == '__main__':

	#set all params
	params = {}
	params['min_classes'] = 2
	params['max_classes'] = 10

	#get filenames
	filenames = gen.get_filenames()

	#get class list
	unique_classes = np.unique([x.split('/')[0] for x in filenames]).tolist()



	#phase 1									------------------------

	#select initial classes
	initial_classes, remaining_classes = gen.random_class_picker(unique_classes, params)
	print(initial_classes)


	#perform train/test split

	#train i3d

	#train ti3d

	#train evm

	#evaluate


	#end phase 1






	#phase 2									-----------------------
	new_classes_list = []
	go = True
	iteration = 0 
	while go:
		#select z new classes
		new_classes, remaining_classes = gen.random_class_picker(remaining_classes, params)
		new_classes_list.append(new_classes)
		print(new_classes)
	
		#perform unsupervised train/test split

		#create open-set #iteration

		#extract features ti3d

		#open set recognition on open-set #iteration

		#get rejected set

		#open set evaluation

		#end phase 2










		#phase 3								-----------------------
		
		#estimate number of clusters in the rejected set

		#cluster with hierarchical agglomerative ward clustering

		#assign labels

		#get rejected set labels 

		#evaluate clustering performance

		#end phase 3







		#phase 4								------------------------
	
		#evaluate known test set and unsupervised test set (Before incremental learning)

		#finetune ti3d (incremental mode)

		#increment evm 

		#evaluate known test set and unsupervised test set (After incremental learning)


		#increment known test set (may be changed later)



		#end


