import numpy as np
import generator as gen
import finetune_i3d 
import utils
import os
import argparse
from keras.models import load_model
from keras import backend as K 
import evm_classification as evm
import classification_metrics as metrics
import k_estimator
import hierarchical
import evaluation
if __name__ == '__main__':


	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help='Use which model?', type=str, choices=['kinetics','ucf101'], default = 'ucf101')
	parser.add_argument('--list_id', help='Video list id?', type=int, default = 0)
	parser.add_argument('--n_train_classes', help='How many classes at training time?', type=int, default = 2)
	parser.add_argument('--n_test_classes', help='How many classes at test time?', type=int, default = 3)
	parser.add_argument('--n_folds', help='How many folds?', type=int, default = 30)
	parser.add_argument('--train_ratio', help='Fraction of the training set to use for training the cnn. The rest is used for validation', type=float, default = 0.7)
	parser.add_argument('--num_frames', help='Number of frames in the temporal dimension', type=int, default = 64)
	parser.add_argument('--num_frames_test', help='Number of frames in the temporal dimension during test time', type=int, default = 250)
	parser.add_argument('--frame_height', help='Frame height', type=int, default = 224)
	parser.add_argument('--frame_width', help='Frame width', type=int, default = 224)
	parser.add_argument('--max_epochs', help='maximum number of epochs', type=int, default = 10)
	parser.add_argument('--triplet_epochs', help='triplet net max number of epochs', type=int, default = 50)
	parser.add_argument('--triplet_lr', help='triplet learning rate with sgd', type=int, default = 0.001)
	parser.add_argument('--margin', help='triplet net margin parameter', type=float, default = 0.2)
	parser.add_argument('--batch_size', help='batch size', type=int, default = 6)
	parser.add_argument('--triplet_batch_size', help='batch size of triplet net', type=int, default = 128)
	parser.add_argument('--tail_size', help='weibull tail size for evm', type=int, default = 10)
	parser.add_argument('--cover_threshold', help='evm cover threshold', type=float, default = 0.1)
	parser.add_argument('--classification_threshold', help='probability threshold for accepting points in evm', type=float, default = 0.005)
	parser.add_argument('--output_path', help='Where to output results', type=str, default = 'prototype_results/')
	parser.add_argument('--seed', help='Random seed. 123450 sets seed to random', type=int, default = 5)
	parser.add_argument('--gpu', help='gpu id', type=int)
	parser.add_argument('--expid', help='exp id', type=str)

	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)


	#set all params
	params = {}
	params['min_classes'] = 2
	params['max_classes'] = 10
	params['seed'] = args.seed
	params['init_seed'] = params['seed']
	params['model_type'] = 'cnn'

	np.random.seed(params['seed'])

	params['model'] = args.model
	params['list_id'] = args.list_id
	params['n_train_classes'] = args.n_train_classes
	params['n_test_classes'] = args.n_test_classes
	params['n_folds'] = args.n_folds
	params['output_path'] = args.output_path
	params['fold'] = 0

	#ignore errors - ignores classification errors until phase 4. Applies to EVM rejection and clustering
	params['ignore_errors'] = True

	#parameters for the cnn
	params['train_ratio'] = args.train_ratio
	params['num_frames'] = args.num_frames
	params['num_frames_test'] = args.num_frames_test
	params['frame_height'] = args.frame_height
	params['frame_width'] = args.frame_width
	params['max_epochs'] = args.max_epochs
	params['batch_size'] = args.batch_size



	#parameters for the triplet net
	params['triplet_epochs'] = args.triplet_epochs
	params['margin'] = args.margin
	params['triplet_lr'] = args.triplet_lr
	params['triplet_batch_size'] = args.triplet_batch_size

	#parameters for the evm
	params['tail_size'] = args.tail_size
	params['cover_threshold'] = args.cover_threshold
	#params['classification_threshold'] = args.classification_threshold
	#multi_classification_threshold = [0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.999, 0.9999]
	multi_classification_threshold = [0.001]

	params['iteration'] = 0 


	#get filenames
	filenames = gen.get_filenames()	
	all_categories = gen.get_all_categories(filenames)
	dict_map = utils.map_labels(all_categories)

	#get class list
	unique_classes = np.unique([x.split('/')[0] for x in filenames]).tolist()
	
	PATH_TO_FRAMES = '/home/users/datasets/UCF-101_opticalflow/'


	train_i3d_features = []
	train_i3d_labels = []

	test_i3d_features = []
	test_i3d_labels = []




















	#PHASE 1									------------------------


	#perform train/test split

	train, train_labels, test, test_labels = utils.train_test_split_groups(filenames, unique_classes, params)
	int_train_labels = utils.convert_labels_to_int(train_labels, dict_map)
	int_test_labels = utils.convert_labels_to_int(test_labels, dict_map)



	#select initial classes

	perm = np.random.permutation(np.array(unique_classes).shape[0])
	class_shuffle = np.array(unique_classes)[perm]
	print(unique_classes, class_shuffle)
	

	#pick n classes between minclasses and maxclasses
	initial_n_classes = np.random.randint(params['min_classes'],params['max_classes'])
	initial_classes = class_shuffle[0:initial_n_classes]
	current_classes = initial_classes.copy()
	remaining_classes = class_shuffle[initial_n_classes:]

	initial_train = []
	initial_train_labels = []
	initial_test = []
	initial_test_labels = []

	for t, tl in zip(train, train_labels):
		if tl in initial_classes:
			initial_train.append(t)
			initial_train_labels.append(tl)

	for t, tl in zip(test, test_labels):
		if tl in initial_classes:
			initial_test.append(t)
			initial_test_labels.append(tl)

	int_initial_train_labels = utils.convert_labels_to_int(initial_train_labels, dict_map)
	int_initial_test_labels = utils.convert_labels_to_int(initial_test_labels, dict_map)

	open_y_test = [x if x in int_initial_train_labels else 0 for x in int_initial_test_labels]

	class_history = []
	class_history.append(initial_classes)
	total_classes = initial_n_classes



	#train i3d
	all_categories_train_fold = gen.get_all_categories(initial_train) 
	dict_map_train_fold = utils.map_labels(all_categories_train_fold)
	all_categories_test_fold = gen.get_all_categories(initial_test)
	dict_map_test_fold = utils.map_labels(all_categories_test_fold)


	try:
		params['model_type'] = 'cnn'
		model = utils.load_i3d_model(params)
		train_features, test_features, int_initial_train_labels ,int_initial_test_labels = utils.load_features(params)
		train_i3d_features.append(train_features)
		train_i3d_labels.append(int_initial_train_labels)
		test_i3d_features.append(test_features)
		test_i3d_labels.append(int_initial_test_labels)

	
	except:
		params['model_type'] = 'cnn'
		model, hist_cnn = finetune_i3d.finetune(initial_train, int_initial_train_labels, dict_map_train_fold, params)
		train_features, test_features = finetune_i3d.extract_features(model, initial_train, initial_train_labels, initial_test, initial_test_labels, dict_map_test_fold, params)
		utils.save_i3d_model(model, params) #very expensive storage
		utils.save_features(train_features, test_features, int_initial_train_labels, int_initial_test_labels, int_initial_test_labels , params)
		train_i3d_features.append(train_features)
		test_i3d_features.append(test_features)
		train_i3d_labels.append(int_initial_train_labels)
		test_i3d_labels.append(int_initial_test_labels)


	
	#train ti3d


	#train triplet net and extract features
	try:
		params['model_type'] = 'triplet'
		ti3d_model = utils.load_ti3d_model(params)
		train_features_ti3d, test_features_ti3d, _, __ = utils.load_ti3d_features(params)

	except Exception as e:
		print(e)
		params['model_type'] = 'triplet'
		train_features_ti3d, test_features_ti3d, hist_triplet, ti3d_model = finetune_i3d.finetune_triplet_net(x_train = train_features, int_y_train = int_initial_train_labels, x_test = test_features, params = params)
		utils.save_ti3d_model(ti3d_model, params) #very expensive storage
		utils.save_ti3d_features(train_features_ti3d, test_features_ti3d, int_initial_train_labels, int_initial_test_labels, int_initial_test_labels , params)
		#K.clear_session() # this is very important

		print(train_features_ti3d.shape)
	
	
	#train evm


	
	try:
		evms_triplet = utils.load_evm_model(params)
		print('successfully loaded evm model')
	except:

		evms_triplet = evm.fit(train_features_ti3d, int_initial_train_labels, params)
		utils.save_evm_models(evms_triplet, params)



	#evaluate

	#todo - define: n train classes / n test classes
	openness = utils.openness(params['n_train_classes'],params['n_test_classes'], params['n_train_classes']) 
	params['openness'] = 0

	
	for current_th in multi_classification_threshold: 
		params['model_type'] = 'phase_1'
    
		params['classification_threshold'] = current_th
		params['model_type'] = 'phase_1'
		# classify triplet model data with evm and get classification metrics

		pred = evm.predict(evms_triplet, test_features_ti3d, params)
		classif_rep, cm = metrics.classif_report(open_y_test, pred)
		youdens_index = metrics.youdens_index(open_y_test, pred)
		closed_f1_score = metrics.closed_f1_score(open_y_test, pred)

		print ('classification report:', classif_rep)
		#print ('confusion matrix:\n', cm)
		print ('youdens index:',youdens_index)
		print ('closed f1 score:', closed_f1_score)

		utils.generate_report(youdens_index, closed_f1_score, classif_rep, cm, params)
		utils.generate_clustering_report(test_features_ti3d,open_y_test,pred, params)


	#end phase 1












	#phase 2									-----------------------
	

	#initial_n_classes = np.random.randint(params['min_classes'],params['max_classes'])
	#initial_classes = class_shuffle[0:initial_n_classes]
	#current_classes = initial_classes.copy()
	#remaining_classes = class_shuffle[initial_n_classes:]

	#class_history = []

	while total_classes < 101:
		params['iteration'] += 1
		#select z new classes

		new_n_classes = np.random.randint(params['min_classes'],params['max_classes'])
		new_classes = class_shuffle[total_classes:total_classes + new_n_classes]
		total_classes += new_n_classes
		print('new selected classes:',new_classes)
		
		#new data

		new_train = []
		new_train_labels = []
		new_test = []
		new_test_labels = []

		for t, tl in zip(train, train_labels):
			if tl in new_classes:
				new_train.append(t)
				new_train_labels.append(tl)

		for t, tl in zip(test, test_labels):
			if tl in new_classes:
				new_test.append(t)
				new_test_labels.append(tl)

		int_new_train_labels = utils.convert_labels_to_int(new_train_labels, dict_map)
		int_new_test_labels = utils.convert_labels_to_int(new_test_labels, dict_map)
		open_new_y_train = [x if x in int_initial_train_labels else 0 for x in int_new_train_labels]
		open_new_y_test = [x if x in int_initial_train_labels else 0 for x in int_new_test_labels]

		print(int_new_test_labels, open_new_y_test)




		#extract features ti3d

		all_categories_new_train = gen.get_all_categories(new_train)
		all_categories_new_test = gen.get_all_categories(new_test)

		dict_map_new_train = utils.map_labels(all_categories_new_train)
		dict_map_new_test = utils.map_labels(all_categories_new_test)

		try:
			params['model_type'] = 'cnn'
			#new_train_features = np.load('new_train_features_'+str(params['iteration'])+'.npy')
			#new_test_features = np.load('new_test_features_'+str(params['iteration'])+'.npy')
			new_train_features, new_test_features, int_new_train_labels, int_new_test_labels = utils.load_features(params, prefix = 'phase_2')
			
			train_i3d_features.append(new_train_features)
			test_i3d_features.append(new_test_features)
			train_i3d_labels.append(int_new_train_labels)
			test_i3d_labels.append(int_new_test_labels)


		except Exception as e:
			print(e)
			params['model_type'] = 'cnn'
			new_train_features = finetune_i3d.extract_features_single(model, new_train, new_train_labels, dict_map_new_train, params, len(np.unique(initial_train_labels)))
			new_test_features = finetune_i3d.extract_features_single(model, new_test, new_test_labels, dict_map_new_test, params, len(np.unique(initial_train_labels)))
			utils.save_features(new_train_features, new_test_features, int_new_train_labels, int_new_test_labels, int_new_test_labels , params, prefix='phase_2')
			#np.save('new_train_features_'+str(params['iteration'])+'.npy', new_train_features)
			#np.save('new_test_features_'+str(params['iteration'])+'.npy', new_test_features)

			train_i3d_features.append(new_train_features)
			test_i3d_features.append(new_test_features)
			train_i3d_labels.append(int_new_train_labels)
			test_i3d_labels.append(int_new_test_labels)

		try:
			params['model_type'] = 'triplet'
			#new_train_triplet_features = np.load('new_train_triplet_features_'+str(params['iteration'])+'.npy')
			#new_test_triplet_features = np.load('new_test_triplet_features_'+str(params['iteration'])+'.npy')
			new_train_triplet_features, new_test_triplet_features, int_new_train_labels, int_new_test_labels = utils.load_ti3d_features(params, prefix = 'phase_2')

		except Exception as e:
			print(e)
			params['model_type'] = 'triplet'

			new_train_triplet_features = finetune_i3d.extract_features_triplet_net(new_train_features, new_train_labels, params, warm_start_model = ti3d_model)
			new_test_triplet_features = finetune_i3d.extract_features_triplet_net(new_test_features, new_test_labels, params, warm_start_model = ti3d_model)
			utils.save_ti3d_features(new_train_triplet_features, new_test_triplet_features, int_new_train_labels, int_new_test_labels, int_new_test_labels , params, prefix='phase_2')

			#np.save('new_train_triplet_features_'+str(params['iteration'])+'.npy',new_train_triplet_features)
			#np.save('new_test_triplet_features_'+str(params['iteration'])+'.npy',new_test_triplet_features)


		print(new_train_features.shape, new_train_triplet_features.shape)
		print(new_test_features.shape, new_test_triplet_features.shape)
		print(len(int_new_train_labels), len(int_new_test_labels))



		#get rejected set (train)
		if params['ignore_errors'] == True:
			rejected_set_idx= np.where(np.array(int_new_train_labels)==0)[0]
			rejected_set_features= np.array(new_train_triplet_features)[rejected_set_idx].copy()
			rejected_set_labels = np.array(int_new_train_labels)[rejected_set_idx].copy()
		else:
			for current_th in multi_classification_threshold:
				params['classification_threshold'] = current_th
				pred = evm.predict(evms_triplet, new_train_triplet_features, params)
				rejected_set_idx = np.where(np.array(pred)==0)[0]
				rejected_set_features = np.array(new_train_triplet_features)[rejected_set_idx].copy() 
				rejected_set_labels = np.array(int_new_train_labels)[rejected_set_idx].copy()
				print(pred)
		

		

		#open set recognition on test set

		#get all known classes	
		known_classes = np.concatenate(class_history).ravel()
		int_known_classes = utils.convert_labels_to_int(known_classes, dict_map)
		known_data = []
		known_data_labels = []
	


		flattened_test_i3d_features, flattened_test_i3d_labels = np.concatenate(test_i3d_features), np.concatenate(test_i3d_labels)
		for t, tl in zip(flattened_test_i3d_features, flattened_test_i3d_labels):
			if tl in int_known_classes:
				known_data.append(t)
				known_data_labels.append(tl)
		known_data = np.array(known_data)
		known_data_labels = np.array(known_data_labels)

		#update all sets with the latest ti3d weights


		known_test_triplet_features = finetune_i3d.extract_features_triplet_net(known_data, known_data_labels, params, warm_start_model = ti3d_model)
		

		
		#merge known and new data

		full_phase_2_data = np.concatenate((known_test_triplet_features,new_test_triplet_features))
		full_phase_2_labels = np.concatenate((known_data_labels,int_new_test_labels))
		full_phase_2_open_set_labels = [x if x in known_data_labels else 0 for x in full_phase_2_labels]
		

		for current_th in multi_classification_threshold:      
			params['classification_threshold'] = current_th
			params['model_type'] = 'phase_2'
			# classify triplet model data with evm and get classification metrics

			pred = evm.predict(evms_triplet, full_phase_2_data, params)
			classif_rep, cm = metrics.classif_report(full_phase_2_open_set_labels, pred)
			youdens_index = metrics.youdens_index(full_phase_2_open_set_labels, pred)
			closed_f1_score = metrics.closed_f1_score(full_phase_2_open_set_labels, pred)
	
			print ('classification report:', classif_rep)
			#print ('confusion matrix:\n', cm)
			print ('youdens index:',youdens_index)
			print ('closed f1 score:', closed_f1_score)
	
			utils.generate_report(youdens_index, closed_f1_score, classif_rep, cm, params)
			utils.generate_clustering_report(full_phase_2_data,full_phase_2_open_set_labels,pred, params)


		input('1')


		#end phase 2





		#phase 3								-----------------------
		
		#estimate number of clusters in the rejected set

		top_k = 5
		estimated_k, gaps = k_estimator.estimate_dendrogap(rejected_set_features_true,top_k, normalize_data = True)
		estimated_k = k_estimator.best_silhouette(rejected_set_features_true, estimated_k, metric = 'cosine')
		print('estimated k:',estimated_k)
		print('true k', len(np.unique(rejected_set_labels_true)))

		#cluster with hierarchical agglomerative ward clustering


		#perform hierarchical
		print('Performing hierarchical clustering with ward linkage')
		#get rejected set labels 
		hierarchical_preds = hierarchical.hierarchical(rejected_set_features_true, n_clusters = estimated_k, affinity = 'euclidean', linkage = 'ward', distance_threshold= None, normalize_data = True)
		
		

		#assign labels
		hierarchical_preds = ['new_class_'+str(x)+'_iter_'+str(params['iteration']) for x in hierarchical_preds ]

		#evaluate clustering performance

		obs_m = utils.build_observation_matrix(hierarchical_preds)
		print(evaluation.clustering_metrics(rejected_set_features_true,rejected_set_labels_true,hierarchical_preds))
		
		#end phase 3

		






		#phase 4								------------------------
	
		#evaluate known test set and unsupervised test set (Before incremental learning)
		#test_features_ti3d, initial_test_labels / new_test_triplet_features, new_test_labels
				

		#get extreme vectors in i3d feature space
		extreme_vectors_features_i3d = []
		extreme_vectors_labels = []
		extreme_vectors_idx, extreme_vectors_keys = evm.extreme_vectors_idx(evms_triplet)

		for count,cl in enumerate(np.unique(int_initial_train_labels)): # train one evm for each positive class
			print ('training evm for class', cl)
			#separate the positive class from the rest
			positives = [x for i,x in enumerate(x_train_features) if int_initial_train_labels[i] == cl]
			print(positives, len(positives))
			for evi in extreme_vectors_idx[count]:
				print(evi)
				extreme_vectors_features_i3d.append(positives[evi])
				extreme_vectors_labels.append(cl)


		extreme_vectors_features_i3d = np.array(extreme_vectors_features_i3d)
		print(extreme_vectors_features_i3d, extreme_vectors_features_i3d.shape)
		print(extreme_vectors_labels)



		'''
		#validate  ----- OK
		extreme_vectors = evm.extreme_vectors(evms_triplet)
		extreme_vectors_t = finetune_i3d.extract_features_triplet_net(extreme_vectors_features_i3d, extreme_vectors_labels, params, warm_start_model = ti3d_model)
		extreme_vectors_t = np.array(extreme_vectors_t)
		extreme_vectors = np.array(extreme_vectors)

		np.save('extreme_vectors_t.npy',extreme_vectors_t )

		for a,b in zip(extreme_vectors,extreme_vectors_t):
			print(a[0:10],b[0:10])
			print(extreme_vectors.shape,extreme_vectors_t.shape)
		'''






		#evaluation before incremental learning
		for current_th in multi_classification_threshold:      
			params['classification_threshold'] = current_th
			params['model_type'] = 'triplet'
			# classify triplet model data with evm and get classification metrics


			preds = evm.predict(evms_triplet, test_features_ti3d, params)
			print('initial test set')
			print(evaluation.clustering_metrics(test_features_ti3d,initial_test_labels,preds))

			print('new test set')
			preds = evm.predict(evms_triplet, new_test_triplet_features, params)
			print(evaluation.clustering_metrics(new_test_triplet_features,new_test_labels,preds))


		#finetune ti3d (incremental mode)

		#something like this
		x_train_features_ti3d_incremental, test_features_ti3d_incremental, hist_triplet, ti3d_model_incremental = finetune_i3d.finetune_triplet_net(x_train = extreme_vectors_features_i3d, int_y_train = extreme_vectors_labels, x_test = test_features, params = params, warm_start_model = ti3d_model)

		print(x_train_features_ti3d_incremental)

		input('jaja')		 

		#increment evm 

		evms = evm.increment_evm(evms_triplet, rejected_set_features_true, hierarchical_preds, params)

		#evaluate known test set and unsupervised test set (After incremental learning)

		for current_th in multi_classification_threshold:      # aqui tem que incluir as rotinas multiparams
			params['classification_threshold'] = current_th
			params['model_type'] = 'triplet'
			# classify triplet model data with evm and get classification metrics


			preds = evm.predict(evms_triplet, test_features_ti3d, params)
			print('initial test set after incrementing evm')
			print(evaluation.clustering_metrics(test_features_ti3d,initial_test_labels,preds))

			print('new test set after incrementing evm')
			preds = evm.predict(evms_triplet, new_test_triplet_features, params)
			print(evaluation.clustering_metrics(new_test_triplet_features,new_test_labels,preds))


		input('jaja')


		#increment known test set (may be changed later)


		#end

		input('end of loop')
