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



	#get filenames
	filenames = gen.get_filenames()	
	all_categories = gen.get_all_categories(filenames)
	dict_map = utils.map_labels(all_categories)

	#get class list
	unique_classes = np.unique([x.split('/')[0] for x in filenames]).tolist()

	PATH_TO_FRAMES = '/home/users/datasets/UCF-101_opticalflow/'


	#phase 1									------------------------

	#select initial classes
	initial_classes, remaining_classes = gen.random_class_picker(unique_classes, params)
	n_picked_classes = len(initial_classes)
	#print(filenames)
	#print(initial_classes)



	#perform train/test split

	initial_train, initial_train_labels, initial_test, initial_test_labels = utils.train_test_split_groups(filenames, initial_classes, params)
	int_initial_train_labels = utils.convert_labels_to_int(initial_train_labels, dict_map)
	int_initial_test_labels = utils.convert_labels_to_int(initial_test_labels, dict_map)
	open_y_test = [x if x in int_initial_train_labels else 0 for x in int_initial_test_labels]



	#train i3d
	all_categories_train_fold = gen.get_all_categories(initial_train) 
	dict_map_train_fold = utils.map_labels(all_categories_train_fold)
	all_categories_test_fold = gen.get_all_categories(initial_test)
	dict_map_test_fold = utils.map_labels(all_categories_test_fold)


	try:
		model = utils.load_i3d_model(params)
		x_train_features, x_test_features, _ ,__ = utils.load_features(params)
	
	except:
		model, hist_cnn = finetune_i3d.finetune(initial_train, int_initial_train_labels, dict_map_train_fold, params)
		x_train_features, x_test_features = finetune_i3d.extract_features(model, initial_train, initial_train_labels, initial_test, initial_test_labels, dict_map_test_fold, params)
		utils.save_i3d_model(model, params) #very expensive storage
		utils.save_features(x_train_features, x_test_features, int_initial_train_labels, int_initial_test_labels, int_initial_test_labels , params)

	#train ti3d


	#train triplet net and extract features
	try:
		ti3d_model = utils.load_ti3d_model(params)
		x_train_features_ti3d, x_test_features_ti3d, _, __ = utils.load_ti3d_features(params)

	except Exception as e:
		print(e)
		x_train_features_ti3d, x_test_features_ti3d, hist_triplet, ti3d_model = finetune_i3d.finetune_triplet_net(x_train = x_train_features, int_y_train = int_initial_train_labels, x_test = x_test_features, params = params)
		utils.save_ti3d_model(ti3d_model, params) #very expensive storage
		utils.save_ti3d_features(x_train_features_ti3d, x_test_features_ti3d, int_initial_train_labels, int_initial_test_labels, int_initial_test_labels , params)
		#K.clear_session() # this is very important

		print(x_train_features_ti3d.shape)
	

	#train evm


	params['model_type'] = 'triplet'
	
	try:
		evms_triplet = utils.load_evm_model(params)
		print('successfully loaded evm model')
	except:

		evms_triplet = evm.fit(x_train_features_ti3d, int_initial_train_labels, params)
		utils.save_evm_models(evms_triplet, params)




	#evaluate

	openness = utils.openness(params['n_train_classes'],params['n_test_classes'], params['n_train_classes']) 
	params['openness'] = openness

	'''
	for current_th in multi_classification_threshold:      # aqui tem que incluir as rotinas multiparams
		params['classification_threshold'] = current_th
		params['model_type'] = 'triplet'
		# classify triplet model data with evm and get classification metrics

		pred = evm.predict(evms_triplet, x_test_features_ti3d, params)
		classif_rep, cm = metrics.classif_report(open_y_test, pred)
		youdens_index = metrics.youdens_index(open_y_test, pred)
		closed_f1_score = metrics.closed_f1_score(open_y_test, pred)

		print ('classification report:', classif_rep)
		#print ('confusion matrix:\n', cm)
		print ('youdens index:',youdens_index)
		print ('closed f1 score:', closed_f1_score)

		utils.generate_report(youdens_index, closed_f1_score, classif_rep, cm, params)
	'''


	#end phase 1









	#phase 2									-----------------------
	new_classes_list = []
	
	iteration = 0 
	while n_picked_classes < 101:
		#select z new classes
		new_classes, remaining_classes = gen.random_class_picker(remaining_classes, params)
		new_classes_list.append(new_classes)
		print(new_classes)
		n_picked_classes += len(new_classes)

		#perform unsupervised train/test split


		new_train, new_train_labels, new_test, new_test_labels = utils.train_test_split_groups(filenames, new_classes, params)
		int_new_train_labels = utils.convert_labels_to_int(new_train_labels, dict_map)
		int_new_test_labels = utils.convert_labels_to_int(new_test_labels, dict_map)
		open_new_y_test = [x if x in int_new_train_labels else 0 for x in int_new_test_labels]


		print(new_train)

		


		#create open-set #iteration
		open_set_i = initial_test + new_train
		open_set_i_labels = initial_test_labels + new_train_labels
	


		#extract features ti3d

		all_categories_open_set_i = gen.get_all_categories(open_set_i)
		all_categories_new_test = gen.get_all_categories(new_test)

		dict_map_open_set_i = utils.map_labels(all_categories_open_set_i)
		dict_map_new_test = utils.map_labels(all_categories_new_test)

		try:
			open_set_i_features = np.load('open_set_i_features.npy')
			new_test_features = np.load('new_test_features.npy')
		except Exception as e:
			print(e)
			open_set_i_features = finetune_i3d.extract_features_single(model, open_set_i, open_set_i_labels, dict_map_open_set_i, params, len(np.unique(initial_train_labels)))
			new_test_features = finetune_i3d.extract_features_single(model, new_test, new_test_labels, dict_map_new_test, params, len(np.unique(initial_train_labels)))

			np.save('open_set_i_features.npy', open_set_i_features)
			np.save('new_test_features.npy', new_test_features)

		try:
			open_set_i_triplet_features = np.load('open_set_i_triplet_features.npy')
			new_test_triplet_features = np.load('new_test_triplet_features.npy')

		except Exception as e:
			print(e)
			open_set_i_triplet_features = finetune_i3d.extract_features_triplet_net(open_set_i_features, open_set_i_labels, params, warm_start_model = ti3d_model)
			new_test_triplet_features = finetune_i3d.extract_features_triplet_net(new_test_features, new_test_labels, params, warm_start_model = ti3d_model)
			np.save('open_set_i_triplet_features.npy',open_set_i_triplet_features)
			np.save('new_test_triplet_features.npy',new_test_triplet_features)

		#print(open_set_i_triplet_features, open_set_i_triplet_features.shape)

		#open set recognition on open-set #iteration
		int_open_set_i_labels = utils.convert_labels_to_int(open_set_i_labels, dict_map)
		new_open_y_test = [x if x in int_initial_train_labels else 0 for x in int_open_set_i_labels]
	
		for current_th in multi_classification_threshold:      # aqui tem que incluir as rotinas multiparams
			params['classification_threshold'] = current_th
			params['model_type'] = 'triplet'
			# classify triplet model data with evm and get classification metrics

			pred = evm.predict(evms_triplet, open_set_i_triplet_features, params)
			classif_rep, cm = metrics.classif_report(new_open_y_test, pred)
			youdens_index = metrics.youdens_index(new_open_y_test, pred)
			closed_f1_score = metrics.closed_f1_score(new_open_y_test, pred)

			print ('classification report:', classif_rep)
			#print ('confusion matrix:\n', cm)
			print ('youdens index:',youdens_index)
			print ('closed f1 score:', closed_f1_score)

			utils.generate_report(youdens_index, closed_f1_score, classif_rep, cm, params)




		#get rejected set

		rejected_set_idx = np.where(np.array(pred)==0)[0]
		rejected_set_idx_true = np.where(np.array(new_open_y_test)==0)[0]

		rejected_set_features = np.array(open_set_i_triplet_features)[rejected_set_idx].copy() 
		rejected_set_features_true = np.array(open_set_i_triplet_features)[rejected_set_idx_true].copy()
		
		rejected_set_labels = np.array(open_set_i_labels)[rejected_set_idx].copy()
		rejected_set_labels_true = np.array(open_set_i_labels)[rejected_set_idx_true].copy()



		#print(rejected_set_idx, rejected_set_idx_true)

		#open set evaluation

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
		hierarchical_preds = ['new_class_'+str(x)+'_iter_'+str(iteration) for x in hierarchical_preds ]

		#evaluate clustering performance

		obs_m = utils.build_observation_matrix(hierarchical_preds)
		print(evaluation.clustering_metrics(rejected_set_features_true,rejected_set_labels_true,hierarchical_preds))
		
		#end phase 3

		






		#phase 4								------------------------
	
		#evaluate known test set and unsupervised test set (Before incremental learning)
		#x_test_features_ti3d, initial_test_labels / new_test_triplet_features, new_test_labels
				


		for current_th in multi_classification_threshold:      # aqui tem que incluir as rotinas multiparams
			params['classification_threshold'] = current_th
			params['model_type'] = 'triplet'
			# classify triplet model data with evm and get classification metrics


			preds = evm.predict(evms_triplet, x_test_features_ti3d, params)
			print('initial test set')
			print(evaluation.clustering_metrics(x_test_features_ti3d,initial_test_labels,preds))

			print('new test set')
			preds = evm.predict(evms_triplet, new_test_triplet_features, params)
			print(evaluation.clustering_metrics(new_test_triplet_features,new_test_labels,preds))


		#finetune ti3d (incremental mode)

		#something like this
		#x_train_features_ti3d, x_test_features_ti3d, hist_triplet, ti3d_model = finetune_i3d.finetune_triplet_net(x_train = x_train_features, int_y_train = int_initial_train_labels, x_test = x_test_features, params = params, warm_start_model = ti3d_model)

		#increment evm 

		evms = evm.increment_evm(evms_triplet, rejected_set_features_true, hierarchical_preds, params)

		#evaluate known test set and unsupervised test set (After incremental learning)

		for current_th in multi_classification_threshold:      # aqui tem que incluir as rotinas multiparams
			params['classification_threshold'] = current_th
			params['model_type'] = 'triplet'
			# classify triplet model data with evm and get classification metrics


			preds = evm.predict(evms_triplet, x_test_features_ti3d, params)
			print('initial test set after incrementing evm')
			print(evaluation.clustering_metrics(x_test_features_ti3d,initial_test_labels,preds))

			print('new test set after incrementing evm')
			preds = evm.predict(evms_triplet, new_test_triplet_features, params)
			print(evaluation.clustering_metrics(new_test_triplet_features,new_test_labels,preds))


		input('jaja')


		#increment known test set (may be changed later)


		iteration += 1
		#end

		input('end of loop')

