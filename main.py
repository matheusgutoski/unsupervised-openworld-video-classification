import numpy as np
import generator as gen
import finetune_i3d 
import utils
import os
import argparse
from keras.models import load_model
from keras import backend as K 

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

	#os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)


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

	#print(filenames)
	#print(initial_classes)



	#perform train/test split

	initial_train, initial_train_labels, initial_test, initial_test_labels = utils.train_test_split_groups(filenames, initial_classes, params)
	int_initial_train_labels = utils.convert_labels_to_int(initial_train_labels, dict_map)
	int_initial_test_labels = utils.convert_labels_to_int(initial_test_labels, dict_map)

	


	#train i3d

	all_categories_train_fold = gen.get_all_categories(initial_train) 
	dict_map_train_fold = utils.map_labels(all_categories_train_fold)
	all_categories_test_fold = gen.get_all_categories(initial_test)
	dict_map_test_fold = utils.map_labels(all_categories_test_fold)


	try:
		model = utils.load_i3d_model(params)
		x_train_features,_, x_test_features,__ = utils.load_features(params)
	except:
		#model, hist_cnn = finetune_i3d.finetune(initial_train, int_initial_train_labels, dict_map_train_fold, params)
		x_train_features, x_test_features = finetune_i3d.extract_features(model, initial_train, initial_train_labels, initial_test, initial_test_labels, dict_map_test_fold, params)
		utils.save_i3d_model(model, params) #very expensive storage
		utils.save_features(x_train_features, x_test_features, int_initial_train_labels, int_initial_test_labels, int_initial_test_labels , params)
		K.clear_session() # this is very important

	#train ti3d

	input('jeje')

	#train triplet net and extract features
	x_train_features_triplet, x_test_features_triplet, hist_triplet = finetune_i3d.finetune_triplet_net(x_train_features, int_initial_train_labels, x_test_features, int_initial_test_labels, params)
	print(x_train_features_triplet.shape)
	input()
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


