from sklearn.model_selection import train_test_split
import numpy as np

PATH_TO_LISTS = '/home/users/matheus/doutorado/i3d/keras-kinetics-i3d/data/ucf101/'
LIST_FILENAMES = ['full_list.txt']


def get_video_list(list_id):
        with open(PATH_TO_LISTS+LIST_FILENAMES[list_id]) as f:
                lines = f.read().splitlines()

        return lines

def get_all_categories(LIST):
        all_categories = np.unique([x.split('/')[0] for x in LIST])
        return all_categories


def generate_ucf101_folds(FULL_LIST, params):
        unique_classes = np.unique([x.split('/')[0] for x in FULL_LIST])
        total_classes = len(unique_classes)

        for fold in range(params['n_folds']):
                available_classes = unique_classes
                # first select n_train_classes at random from available classes to be known classes
                known_classes = np.random.choice(unique_classes, params['n_train_classes'], replace = False)
                # remove selected classes from the pool of available classes
                available_classes = [i for i in available_classes if i not in known_classes]
                # select n_test_classes - n_train_classes at random from available classes (to be unknown classes)
                unknown_classes = np.random.choice(available_classes, params['n_test_classes'] - params['n_train_classes'], replace = False)
                #split known classes into training and test knowns. Split must ensure each group of subclasses remain united
                known_videos = [x.split('.')[0] for x in FULL_LIST if x.split('/')[0] in known_classes]
                unknown_videos = [x.split('.')[0] for x in FULL_LIST if x.split('/')[0] in unknown_classes]
                unknown_labels = [x.split('/')[0] for x in unknown_videos]

                merged_list = [] #this list is for merging groups of the same class so that we can use sklearns train_test_split method. Each element in the list is a group of videos that belong to the same class and group
                for cl in known_classes:
                        videos_current_class = sorted([x for x in known_videos if x.split('/')[0] == cl])
                        groups_current_class = np.unique([x.split('_')[2] for x in videos_current_class])
                        for g in groups_current_class:
                                videos_group_current_class = sorted([x for x in videos_current_class if x.split('_')[2] == g])
                                merged_list.append(videos_group_current_class)
                   
                merged_list_labels = [x[0].split('/')[0] for x in merged_list]

                # employ sklearns train test split method to generate train/test knowns
                merged_knowns_train, merged_knowns_test, merged_knowns_train_labels , merged_knowns_test_labels = train_test_split(merged_list, merged_list_labels, train_size = 0.7, random_state = params['seed'], shuffle = True, stratify = merged_list_labels)

                #unmerge lists
                train = []
                train_labels = []
                for m, l in zip(merged_knowns_train, merged_knowns_train_labels):
                        for n in m:
                                train.append(n)
                                train_labels.append(l)
                
                known_test = []
                known_test_labels = []
                for m, l in zip(merged_knowns_test, merged_knowns_test_labels):
                        for n in m:
                                known_test.append(n)
                                known_test_labels.append(l)
                #join knowns test and unknowns 
                test = known_test + unknown_videos
                test_labels = known_test_labels + unknown_labels      
                '''        
                # actually read the features using the generated lists
                train, _, __ = read_dataset(train, params['model'])
                test, _, __ = read_dataset(test, params['model'])

                train = np.round(train, decimals=3)
                test = np.round(test, decimals=3)
                '''
                yield train, train_labels, test, test_labels
                params['seed'] += 1




def read_kinetics_features(train, test):
        features_path = '/home/users/matheus/doutorado/i3d/keras-kinetics-i3d/git/keras_i3d/features_i3d_video/'

        train_features = []
        test_features = []

        for t in train:
                feats = np.load(features_path + t + '/kinetics_features.npy')
                train_features.append(feats)

        for t in test:
                feats = np.load(features_path + t + '/kinetics_features.npy')
                test_features.append(feats)

        return np.squeeze(np.array(train_features)), np.squeeze(np.array(test_features))



def get_filenames():
	full_list_path = '/home/users/matheus/doutorado/i3d/keras-kinetics-i3d/data/ucf101/full_list.txt'
	with open(full_list_path) as f:
        	lines = f.read().splitlines()
	return lines

def class_generator(params=0):
	import random	
	import numpy as np
	np.random.seed(2021)
	random.seed(2021)

	lines = get_filenames()
	unique_classes = np.unique([x.split('/')[0] for x in lines]).tolist()
	total_classes = len(unique_classes)
	remaining_classes = unique_classes
	

	#print(unique_classes,total_classes)
	
	classes_left = True
	while classes_left:
		if len(remaining_classes) >= params['max_classes']:
			selected_classes = np.random.choice(remaining_classes, random.randint(params['min_classes'], params['max_classes']), replace = False).tolist()
			#print(selected_classes)
			for s in selected_classes:
				remaining_classes.remove(s)

			yield selected_classes
		else:
			yield remaining_classes
			classes_left = False



def random_class_picker(classes, params):
	import random	
	import numpy as np
	np.random.seed(2021)
	random.seed(2021)


	remaining_classes = classes

	try:
		selected_classes = np.random.choice(remaining_classes, random.randint(params['min_classes'], params['max_classes']), replace = False).tolist()
	except Exception as e:
		print(e)
		print('returning all remaining classes')
		selected_classes = remaining_classes

	for s in selected_classes:
		remaining_classes.remove(s)

	if len(remaining_classes) == 1:
		selected_classes.append(remaining_classes[0])

	return selected_classes, remaining_classes

'''
params = {}

params['min_classes'] = 2
params['max_classes'] = 10

sum = 0

filenames = get_filenames()


iteration = 0
for selected in class_generator(params):
	print(selected)
	sum += len(selected)

	data, labels, groups = read_data(selected, filenames)
	known_x_train, known_y_train, known_x_test, known_y_test = train_test_group_split(groups, data, labels)


	#first iteration. Initialize and train initial models
	if iteration == 0:
		i3d_model = initialize_i3d()
		train_i3d(i3d_model, known_x_train, known_y_train, params)
	
		ti3d_model = initialize_ti3d()
		train_ti3d(ti3d_model, known_x_train, known_y_train,params)
		save_model(ti3d_model, params)


		known_x_train_features, known_x_test_features = extract_features_ti3d(ti3d_model, known_x_train, known_x_test)

		evm_model = initialize_evm()
		train_evm(known_x_train_features, known_y_train, params)
		save_model(evm_model,params)

	else:
		aa

	iteration += 1
print(sum)

'''

