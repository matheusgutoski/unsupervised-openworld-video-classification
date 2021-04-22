import EVM
import numpy as np
import os
import scipy


def fit(x_train, y_train, params, i3d_features = None):
	evms = {}
	params = params.copy()
	if params['tail_size'] <= 1.0:
		params['tail_size'] = len(x_train)*params['tail_size']
	print('tail size:', params['tail_size'])

	if i3d_features is not None:
		original_extreme_vectors, original_extreme_vectors_labels = [],[]

	for cl in np.unique(y_train): # train one evm for each positive class
		print ('training evm for class', cl)
		#separate the positive class from the rest
		positives = [x for i,x in enumerate(x_train) if y_train[i] == cl]
		negatives = [x for i,x in enumerate(x_train) if y_train[i] != cl]

		if i3d_features is not None:
			positives_i3d =  [x for i,x in enumerate(i3d_features) if y_train[i] == cl]

		evm = EVM.EVM(tailsize=params['tail_size'], cover_threshold = params['cover_threshold'], distance_function=scipy.spatial.distance.cosine)
		#evm = EVM.EVM(tailsize=params['tail_size'], cover_threshold = params['cover_threshold'], distance_function=scipy.spatial.distance.euclidean)
		#evm.train(positives = positives, negatives = negatives, parallel = 8)
		evm.train(positives = positives, negatives = negatives)
		evms[cl] = evm

		if i3d_features is not None:
			print(evm._extreme_vectors)
			for ev in evm._extreme_vectors:
				original_extreme_vectors.append(positives_i3d[ev])
				original_extreme_vectors_labels.append(cl)
				

	if i3d_features is not None:
		return evms, original_extreme_vectors, original_extreme_vectors_labels
	else:
		return evms



def predict(evms, x_test, params):
        print ('predicting samples')
        predictions = []
        probabilities_per_evm = {}
        for key, evm in evms.items():    
                probabilities = evm.probabilities(x_test)
                max_prob = np.amax(probabilities, axis = 1)
                probabilities_per_evm[key] = max_prob

        keys = list(probabilities_per_evm.keys())
        values = np.array(list(probabilities_per_evm.values()))

        max_i = np.array(np.max(values, axis = 0))
        argmax_i = np.argmax(values, axis=0)

        for m, argm in zip(max_i, argmax_i):
                if m > params['classification_threshold']:
                        predictions.append(keys[argm])
                else:
                        predictions.append(0)

        return predictions


def increment_evm_fixed_rep(evms, x_train, y_train, params):
	
	print('incrementing evms...')

	#get previous extreme vectors


	extreme_negatives = []
	for evm in evms:
		extreme_negatives.append(evms[evm]._positives[evms[evm]._extreme_vectors])

	extreme_negatives = np.vstack(extreme_negatives)

	for cl in np.unique(y_train): # train one evm for each positive class
		print ('training evm for class', cl)
		#separate the positive class from the rest
		positives = [x for i,x in enumerate(x_train) if y_train[i] == cl]
		negatives = [x for i,x in enumerate(x_train) if y_train[i] != cl]

		negatives = negatives + extreme_negatives.tolist()
		evm = EVM.EVM(tailsize=params['tail_size'], cover_threshold = params['cover_threshold'], distance_function=scipy.spatial.distance.cosine)
		#evm = EVM.EVM(tailsize=params['tail_size'], cover_threshold = params['cover_threshold'], distance_function=scipy.spatial.distance.euclidean)
		evm.train(positives = positives, negatives = negatives, parallel = 8)
		evms[cl] = evm




	return evms


def extreme_vectors_idx(evms):
	extreme_vector_idx = []
	labels = []
	for key,evm in evms.items():
		labels.append(key)
	for evm in evms:
		extreme_vector_idx.append(evms[evm]._extreme_vectors)
		print(evms[evm]._extreme_vectors)
	print('extreme vector indexes:',extreme_vector_idx)
	print('extreme vector labels:', labels)
	return extreme_vector_idx, labels


def extreme_vectors(evms):
	extreme_vectors = []
	for evm in evms:
		extreme_vectors.append(evms[evm]._positives[evms[evm]._extreme_vectors])

	extreme_vectors = np.vstack(extreme_vectors)
	return extreme_vectors


def increment_evm(extreme_vectors, extreme_vectors_labels, new_train_features, new_train_labels, params, i3d_features = None):
	print('Training incremental evms...')
	evms = {}
	params = params.copy()

	if params['tail_size'] <= 1.0:
		params['tail_size'] = len(new_train_features)*params['tail_size']
	print('tail size:', params['tail_size'])

	if i3d_features is not None:
		original_extreme_vectors, original_extreme_vectors_labels = [],[]

	for cl in np.unique(extreme_vectors_labels): # train one evm for each extreme vector class
		print ('training evm for extreme vector class', cl)
		#separate the positive class from the rest
		positives = [x for i,x in enumerate(extreme_vectors) if extreme_vectors_labels[i] == cl]
		negatives = [x for i,x in enumerate(extreme_vectors) if extreme_vectors_labels[i] != cl]

		negatives = negatives + new_train_features.tolist()
		evm = EVM.EVM(tailsize=params['tail_size'], cover_threshold = None, distance_function=scipy.spatial.distance.cosine)
		#evm.train(positives = positives, negatives = negatives, parallel = 8)
		evm.train(positives = positives, negatives = negatives)

		evms[cl] = evm

	for cl in np.unique(new_train_labels): # train one evm for each extreme vector class
		print ('training evm for new class', cl)
		#separate the positive class from the rest
		positives = [x for i,x in enumerate(new_train_features) if new_train_labels[i] == cl]
		negatives = [x for i,x in enumerate(new_train_features) if new_train_labels[i] != cl]

		if i3d_features is not None:
			positives_i3d =  [x for i,x in enumerate(i3d_features) if new_train_labels[i] == cl]

		negatives = negatives + extreme_vectors.tolist()
		evm = EVM.EVM(tailsize=params['tail_size'], cover_threshold = params['cover_threshold'], distance_function=scipy.spatial.distance.cosine)
		#evm.train(positives = positives, negatives = negatives, parallel = 8)
		evm.train(positives = positives, negatives = negatives)
		evms[cl] = evm

		if i3d_features is not None:
			print(evm._extreme_vectors)
			for ev in evm._extreme_vectors:
				original_extreme_vectors.append(positives_i3d[ev])
				original_extreme_vectors_labels.append(cl)
				

	if i3d_features is not None:
		return evms, original_extreme_vectors, original_extreme_vectors_labels
	else:
		return evms
