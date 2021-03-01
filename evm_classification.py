import EVM
import numpy as np
import os
import scipy
def fit(x_train, y_train, params):
        evms = {}
        for cl in np.unique(y_train): # train one evm for each positive class
                print ('training evm for class', cl)
                #separate the positive class from the rest
                positives = [x for i,x in enumerate(x_train) if y_train[i] == cl]
                negatives = [x for i,x in enumerate(x_train) if y_train[i] != cl]

                evm = EVM.EVM(tailsize=params['tail_size'], cover_threshold = params['cover_threshold'], distance_function=scipy.spatial.distance.cosine)
                #evm = EVM.EVM(tailsize=params['tail_size'], cover_threshold = params['cover_threshold'], distance_function=scipy.spatial.distance.euclidean)
                evm.train(positives = positives, negatives = negatives, parallel = 8)

                evms[cl] = evm

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


def increment_evm(evms, x_train, y_train, params):
	
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

	print('extreme vector indexes:',extreme_vector_idx)
	print('extreme vector labels:', labels)
	return extreme_vector_idx, labels


def extreme_vectors(evms):
	extreme_vectors = []
	for evm in evms:
		extreme_vectors.append(evms[evm]._positives[evms[evm]._extreme_vectors])

	extreme_vectors = np.vstack(extreme_vectors)
	return extreme_vectors