import evaluation


def testcase():
	import numpy as np

	#test case
	results = []
	params = {}

	#iter 1

	dict = {}
	dict['x'] = np.random.rand(9,2)
	dict['y'] = [1,1,1,2,2,2,3,3,3]
	dict['preds'] = [1,1,2,2,2,3,1,3,3]
	dict['tasks'] = [[1,2,3]]
	results.append(dict)

	#iter 2

	dict = {}
	dict['x'] = np.random.rand(15,100)
	dict['y'] = [1,1,1,2,2,2,3,3,3,4,4,5,5,6,6]
	dict['preds'] = [2,1,2,2,2,3,1,3,3,4,4,5,5,5,6]
	dict['tasks'] = [[1,2,3],[4,5,6]]
	results.append(dict)

	#iter 3

	dict = {}
	dict['x'] = np.random.rand(21,100)
	dict['y'] = [1,1,1,2,2,2,3,3,3,4,4,5,5,6,6,7,7,8,8]
	dict['preds'] = [5,7,2,2,2,3,1,3,3,4,4,5,5,5,6,7,7,8,8]
	dict['tasks'] = [[1,2,3],[4,5,6],[7,8]]
	results.append(dict)

	forgetting, full_evaluation = evaluation.full_evaluation(results, params = [])

	print(forgetting)

testcase()