import numpy as np
from sklearn import metrics as me
import os
import utils


def silhouette_scores(x,pred,metric, output_path):
	sil_scores = []
	min_k = np.unique(pred[0]).shape[0]
	max_k = min_k + len(pred) -1
	#print ('\n\nmin k', min_k, max_k)
	for p in pred:
		sil_scores.append(me.silhouette_score(x,p,metric=metric))

	plots.plot_silhouette(sil_scores, min_k, max_k, output_path)

	return sil_scores


def clustering_metrics(x,y,pred):

	completeness_score = me.completeness_score(y,pred)
	homogeneity_score = me.homogeneity_score(y,pred)
	vmeasure_score = me.v_measure_score(y,pred)
	adjusted_rand_score = me.adjusted_rand_score(y,pred)
	adjusted_mutual_info_score = me.adjusted_mutual_info_score(y,pred)

	try:
		calinski_harabasz_score = me.calinski_harabasz_score(x,pred)
	except:
		calinski_harabasz_score = 0

	try:
		davies_bouldin_score = me.davies_bouldin_score(x,pred)
	except:
		davies_bouldin_score = 0

	fowlkes_mallows_score = me.fowlkes_mallows_score(y,pred)
	mutual_info_score = me.mutual_info_score(y,pred)
	normalized_mutual_info_score = me.normalized_mutual_info_score(y,pred)

	try:
		sil_score_euc = me.silhouette_score(x,pred,metric='euclidean')
	except:
		sil_score_euc = 0

	try:
		sil_score_cos = me.silhouette_score(x,pred,metric='cosine')
	except:
		sil_score_cos = 0	

	dict = {}
	dict['completeness'] = completeness_score
	dict['homogeneity'] = homogeneity_score
	dict['vmeasure'] = vmeasure_score
	dict['adjusted_rand'] = adjusted_rand_score
	dict['adjusted_mutual_info'] = adjusted_mutual_info_score
	dict['calinski_harabasz'] = calinski_harabasz_score 
	dict['davies_bouldin'] = davies_bouldin_score
	dict['fowlkes_mallows'] = fowlkes_mallows_score
	dict['mutual_info'] = mutual_info_score
	dict['normalized_mutual_info'] = normalized_mutual_info_score
	dict['silhouette_score_euc'] = sil_score_euc
	dict['silhouette_score_cos'] = sil_score_cos 
	dict['k'] = np.unique(pred).shape[0]
	dict['n_clusters_gt'] = np.unique(y).shape[0]


	return dict

def report(x,y,pred, output_path, **kwargs):
	utils.make_dirs(output_path)
	utils.make_dirs(output_path+'metrics/')
	#utils.make_dirs(output_path+'plots/silhouette/euclidean/')
	#utils.make_dirs(output_path+'plots/silhouette/cosine/')

	#get metrics
	results = clustering_metrics(x,y,pred)	

	utils.save_report(results, output_path+'metrics/', additional_info = kwargs)

	#save predictions
	utils.save_predictions(pred, output_path)







def full_evaluation(results, params):
	#results is a list of dictionaries, where each dictionary represents one iteration of results
	
	#dict['x'] are features
	#dict['y'] are the gt labels
	#dict['preds'] are the predictions
	#dict['tasks'] are the classes included in each task. Each position of the array is an array of classes

	#retorna 
	#	  forgetting_per_task: um array onde cada posicao eh um dicionario que contem o forgetting de uma das tasks. Leva em consideracao todas as metricas.
	#	  per_iteration_metrics: metricas por task para cada iteracao


	

	per_iteration_metrics = []
	for r in results:
		per_task_metrics = []
		x = r['x']
		y = r['y']
		preds = r['preds']
		tasks = r['tasks']

		#print('tasks:',tasks)
		#print('y',y)
		for h,t in enumerate(tasks):
			#print('\n\n',h,'\n\n')
			relevant_x = [x[i] for i in range(len(y)) if y[i] in t]
			relevant_y = [y[i] for i in range(len(y)) if y[i] in t]
			relevant_preds = [preds[i] for i in range(len(y)) if y[i] in t]

			#print('relevant y',relevant_y)
			#print(relevant_preds)
			#print('task',t)
			#print(y)
			
			cl_metrics = clustering_metrics(relevant_x,relevant_y,relevant_preds)
			#print(relevant_y)
			#print(relevant_preds)
			#for k,v in cl_metrics.items():
			#	print(k,v)
			
			per_task_metrics.append(cl_metrics)
			#input('end of task')

		#input('end of iteration')
		per_iteration_metrics.append(per_task_metrics)


	#compute forgetting for each task, starting at task 0...n
	forgetting_per_task = []

	total_tasks = len(results[-1]['tasks'])
	#print('total tasks:',total_tasks)	


	for i in range(total_tasks):
		metrics_current_task = []

		#find maximum value of each metric in task i
		#print('current task:',i)
		for j,p in enumerate(per_iteration_metrics):
			if i <= j:
				#print("current j",j)
				#print(p[i])
				metrics_current_task.append(p[i])
				#input('')
		#print(metrics_current_task)

		keys = metrics_current_task[0].keys()
		max_metrics_task = {}
		forgetting_task = {}
		#print(max_metrics_task)
		for k in keys:
			print(k)
			current_metrics = [r[k] for r in metrics_current_task]
			print(current_metrics)
			max_metrics_task[k] = np.amax(current_metrics)
			forgetting_task[k] =  np.amax(current_metrics) - current_metrics[-1]
	
			#print(max_metrics_task[k])
			#print(forgetting_task[k])	

			#input('m')
		forgetting_per_task.append(forgetting_task)


	
	return forgetting_per_task, per_iteration_metrics


def single_evaluation_clustering(x,y,pred,params, return_dict=False,**kwargs):
	
	import evaluation
	from utils import makedirs, gen_exp_id

	output_path = params['output_path']
	exp_id = gen_exp_id(params)

	output_path += exp_id
	output_path += str(params['fold']) + '/'
	output_path += str(params['model_type']) + '/'
	output_path += str(params['iteration']) + '/'
	output_path += str(params['classification_threshold']) + '/'
	makedirs(output_path)

	results = evaluation.clustering_metrics(x,y,pred)

	if return_dict:
		return results

	f = open(output_path+'clustering_report.txt', 'w')


	f.write('Ground truth number of clusters = ' + str(results['n_clusters_gt'])+'\n')
	f.write('Number of clusters found = '+str(results['k'])+'\n')

	#check kwargs for additional info
	if kwargs.get('additional_info') is not None:
		dict = kwargs.get('additional_info')
		for key,value in dict.items():
			print(key,value)
			f.write(str(key)+': ' +str(value)+'\n')

	f.write('\nHomogeneity: '+str(results['homogeneity'])+'\n')
	f.write('Completeness: '+str(results['completeness'])+'\n')	
	f.write('V-measure: '+str(results['vmeasure'])+'\n')	
	f.write('Adjusted Rand score: '+str(results['adjusted_rand'])+'\n')
	f.write('Adjusted mutual info: '+str(results['adjusted_mutual_info'])+'\n')	
	f.write('Calinski Harabasz score: '+str(results['calinski_harabasz'])+'\n')
	f.write('Davies Bouldin score: '+str(results['davies_bouldin'])+'\n')
	f.write('Fowlkes Mallows score: '+str(results['fowlkes_mallows'])+'\n')
	f.write('Mutual Info score: '+str(results['mutual_info'])+'\n')
	f.write('Normalized Mutual Info score: '+str(results['normalized_mutual_info'])+'\n')
	f.write('Silhouette score (Euclidean distance): '+str(results['silhouette_score_euc'])+'\n')
	f.write('Silhouette score (Cosine distance): '+str(results['silhouette_score_cos'])+'\n')
 

	f.write('\n\n')

	f.close()
	
	



def single_evaluation_openset(y,pred,params):
	import classification_metrics as metrics
	import utils
	import re

	y = [int(re.search(r'\d+', str(x)).group()) for x in y]
	pred = [int(re.search(r'\d+', str(x)).group()) for x in pred]

	
	classif_rep, cm = metrics.classif_report(y, pred)
	youdens_index = metrics.youdens_index(y, pred)
	closed_f1_score = metrics.closed_f1_score(y, pred)
	
	print ('classification report:', classif_rep)
	print ('confusion matrix:\n', cm)
	print ('youdens index:',youdens_index)
	print ('closed f1 score:', closed_f1_score)
	
	utils.generate_report(youdens_index, closed_f1_score, classif_rep, cm, params)









