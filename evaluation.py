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
		calinski_harabasz_score = 'undefined'

	try:
		davies_bouldin_score = me.davies_bouldin_score(x,pred)
	except:
		davies_bouldin_score = 'undefined'

	fowlkes_mallows_score = me.fowlkes_mallows_score(y,pred)
	mutual_info_score = me.mutual_info_score(y,pred)
	normalized_mutual_info_score = me.normalized_mutual_info_score(y,pred)

	try:
		sil_score_euc = me.silhouette_score(x,pred,metric='euclidean')
	except:
		sil_score_euc = 'undefined'

	try:
		sil_score_cos = me.silhouette_score(x,pred,metric='cosine')
	except:
		sil_score_cos = 'undefined'	

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















#discontinued functions
'''
def clustering_metrics(x,y,pred):
	completeness_scores = []
	homogeneity_scores = []
	vmeasure_scores = []	
	adjusted_rand_scores = []
	adjusted_mutual_info_scores = []

	#new
	calinski_harabasz_scores = []
	davies_bouldin_scores = []
	fowlkes_mallows_scores = []
	mutual_info_scores = []
	normalized_mutual_info_scores = []
 


	for p in pred:
		completeness_scores.append(me.completeness_score(y,p))
		homogeneity_scores.append(me.homogeneity_score(y,p))
		vmeasure_scores.append(me.v_measure_score(y,p))
		adjusted_rand_scores.append(me.adjusted_rand_score(y,p))
		adjusted_mutual_info_scores.append(me.adjusted_mutual_info_score(y,p))

		calinski_harabasz_scores.append(me.calinski_harabasz_score(x,p))
		davies_bouldin_scores.append(me.davies_bouldin_score(x,p))
		fowlkes_mallows_scores.append(me.fowlkes_mallows_score(y,p))
		mutual_info_scores.append(me.mutual_info_score(y,p))
		normalized_mutual_info_scores.append(me.normalized_mutual_info_score(y,p))
	


	dict = {}
	dict['completeness'] = completeness_scores
	dict['homogeneity'] = homogeneity_scores
	dict['vmeasure'] = vmeasure_scores
	dict['adjusted_rand'] = adjusted_rand_scores
	dict['adjusted_mutual_info'] = adjusted_mutual_info_scores

	dict['calinski_harabasz'] = calinski_harabasz_scores 
	dict['davies_bouldin'] = davies_bouldin_scores
	dict['fowlkes_mallows'] = fowlkes_mallows_scores
	dict['mutual_info'] = mutual_info_scores
	dict['normalized_mutual_info'] = normalized_mutual_info_scores
 
 


	return dict

def report(x,y,pred, output_path):
	utils.make_dirs(output_path)
	utils.make_dirs(output_path+'metrics/')
	#utils.make_dirs(output_path+'plots/silhouette/euclidean/')
	#utils.make_dirs(output_path+'plots/silhouette/cosine/')

	#needed for the report
	k = np.unique(pred).shape[0]

	
	#get metrics
	results = clustering_metrics(x,y,pred)
	
	#get and plot silhouette scores
	sil_scores_euc = silhouette_scores(x,pred,metric='euclidean', output_path = output_path+'plots/silhouette/euclidean/')	
	sil_scores_cos = silhouette_scores(x,pred,metric='cosine', output_path = output_path+'plots/silhouette/cosine/')	

	results['silhouette_scores_euc'] = sil_scores_euc
	results['silhouette_scores_cos'] = sil_scores_cos	
	results['k'] = [min_k,max_k]
	results['n_clusters_gt'] = np.unique(y).shape[0]

	utils.save_report(results, output_path+'metrics/')
 
	#save predictions
	utils.save_predictions(pred, output_path)


def report(x,y,pred):
	silhouette_euc = me.silhouette_score(x,pred,metric='euclidean')
	silhouette_cos = me.silhouette_score(x,pred,metric='cosine')

	homogeneity = me.homogeneity_score(y,pred)
	completeness = me.completeness_score(y,pred)
	vmeasure = me.v_measure_score(y,pred)

	adjusted_rand_score = me.adjusted_rand_score(y,pred)
	adjusted_mutual_info_score = me.adjusted_mutual_info_score(y,pred)


	print ('silhouette_euc',silhouette_euc)
	print ('silhouette_cos',silhouette_cos)
	print ('homogeneity',homogeneity )
	print ('completeness',completeness) 
	print ('vmeasure',vmeasure )
	print ('adjusted rand score',adjusted_rand_score)
	print ('adjusted mutual info score',adjusted_mutual_info_score)

	return [silhouette_euc,silhouette_cos,homogeneity,completeness,vmeasure,adjusted_rand_score,adjusted_mutual_info_score]
'''
