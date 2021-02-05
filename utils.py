import os
import numpy as np


def train_test_split_groups(filenames, initial_classes, params):
	videos = [x.split('.')[0] for x in filenames if x.split('/')[0] in initial_classes]
	print(videos)

	merged_list = [] #this list is for merging groups of the same class so that we can use sklearns train_test_split method. Each element in the list is a group of videos that belong to the same class and group
	for cl in initial_classes:
		videos_current_class = sorted([x for x in videos if x.split('/')[0] == cl])
		groups_current_class = np.unique([x.split('_')[2] for x in videos_current_class])
		for g in groups_current_class:
			videos_group_current_class = sorted([x for x in videos_current_class if x.split('_')[2] == g])
			merged_list.append(videos_group_current_class)
                   
	merged_list_labels = [x[0].split('/')[0] for x in merged_list]




	# employ sklearns train test split method to generate train/test knowns
	from sklearn.model_selection import train_test_split

	merged_knowns_train, merged_knowns_test, merged_knowns_train_labels , merged_knowns_test_labels = train_test_split(merged_list, merged_list_labels, train_size = 0.7, random_state = params['seed'], shuffle = True, stratify = merged_list_labels)

 	#unmerge lists
	train = []
	train_labels = []
	for m, l in zip(merged_knowns_train, merged_knowns_train_labels):
		for n in m:
			train.append(n)
			train_labels.append(l)
                
	test = []
	test_labels = []
	for m, l in zip(merged_knowns_test, merged_knowns_test_labels):
		for n in m:
			test.append(n)
			test_labels.append(l)

	print(initial_classes, merged_list)

	
	
	return train, train_labels, test, test_labels

def map_labels(classes):
        dict_map = {}
        for i,c in enumerate(sorted(classes)):
                dict_map[c] = i+1
        for key, value in sorted(dict_map.items()):
                print (key, value)
        return dict_map

def convert_labels_to_int(labels, dict_map):
        new_labels = []
        for l in labels:
                new_labels.append(dict_map[l])
        return new_labels

def openness(training_classes, testing_classes, target_classes):
        from math import sqrt
        return 1 - (sqrt((2.*training_classes) / (testing_classes + target_classes)))

def openness_Geng(training_classes, testing_classes):
        from math import sqrt
        return 1 - (sqrt((2.*training_classes) / (testing_classes + training_classes)))

def gen_exp_id(params):
        exp_id = params['model'] + '/'
        exp_id += 'train_' + str(params['n_train_classes']) + '__test_' + str(params['n_test_classes']) + '__'
        exp_id += 'tail_' + str(params['tail_size']) + '__cover_' + str(params['cover_threshold']) + '__'
        exp_id += 'seed_' + str(params['init_seed']) + '/'

        return exp_id

def makedirs(path):
        try:
                os.makedirs(path)
        except Exception as e:
                print(e)

def generate_report(youdens_index, closed_f1_score, classif_rep, cm, params):
        output_path = params['output_path']
        exp_id = gen_exp_id(params)

        output_path += exp_id
        output_path += str(params['fold']) + '/'
        output_path += str(params['model_type']) + '/'
        output_path += str(params['classification_threshold']) + '/'
        makedirs(output_path)

        outfile = open(output_path+'classification_report.txt', 'w')

        outfile.write('problem openness: '+str(params['openness'])+'\n')
        outfile.write(str(classif_rep)+'\n')
        outfile.write('\n' + str(cm) +'\n' )
        outfile.write('\nopen youdens index: '+str(youdens_index) + '\n')
        outfile.write('closed f1 score:' + str(closed_f1_score) + '\n')

        outfile.close()

def save_i3d_model(model, params):
        if params['model'] != 'kinetics':
                print ('saving i3d model...')
                output_path = params['output_path']
                exp_id = gen_exp_id(params)

                output_path += exp_id
                output_path += str(params['fold']) + '/'
                makedirs(output_path)
                model.save(output_path + 'i3d_model.h5')                
        else:
                print ('no need to save kinetics model')

def save_ti3d_model(model, params):
        if params['model'] != 'kinetics':
                print ('saving ti3d model...')
                output_path = params['output_path']
                exp_id = gen_exp_id(params)

                output_path += exp_id
                output_path += str(params['fold']) + '/'
                makedirs(output_path)
                model.save_weights(output_path + 'ti3d_model.h5')                
        else:
                print ('no need to save kinetics model')


def load_i3d_model(params):
        if params['model'] != 'kinetics':
                print ('loading  i3d model...')
                output_path = params['output_path']
                exp_id = gen_exp_id(params)

                output_path += exp_id
                output_path += str(params['fold']) + '/'
                from keras.models import load_model
                return load_model(output_path + 'i3d_model.h5')                




def load_ti3d_model(params):

        print ('loading  i3d model...')
        output_path = params['output_path']
        exp_id = gen_exp_id(params)

        output_path += exp_id
        output_path += str(params['fold']) + '/'

        import finetune_i3d
        model = finetune_i3d.init_ti3d(params)

        weights = model.load_weights(output_path + 'ti3d_model.h5') 

        return model

def save_evm_models(evms,params):
 
        def plot_pdf_weibull(evm_class, i, scale, shape, pdf_plot_filename):
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                def weib(x,n,a):
                        return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)

                x = np.linspace(0.9, 1.1, 1000)
                plt.figure(0)
                plt.ylim(0,250)
                plt.plot(x, weib(x, scale, shape))
                plt.savefig(pdf_plot_filename + str(evm_class) + '_' + str(i)+'.png')
                plt.close()

        def save_weibulls_parameters(evms, params, output_path):
                
                pdf_plot_filename = output_path + 'weibull_plots/'
                makedirs(pdf_plot_filename)
                output_filename = 'weibull_parameters.csv'

                all_parameters = []
                for evm_class, evm_model in evms.items():
                        parameters_this_class = []       
                        for psi in evm_model.margin_weibulls:
                                #scale,shape,sign,translate_amount,small_score
                                psi_parameters = psi.get_params()
                                parameters_this_class.append(psi_parameters)
                        all_parameters.append(parameters_this_class)

                output_csv = open(output_path + output_filename, 'w')

                header = 'class; psi model; scale; shape; sign; translate_amount; small_score'
                output_csv.write(header+'\n')

                for evm_class, evm_models in zip(evms, all_parameters):
                        for i, psi in enumerate(evm_models):
                                scale = np.around(psi[0], decimals = 3)
                                shape = np.around(psi[1], decimals = 3)
                                sign = psi[2]
                                translate = psi[3]
                                small = np.around(psi[4], decimals = 3)
                                plot_pdf_weibull(str(evm_class), str(i), scale, shape, pdf_plot_filename)

                                scale = str(scale).replace('.', ',')
                                shape = str(shape).replace('.', ',')
                                small = str(small).replace('.', ',')

                                output_csv.write(str(evm_class)+';'+str(i)+';'+ str(scale) +';'+ str(shape) +';'+ str(sign)+';'+str(translate)+';'+str(small)+ '\n')

                output_csv.close()

        print ('saving evms...')
        output_path = params['output_path']
        exp_id = gen_exp_id(params)

        output_path += exp_id
        output_path += str(params['fold']) + '/'
        output_path += str(params['model_type']) + '/'
        makedirs(output_path)


        import pickle
        
        dbfile = open(output_path + 'evms.pickle', 'wb')

        pickle.dump(evms, dbfile)
        #load with pickle.load(open('evms.pickle', 'rb'))

        save_weibulls_parameters(evms, params, output_path)


def save_features(x_train_features, x_test_features, y_train, y_test, open_y_test, params):
        print ('saving features...')
        output_path = params['output_path']
        exp_id = gen_exp_id(params)

        output_path += exp_id
        output_path += str(params['fold']) + '/'
        output_path += str(params['model_type']) + '/'
        makedirs(output_path)

        np.save(output_path + 'x_train_features.npy', x_train_features)
        np.save(output_path + 'x_test_features.npy', x_test_features)
        np.save(output_path + 'y_train.npy', y_train)
        np.save(output_path + 'y_test.npy', y_test)
        np.save(output_path + 'open_y_test.npy', open_y_test)

def save_ti3d_features(x_train_features, x_test_features, y_train, y_test, open_y_test, params):
        print ('saving features...')
        output_path = params['output_path']
        exp_id = gen_exp_id(params)

        output_path += exp_id
        output_path += str(params['fold']) + '/'
        output_path += str(params['model_type']) + '/'
        makedirs(output_path)

        np.save(output_path + 'x_train_features_ti3d.npy', x_train_features)
        np.save(output_path + 'x_test_features_ti3d.npy', x_test_features)
        np.save(output_path + 'y_train.npy', y_train)
        np.save(output_path + 'y_test.npy', y_test)
        np.save(output_path + 'open_y_test.npy', open_y_test)


def save_hist(hist, params):
        if hist is not None:
                print ('saving histories...')
                output_path = params['output_path']
                exp_id = gen_exp_id(params)

                output_path += exp_id
                output_path += str(params['fold']) + '/'
                output_path += str(params['model_type']) + '/'
                makedirs(output_path)

                import pickle
                
                dbfile = open(output_path + 'history.pickle', 'wb')

                pickle.dump(hist.history, dbfile)

def save_hist_triplet(hist, params):
        if hist is not None:
                print ('saving histories...')
                output_path = params['output_path']
                exp_id = gen_exp_id(params)

                output_path += exp_id
                output_path += str(params['fold']) + '/'
                output_path += str(params['model_type']) + '/'
                makedirs(output_path)

                import pickle
                
                dbfile = open(output_path + 'history.pickle', 'wb')

                pickle.dump(hist, dbfile)
        
def load_features(params):
        print ('loading features')
        output_path = params['output_path']
        exp_id = gen_exp_id(params)
        output_path += exp_id
        output_path += str(params['fold']) + '/'
        output_path += str(params['model_type']) + '/'

        x = np.load(output_path + 'x_train_features.npy')
        y = np.load(output_path + 'x_test_features.npy')

        w = np.load(output_path + 'y_train.npy')
        z = np.load(output_path + 'y_test.npy')
        return x, y, w, z

def load_ti3d_features(params):
        print ('loading ti3d features')
        output_path = params['output_path']
        exp_id = gen_exp_id(params)
        output_path += exp_id
        output_path += str(params['fold']) + '/'
        output_path += str(params['model_type']) + '/'

        x = np.load(output_path + 'x_train_features_ti3d.npy')
        y = np.load(output_path + 'x_test_features_ti3d.npy')

        w = np.load(output_path + 'y_train.npy')
        z = np.load(output_path + 'y_test.npy')
        return x, y, w, z


def read_pickle_results_file(filename = 'results.pickle'):
        import pickle
        infile = open(filename,'rb')
        results = pickle.load(infile)
        infile.close()
        return results

