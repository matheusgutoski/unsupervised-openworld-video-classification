import argparse
import utils
import pickle
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_path', help='Where to read evms', type=str, 
                default = 'prototype_results/ucf101/train_2__test_3__tail_10__cover_0.1__seed_5/0/phase_4_incremental_ti3d_incremental_evm_tail_0.2_online/9/')
	
	args = parser.parse_args()
        
	evms = pickle.load(open(args.input_path + 'evms.pickle', 'rb'))


	evs_per_class = {}
	total_evs = 0
	for label,evm in evms.items():
		evs_per_class[label] = len(evm._extreme_vectors)
		total_evs += len(evm._extreme_vectors)

	

	output_path = args.input_path+'ev_count.csv'

	print(args.input_path)

	f = open(output_path, 'w')
	
	f.write('class,number of evs\n')
	print('class,number of evs')

	for k,v in evs_per_class.items():
		f.write(str(k)+','+str(v)+'\n')
		print(str(k)+','+str(v))

	f.write('total,'+str(total_evs))
	print('total,'+str(total_evs))

	f.close()
