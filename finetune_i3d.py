# -*- coding: utf-8 -*-
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras
import numpy as np
from data_generator_i3d import DataGenerator, generate_train_val_splits
from i3d_inception import Inception_Inflated3d
#from keras.utils import multi_gpu_model
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Lambda, Activation, Conv3D, Dropout, Input, concatenate
import keras.backend as K
from keras.models import Model, Sequential
from keras.callbacks import LearningRateScheduler, EarlyStopping
from math import ceil
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plot_path = 'plots_test_compression/'
from sklearn.manifold import TSNE
np.random.seed(2020)
try:
	os.makedirs(plot_path)
except:
	print('deu certo')

def plot_embeddings(x,y,path):

	data = []
	labels = []
	for i in np.unique(y):
		idx_label = np.where(i==y)[0]
		idx_label = idx_label[:100]
		for a in idx_label:
			data.append(x[a])
			labels.append(y[a])


	data = np.array(data)
	labels = np.array(labels)


	print ('plotting...')
	print ('training tsne')
	tsne = TSNE(n_components=2, metric = 'cosine')
	x = tsne.fit_transform(data,labels)

	fig = plt.figure(1)

	plt.scatter(data[:,0],data[:,1], c=labels, s=2)
	plt.savefig(path)
	plt.close()

def schedule(epoch):
    if epoch < 2:
        return .01
    elif epoch < 5:
        return .001
    elif epoch < 10:
        return .0001
    else:
        return .00001

def build_model(params, NUM_CLASSES):
        input_shape = (params['num_frames'],params['frame_height'],params['frame_width'],3)
        print ('loading kinetics i3d rgb model...')
        kinetics_model = Inception_Inflated3d(
                include_top=False,
                weights='rgb_imagenet_and_kinetics',
                input_shape=input_shape,
                classes=None)

        model = Sequential()
        model.add(kinetics_model)
        model.add(Dropout(0.5))
        model.add(Conv3D(
                NUM_CLASSES, (1, 1, 1),
                strides=(1, 1, 1),
                padding='same',
                use_bias=True,
                name='1x1_conv'))

        num_frames_remaining = kinetics_model.get_layer('global_avg_pool').output_shape[1]
        model.add(Reshape((num_frames_remaining, -1)))

        # logits (raw scores for each class)
        model.add(Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                           output_shape=lambda s: (s[0], s[2]),name='lambda'))

        model.add(Activation('softmax', name='prediction'))
        model.summary()

        return model

def build_model_test(params, NUM_CLASSES):
        input_shape = (params['num_frames_test'],params['frame_height'],params['frame_width'],3)
        print ('loading kinetics i3d rgb model for test phase...')
        kinetics_model = Inception_Inflated3d(
                include_top=False,
                weights='rgb_imagenet_and_kinetics',
                input_shape=input_shape,
                classes=None)

        model = Sequential()
        model.add(kinetics_model)
        model.add(Dropout(0.5))
        model.add(Conv3D(
                NUM_CLASSES, (1, 1, 1),
                strides=(1, 1, 1),
                padding='same',
                use_bias=True,
                name='1x1_conv'))

        return model




def finetune(x_train, y_train, class_indexes, params):
        x_train, y_train, x_val, y_val = generate_train_val_splits(x_train, y_train, params)

        NUM_FRAMES = params['num_frames']
        NUM_RGB_CHANNELS = 3
        NUM_CLASSES = np.unique(y_train).shape[0]
        
        train_params = {'batch_size': params['batch_size'],
                  'n_classes': NUM_CLASSES,
                  'flow': False,
                  'shuffle': True,
                  'is_training': True,
                  'class_dict' : class_indexes,
                  'num_frames' : NUM_FRAMES}

        val_params = {'batch_size': params['batch_size'],
                  'n_classes': NUM_CLASSES,
                  'flow': False,
                  'shuffle': False,
                  'is_training': False,
                  'class_dict' : class_indexes,
                  'num_frames' : NUM_FRAMES}

        sgd = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
        model = build_model(params, NUM_CLASSES = NUM_CLASSES)        

        es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, baseline = 0.95, patience=0)
        lr_scheduler = LearningRateScheduler(schedule)
        #parallel_model = multi_gpu_model(model, gpus=2)
        #parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        generator = DataGenerator(x_train, **train_params)
        val_generator = DataGenerator(x_val, **val_params)
        steps_per_epoch = len(x_train)/params['batch_size']
        validation_steps = len(x_val)/params['batch_size']
        #steps_per_epoch = 1
        #validation_steps = 1
        history = model.fit_generator(generator = generator,epochs = params['max_epochs'], steps_per_epoch = steps_per_epoch , validation_data = val_generator, validation_steps = validation_steps,  use_multiprocessing = False, workers = 12, callbacks=[lr_scheduler, es]) #Setting multiprocessing to True causes problems with the current implementation.

        return model, history, model.get_weights()


def initialize_i3d_kinetics(params):
        input_shape = (params['num_frames_test'],params['frame_height'],params['frame_width'],3)

        print ('loading kinetics i3d rgb model...')
        kinetics_model = Inception_Inflated3d(
                include_top=False,
                weights='rgb_imagenet_and_kinetics',
                input_shape=input_shape,
                classes=None)

        model = Sequential()
        model.add(kinetics_model)

        num_frames_remaining = kinetics_model.get_layer('global_avg_pool').output_shape[1]
        model.add(Reshape((num_frames_remaining, -1)))

        # logits (raw scores for each class)
        model.add(Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                           output_shape=lambda s: (s[0], s[2]),name='lambda'))
        return model

def extract_features(old_model, x_train, y_train, x_test, y_test, class_indexes, params):
        NUM_CLASSES_TRAIN = np.unique(y_train).shape[0]
        NUM_CLASSES_TEST = np.unique(y_test).shape[0]
        if params['model'] == 'ucf101':
                model = build_model_test(params, NUM_CLASSES_TRAIN)
                model.set_weights(old_model)

                intermediate_model = Sequential()
                intermediate_model.add(Model(inputs=model.input, outputs=model.get_layer('1x1_conv').input)) 
                num_frames_remaining = model.get_layer('1x1_conv').input_shape[1]
                intermediate_model.add(Reshape((num_frames_remaining, -1)))
                # logits (raw scores for each class)
                intermediate_model.add(Lambda(lambda x: K.mean(x, axis=1, keepdims=False), output_shape=lambda s: (s[0], s[2]),name='lambda'))
                model = intermediate_model

                model.summary()
        elif params['model'] == 'kinetics':
                model = old_model

        batch = 1
        gen_params = {'batch_size': batch,
                  'n_classes': NUM_CLASSES_TEST,
                  'flow': False,
                  'shuffle': False,
                  'is_training': False,
                  'class_dict' : class_indexes,
                  'num_frames' : params['num_frames_test']}


        train_generator = DataGenerator(x_train, **gen_params)
        test_generator = DataGenerator(x_test, **gen_params)

        print (len(x_test))
        print (len(x_train))
        x_test_features = model.predict_generator(test_generator, steps=(len(x_test)/batch), workers=8, use_multiprocessing=False, verbose=1)
        x_train_features = model.predict_generator(train_generator, steps=(len(x_train)/batch), workers=8, use_multiprocessing=False, verbose=1)

        print (x_train_features.shape, len(x_train))
        print (x_test_features.shape, len(x_test))

        return np.squeeze(x_train_features), np.squeeze(x_test_features)



def extract_features_single(old_model, x_train, y_train, class_indexes, params, original_n_classes):
        NUM_CLASSES_TRAIN = np.unique(y_train).shape[0]
        if params['model'] == 'ucf101':
                model = build_model_test(params, original_n_classes)
                model.set_weights(old_model)

                intermediate_model = Sequential()
                intermediate_model.add(Model(inputs=model.input, outputs=model.get_layer('1x1_conv').input)) 
                num_frames_remaining = model.get_layer('1x1_conv').input_shape[1]
                intermediate_model.add(Reshape((num_frames_remaining, -1)))
                # logits (raw scores for each class)
                intermediate_model.add(Lambda(lambda x: K.mean(x, axis=1, keepdims=False), output_shape=lambda s: (s[0], s[2]),name='lambda'))
                model = intermediate_model

                model.summary()
        elif params['model'] == 'kinetics':
                model = old_model

        batch = 1
        gen_params = {'batch_size': batch,
                  'n_classes': NUM_CLASSES_TRAIN,
                  'flow': False,
                  'shuffle': False,
                  'is_training': False,
                  'class_dict' : class_indexes,
                  'num_frames' : params['num_frames_test']}


        train_generator = DataGenerator(x_train, **gen_params)

        print (len(x_train))
        x_train_features = model.predict_generator(train_generator, steps=(len(x_train)/batch), workers=8, use_multiprocessing=False, verbose=1)

        print (x_train_features.shape, len(x_train))

        return np.squeeze(x_train_features)


def init_ti3d(params):
        from keras.layers import Input, Conv2D, Lambda, Dense, Flatten, concatenate, Dropout
        from keras.models import Model, Sequential
        from keras.regularizers import l2
        from keras import backend as K
        from keras.optimizers import SGD,Adam
        #import data_generator_openset as gen
        #import utils
        np.random.seed(2020)

        def cos_distance(y_true, y_pred, vects_are_normalized=False):
                x, y = y_true, y_pred
                if not vects_are_normalized:
                        x = K.l2_normalize(x, axis=-1)
                        y = K.l2_normalize(y, axis=-1)
                similarity = K.batch_dot(x, y, axes=1)
                distance = K.constant(1) - similarity
                return K.squeeze(distance, axis=-1)

        def triplet_loss_wrapper(alpha = 0.2):
                def triplet_loss_inner(y_true, y_pred):
                        total_lenght = y_pred.shape.as_list()[-1]
                        anchor = y_pred[:,0:int(total_lenght*1/3)]
                        positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
                        negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

                        pos_dist = cos_distance(anchor, positive)
                        neg_dist = cos_distance(anchor, negative)

                        basic_loss = pos_dist-neg_dist+alpha
                        loss = K.maximum(basic_loss,0.0)
                        return loss
                return triplet_loss_inner

        def create_base_network():
                np.random.seed(2020)
                model = Sequential()
                model.add(Dense(512,name='input', input_shape = (1024,), kernel_initializer='glorot_uniform'))
                #model.add(Dropout(0.1,name='noise'))
                model.add(Dense(256, kernel_initializer='glorot_uniform'))
                model.summary()
                return model

        sgd_optim = SGD(lr=params['triplet_lr'])
        anchor_input = Input((1024,), name='anchor_input')
        positive_input = Input((1024, ), name='positive_input')
        negative_input = Input((1024, ), name='negative_input')

        # Shared embedding layer for positive and negative items
        Shared_DNN = create_base_network()

        encoded_anchor = Shared_DNN(anchor_input)
        encoded_positive = Shared_DNN(positive_input)
        encoded_negative = Shared_DNN(negative_input)

        merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')

        model = Model(inputs=[anchor_input,positive_input, negative_input], outputs=merged_vector)
        model.compile(loss=triplet_loss_wrapper(params['margin']), optimizer=sgd_optim)

        #K.clear_session()
        return model


        


def finetune_triplet_net(x_train, int_y_train, x_test, params, warm_start_model = None):
        from keras.layers import Input, Conv2D, Lambda, Dense, Flatten, concatenate, Dropout
        from keras.models import Model, Sequential
        from keras.regularizers import l2
        from keras import backend as K
        from keras.optimizers import SGD,Adam
        #import data_generator_openset as gen
        #import utils
        np.random.seed(2020)




        def cos_distance(y_true, y_pred, vects_are_normalized=False):
                x, y = y_true, y_pred
                if not vects_are_normalized:
                        x = K.l2_normalize(x, axis=-1)
                        y = K.l2_normalize(y, axis=-1)
                similarity = K.batch_dot(x, y, axes=1)
                distance = K.constant(1) - similarity
                return K.squeeze(distance, axis=-1)

        def triplet_loss_wrapper(alpha = 0.2):
                def triplet_loss_inner(y_true, y_pred):
                        total_lenght = y_pred.shape.as_list()[-1]
                        anchor = y_pred[:,0:int(total_lenght*1/3)]
                        positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
                        negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

                        pos_dist = cos_distance(anchor, positive)
                        neg_dist = cos_distance(anchor, negative)

                        basic_loss = pos_dist-neg_dist+alpha
                        loss = K.maximum(basic_loss,0.0)
                        return loss
                return triplet_loss_inner

        def create_base_network():
                np.random.seed(2020)
                model = Sequential()
                model.add(Dense(512,name='input', input_shape = (1024,), kernel_initializer='glorot_uniform'))
                #model.add(Dropout(0.1,name='noise'))
                model.add(Dense(256, kernel_initializer='glorot_uniform'))
                model.summary()
                return model



        def generate_semihard_triplets(x, y, embeddings, margin = 0.2, max_anchors_class = 2000, max_negatives = 500, use_hard_triplets = True, limit_triplets_class = 50000):
                np.random.seed(2020)
                if use_hard_triplets:
                        print ('generating hard and semihard triplets...')
                else:
                        print ('generating semihard triplets...')

                from sklearn.metrics.pairwise import cosine_distances
                import time
                x = np.array(x)
                y = np.array(y)
                embeddings = np.array(embeddings)

                unique_y = np.unique(y)
                train_triplets = []
                print('unique classes:',unique_y)
                for un in unique_y:
                        #print ('generating triplets for class', un)
                        current_class_idx = np.where(y == un)[0]
                        current_class_data = x[current_class_idx]
                        max_anchors_class = int(current_class_data.shape[0]/3)
                        #input(current_class_data)
                        current_class_embeddings = embeddings[current_class_idx]
                        negative_idx = np.where(y != un)[0]
                        negatives = x[negative_idx]
                        negatives_embeddings = embeddings[negative_idx]
                        #select max negatives
                        perm_neg = np.random.permutation(negatives.shape[0])[:max_negatives]
                        negatives = negatives[perm_neg]                

                        class_semihard_triplets = []
                        class_hard_triplets = []
                        #select max_anchor_class anchors
                        perm = np.random.permutation(current_class_data.shape[0])[:max_anchors_class]
                        anchors = current_class_data[perm].copy()

                        anchors_embeddings = current_class_embeddings[perm].copy()

                        positive_idx = list(set(range(current_class_data.shape[0])) ^ set(perm))
                        positives = current_class_data[positive_idx].copy()
                        positives_embeddings = current_class_embeddings[positive_idx].copy()
                        
                        print(len(anchors_embeddings), len(positives_embeddings))
                        if(len(anchors_embeddings) == 0 or len(positives_embeddings) == 0):
                                print('Could not form anchor positive pairs for class',un)
                                #input('?')
                                continue
                                
                        anchor_positive_distances = cosine_distances(anchors_embeddings, positives_embeddings)
                        anchor_negative_distances = cosine_distances(anchors_embeddings, negatives_embeddings)

                        semihards = 0
                        hards = 0
                        counter = 0
                        for a in range(anchors.shape[0]):
                                for p in range(positives.shape[0]):
                                        for n in range(negatives.shape[0]):
                                                if counter >= limit_triplets_class:
                                                        #print ('triplet limit achieved for this class. breaking...')
                                                        break
                                                ap_d = anchor_positive_distances[a,p]
                                                an_d = anchor_negative_distances[a,n]
                                                diff = np.abs(ap_d - an_d)
                                                if(ap_d < an_d and diff < margin):
                                                        semihards += 1
                                                        counter += 1
                                                        class_semihard_triplets.append([anchors[a],positives[p],negatives[n]])
                                                if use_hard_triplets:
                                                        if(ap_d > an_d):
                                                                hards += 1
                                                                counter += 1
                                                                class_hard_triplets.append([anchors[a],positives[p],negatives[n]])
                                                
                        print ('semihards, hards', semihards, hards)
                        class_semihard_triplets = np.array(class_semihard_triplets)
                        if class_semihard_triplets.shape[0] > 0:
                                train_triplets.append(class_semihard_triplets)
                        if use_hard_triplets:
                                class_hard_triplets = np.array(class_hard_triplets)  
                                if class_hard_triplets.shape[0] > 0:
                                        train_triplets.append(class_hard_triplets)

                if len(train_triplets) == 0:
                        print ('could not generate any more triplets that meet the requirements.')
                        return None
                train_triplets = np.vstack(train_triplets)
                print ('total triplets:',train_triplets.shape)
                return train_triplets


	#FULL_LIST = gen.get_video_list(params['list_id'])
        #all_categories = gen.get_all_categories(FULL_LIST) 
        #dict_map = utils.map_labels(all_categories)

        #int_y_train = utils.convert_labels_to_int(y_train, dict_map)
        #int_y_test = utils.convert_labels_to_int(y_test, dict_map)

        print(type(params), params)
        sgd_optim = SGD(lr=params['triplet_lr'])


        
        anchor_input = Input((1024,), name='anchor_input')
        positive_input = Input((1024, ), name='positive_input')
        negative_input = Input((1024, ), name='negative_input')

        # Shared embedding layer for positive and negative items
        Shared_DNN = create_base_network()

        encoded_anchor = Shared_DNN(anchor_input)
        encoded_positive = Shared_DNN(positive_input)
        encoded_negative = Shared_DNN(negative_input)

        merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')

        model = Model(inputs=[anchor_input,positive_input, negative_input], outputs=merged_vector)
        model.compile(loss=triplet_loss_wrapper(params['margin']), optimizer=sgd_optim)

        if warm_start_model is not None:
                model.set_weights(warm_start_model)

        model.summary()

        epochs = params['triplet_epochs']

        hist_triplet = []
        for i in range(epochs):
                trained_model = Model(inputs=anchor_input, outputs=encoded_anchor)
                print ("getting train embeddings...")
                embeddings = trained_model.predict(x_train)
                #plot_embeddings(embeddings, int_y_train, plot_path+'train_'+str(i)+'.jpg')
                X_train = generate_semihard_triplets(x_train, int_y_train, embeddings, margin = params['margin'], max_anchors_class = 50, max_negatives = 150, use_hard_triplets = True)
                if X_train is None:
                        print ('early stopping...')
                        break
                Anchor = X_train[:,0,:]
                Positive = X_train[:,1,:]
                Negative = X_train[:,2,:]
                Y_dummy = np.empty((Anchor.shape[0],2))
                hist = model.fit([Anchor,Positive,Negative],y=Y_dummy, batch_size=params['triplet_batch_size'], epochs=1, shuffle = True)
                hist_triplet.append(hist.history)

        trained_model = Model(inputs=anchor_input, outputs=encoded_anchor)

        X_train_trm = trained_model.predict(x_train)
        X_test_trm = trained_model.predict(x_test)

        #K.clear_session() # this is very important
        return X_train_trm, X_test_trm, hist_triplet, model, model.get_weights()



def extract_features_triplet_net(x_train, int_y_train, params, warm_start_model = None):
        from keras.layers import Input, Conv2D, Lambda, Dense, Flatten, concatenate, Dropout
        from keras.models import Model, Sequential
        from keras.regularizers import l2
        from keras import backend as K
        from keras.optimizers import SGD,Adam
        #import data_generator_openset as gen
        #import utils
        np.random.seed(2020)




        def cos_distance(y_true, y_pred, vects_are_normalized=False):
                x, y = y_true, y_pred
                if not vects_are_normalized:
                        x = K.l2_normalize(x, axis=-1)
                        y = K.l2_normalize(y, axis=-1)
                similarity = K.batch_dot(x, y, axes=1)
                distance = K.constant(1) - similarity
                return K.squeeze(distance, axis=-1)

        def triplet_loss_wrapper(alpha = 0.2):
                def triplet_loss_inner(y_true, y_pred):
                        total_lenght = y_pred.shape.as_list()[-1]
                        anchor = y_pred[:,0:int(total_lenght*1/3)]
                        positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
                        negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

                        pos_dist = cos_distance(anchor, positive)
                        neg_dist = cos_distance(anchor, negative)

                        basic_loss = pos_dist-neg_dist+alpha
                        loss = K.maximum(basic_loss,0.0)
                        return loss
                return triplet_loss_inner

        def create_base_network():
                np.random.seed(2020)
                model = Sequential()
                model.add(Dense(512,name='input', input_shape = (1024,), kernel_initializer='glorot_uniform'))
                #model.add(Dropout(0.1,name='noise'))
                model.add(Dense(256, kernel_initializer='glorot_uniform'))
                model.summary()
                return model



        def generate_semihard_triplets(x, y, embeddings, margin = 0.2, max_anchors_class = 2000, max_negatives = 500, use_hard_triplets = True, limit_triplets_class = 50000):
                np.random.seed(2020)
                if use_hard_triplets:
                        print ('generating hard and semihard triplets...')
                else:
                        print ('generating semihard triplets...')

                from sklearn.metrics.pairwise import cosine_distances
                import time
                x = np.array(x)
                y = np.array(y)
                embeddings = np.array(embeddings)

                unique_y = np.unique(y)
                train_triplets = []
                for un in unique_y:
                        #print ('generating triplets for class', un)
                        current_class_idx = np.where(y == un)[0]
                        current_class_data = x[current_class_idx]
                        max_anchors_class = int(current_class_data.shape[0]/3)
                        #input(current_class_data)
                        current_class_embeddings = embeddings[current_class_idx]
                        negative_idx = np.where(y != un)[0]
                        negatives = x[negative_idx]
                        negatives_embeddings = embeddings[negative_idx]
                        #select max negatives
                        perm_neg = np.random.permutation(negatives.shape[0])[:max_negatives]
                        negatives = negatives[perm_neg]                

                        class_semihard_triplets = []
                        class_hard_triplets = []
                        #select max_anchor_class anchors
                        perm = np.random.permutation(current_class_data.shape[0])[:max_anchors_class]
                        anchors = current_class_data[perm].copy()

                        anchors_embeddings = current_class_embeddings[perm].copy()

                        positive_idx = list(set(range(current_class_data.shape[0])) ^ set(perm))
                        positives = current_class_data[positive_idx].copy()
                        positives_embeddings = current_class_embeddings[positive_idx].copy()

                        
                        anchor_positive_distances = cosine_distances(anchors_embeddings, positives_embeddings)
                        anchor_negative_distances = cosine_distances(anchors_embeddings, negatives_embeddings)

                        semihards = 0
                        hards = 0
                        counter = 0
                        for a in range(anchors.shape[0]):
                                for p in range(positives.shape[0]):
                                        for n in range(negatives.shape[0]):
                                                if counter >= limit_triplets_class:
                                                        #print ('triplet limit achieved for this class. breaking...')
                                                        break
                                                ap_d = anchor_positive_distances[a,p]
                                                an_d = anchor_negative_distances[a,n]
                                                diff = np.abs(ap_d - an_d)
                                                if(ap_d < an_d and diff < margin):
                                                        semihards += 1
                                                        counter += 1
                                                        class_semihard_triplets.append([anchors[a],positives[p],negatives[n]])
                                                if use_hard_triplets:
                                                        if(ap_d > an_d):
                                                                hards += 1
                                                                counter += 1
                                                                class_hard_triplets.append([anchors[a],positives[p],negatives[n]])
                                                
                        print ('semihards, hards', semihards, hards)
                        class_semihard_triplets = np.array(class_semihard_triplets)
                        if class_semihard_triplets.shape[0] > 0:
                                train_triplets.append(class_semihard_triplets)
                        if use_hard_triplets:
                                class_hard_triplets = np.array(class_hard_triplets)  
                                if class_hard_triplets.shape[0] > 0:
                                        train_triplets.append(class_hard_triplets)

                if len(train_triplets) == 0:
                        print ('could not generate any more triplets that meet the requirements.')
                        return None
                train_triplets = np.vstack(train_triplets)
                #print ('total triplets:',train_triplets.shape)
                return train_triplets


	#FULL_LIST = gen.get_video_list(params['list_id'])
        #all_categories = gen.get_all_categories(FULL_LIST) 
        #dict_map = utils.map_labels(all_categories)

        #int_y_train = utils.convert_labels_to_int(y_train, dict_map)
        #int_y_test = utils.convert_labels_to_int(y_test, dict_map)

        print(type(params), params)
        sgd_optim = SGD(lr=params['triplet_lr'])

        #K.clear_session() # this is very important
        
        anchor_input = Input((1024,), name='anchor_input')
        positive_input = Input((1024, ), name='positive_input')
        negative_input = Input((1024, ), name='negative_input')

        # Shared embedding layer for positive and negative items
        Shared_DNN = create_base_network()

        encoded_anchor = Shared_DNN(anchor_input)
        encoded_positive = Shared_DNN(positive_input)
        encoded_negative = Shared_DNN(negative_input)

        merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')

        model = Model(inputs=[anchor_input,positive_input, negative_input], outputs=merged_vector)
        model.compile(loss=triplet_loss_wrapper(params['margin']), optimizer=sgd_optim)

        if warm_start_model is not None:
                #model.set_weights(warm_start_model.get_weights())  
                model.set_weights(warm_start_model)
              


       
        model.summary()

        
        trained_model = Model(inputs=anchor_input, outputs=encoded_anchor)

        X_train_trm = trained_model.predict(x_train)

        K.clear_session() # this is very important
        return X_train_trm


def old_finetune_triplet_net(x_train, y_train, x_test, y_test, params):

        import numpy as np
        import random
        from keras.regularizers import l2
        from keras import backend as K
        from keras.optimizers import SGD,Adam
        from keras.losses import binary_crossentropy
        from itertools import permutations
        import sklearn
        import triplet_net as triplet

        print (x_train.shape)
        print( y_train)
        X_train, x_val = triplet.generate_triplet(x_train,y_train, testsize=0.2, ap_pairs=30, an_pairs=30)
        #x_val, _ = generate_triplet(x_test_flat_known,y_test_known, ap_pairs=100, an_pairs=100,testsize=0)
        #adam_optim = Adam()
        sgd_optim = SGD(lr=0.0001)

        anchor_input = Input((1024,), name='anchor_input')
        positive_input = Input((1024, ), name='positive_input')
        negative_input = Input((1024, ), name='negative_input')

        # Shared embedding layer for positive and negative items
        Shared_DNN = triplet.create_base_network()

        encoded_anchor = Shared_DNN(anchor_input)
        encoded_positive = Shared_DNN(positive_input)
        encoded_negative = Shared_DNN(negative_input)

        merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')

        model = Model(inputs=[anchor_input,positive_input, negative_input], outputs=merged_vector)
        model.compile(loss=triplet.triplet_loss, optimizer=sgd_optim)

        model.summary()

        Anchor = X_train[:,0,:]
        Positive = X_train[:,1,:]
        Negative = X_train[:,2,:]
        Anchor_test = x_val[:,0,:]
        Positive_test = x_val[:,1,:]
        Negative_test = x_val[:,2,:]

        Y_dummy = np.empty((Anchor.shape[0],300))
        Y_dummy2 = np.empty((Anchor_test.shape[0],1))

        hist = model.fit([Anchor,Positive,Negative],y=Y_dummy,validation_data=([Anchor_test,Positive_test,Negative_test],Y_dummy2), batch_size=512, epochs=params['triplet_epochs'])

        trained_model = Model(inputs=anchor_input, outputs=encoded_anchor)

        #trained_model.load_weights('triplet_model_MNIST.hdf5')


        X_train_trm = trained_model.predict(x_train)
        X_test_trm = trained_model.predict(x_test)
        return X_train_trm, X_test_trm, hist

