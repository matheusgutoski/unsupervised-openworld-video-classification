import cv2
import numpy as np
import os
from glob import glob
import itertools
from random import randint
from keras.utils.np_utils import to_categorical
import time
import keras
from sklearn.model_selection import train_test_split

PATH_TO_FRAMES = "/home/users/datasets/hmdb51_opticalflow/"

print("Path to images:", PATH_TO_FRAMES)


def resize_frames(frames, minsize=256):
    resized_frames = []
    frames = np.squeeze(frames)
    for frame in frames:
        if len(frame.shape) == 3:
            if frame.shape[1] >= frame.shape[0]:
                if frame.shape[0] != minsize:
                    scale = float(minsize) / float(frame.shape[0])
                    frame = np.array(
                        cv2.resize(frame, (int(frame.shape[1] * scale + 1), minsize))
                    ).astype(np.float32)
            else:
                if frame.shape[1] != minsize:
                    scale = float(minsize) / float(frame.shape[1])
                    frame = np.array(
                        cv2.resize(frame, (minsize, int(frame.shape[0] * scale + 1)))
                    ).astype(np.float32)
        resized_frames.append(frame)
    resized_frames = np.array(resized_frames)
    return resized_frames


def random_flip(frames):  # assuming 0.5 probability
    rand = randint(0, 9)
    if rand < 5:
        flipped = []
        for frame in frames:
            frame = cv2.flip(frame, 1)
            flipped.append(frame)

        flipped = np.array(flipped)
        return flipped
    else:
        return frames


def random_crop(frames, crop_size=224):
    crop_x = randint(0, int(frames.shape[1] - crop_size))
    crop_y = randint(0, int(frames.shape[2] - crop_size))
    frames = frames[:, crop_x : crop_x + crop_size, crop_y : crop_y + crop_size, :]
    return frames


def center_crop(frames, crop_size=224):
    crop_x = int((frames.shape[1] - crop_size) / 2)
    crop_y = int((frames.shape[2] - crop_size) / 2)
    frames = frames[:, crop_x : crop_x + crop_size, crop_y : crop_y + crop_size, :]
    return frames


def random_temporal_crop(filenames, min_video_length, extra_frames):
    desired_frames = min_video_length + extra_frames
    if (
        len(filenames) < desired_frames
    ):  # in this case we must loop through the video to ensure there is a sufficient amount of frames. Extra frames gives a   margin for the temporal crop.
        for a in itertools.cycle(filenames):
            filenames.append(a)
            if (
                len(filenames) >= desired_frames
            ):  # give the video extra frames for the random temporal sampling
                break
    # apply the random temporal cropping
    starting_frame = randint(0, len(filenames) - min_video_length)
    filenames = filenames[starting_frame : starting_frame + min_video_length]
    return filenames


def center_temporal_crop(filenames, min_video_length):
    desired_frames = min_video_length
    if (
        len(filenames) < desired_frames
    ):  # in this case we must loop through the video to ensure there is a sufficient amount of frames.
        for a in itertools.cycle(filenames):
            filenames.append(a)
            if len(filenames) >= desired_frames:
                break
    # apply the center temporal cropping
    starting_frame = int((len(filenames) - min_video_length) / 2)
    filenames = filenames[starting_frame : starting_frame + min_video_length]

    return filenames


def read_rgb_images(files, is_training):
    images = []

    try:
        for f in files:
            # print(f)
            current_image = cv2.imread(f)
            # print(current_image)
            current_image = cv2.cvtColor(
                current_image, cv2.COLOR_BGR2RGB
            )  # input uses RGB images
            current_image = np.expand_dims(current_image, axis=0)
            images.append(current_image)
        images = np.vstack(images)
        images = np.expand_dims(images, axis=0)

        # perform augmentation here if needed
        images = rescale(images)
        images = resize_frames(images)
        if is_training == True:
            images = random_crop(images, crop_size=224)
            images = random_flip(images)
        else:
            images = center_crop(images, crop_size=224)
        images = np.expand_dims(images, axis=0)
        return images

    except Exception as ee:
        # images = np.zeros((1,64,224,224,3))
        print("error.", files, ee)


def read_flow_images(files_m, is_training):
    images = []
    for f in files_m:
        current_image = cv2.imread(f, 1)
        current_image = np.expand_dims(current_image, axis=0)
        images.append(current_image)

    images = np.vstack(images)
    images = np.expand_dims(images, axis=0)
    images = rescale(images)
    images = resize_frames(images)
    if is_training == True:
        images = random_crop(images, crop_size=224)
        images = random_flip(images)
    else:
        images = center_crop(images, crop_size=224)
    images = np.expand_dims(images, axis=0)
    # images = np.swapaxes(images,)
    flow_x = np.expand_dims(images[:, :, :, :, 2], axis=-1)
    flow_y = np.expand_dims(images[:, :, :, :, 1], axis=-1)
    # return images[:,:,:,:,1:3]
    # print flow_x.shape, flow_y.shape
    return np.concatenate((flow_x, flow_y), axis=-1)


def read_flow_images_old(files_x, files_y, is_training):
    images_x = []
    images_y = []
    for f in files_x:
        current_image = cv2.imread(f, 0)
        current_image = np.expand_dims(current_image, axis=0)
        images_x.append(current_image)
    for f in files_y:
        current_image = cv2.imread(f, 0)
        current_image = np.expand_dims(current_image, axis=0)
        images_y.append(current_image)

    images_x = np.vstack(images_x)
    images_x = np.expand_dims(images_x, axis=0)
    images_x = np.expand_dims(images_x, axis=-1)

    images_y = np.vstack(images_y)
    images_y = np.expand_dims(images_y, axis=0)
    images_y = np.expand_dims(images_y, axis=-1)

    images = np.concatenate((images_x, images_y), axis=-1)

    # perform augmentation here
    images = rescale(images)
    images = resize_frames(images)
    if is_training == True:
        images = random_crop(images, crop_size=224)
        images = random_flip(images)
    else:
        images = center_Crop(images, crop_size=224)
    images = np.expand_dims(images, axis=0)
    # print images, images.shape
    return images


def save_batch(batch, labels, flow=False):
    save_dir = "debug_images/"
    # print batch.shape
    for b, l in zip(batch, labels):
        # print b.shape
        try:
            os.makedirs(save_dir + str(l))
        except:
            pass
        for i, _b in enumerate(b):
            if not flow:
                _b = (_b + 1) * 128.0
                # print _b.shape
                _b = _b[:, :, ::-1]
                cv2.imwrite(save_dir + str(l) + "/" + str(i) + ".jpg", _b)
            else:
                newdim = np.zeros((_b.shape[0], _b.shape[1], 1))
                newdim[...] = 128
                # print _b.shape
                _b = (_b + 1) * 128.0
                _b = np.concatenate((newdim, _b), axis=-1)
                _b = _b[:, :, ::-1]
                cv2.imwrite(save_dir + str(l) + "/" + str(i) + ".jpg", _b)


def rescale(frames):
    frames = frames / 128.0 - 1.0
    return frames


def generate_train_val_splits(
    x_train, y_train, params
):  # split known classes into training and val knowns. Split must ensure each group of subclasses remain united
    videos = [x.split(".")[0] for x in x_train]
    known_classes = np.unique([x.split("/")[0] for x in x_train])
    merged_list = (
        []
    )  # this list is for merging groups of the same class so that we can use sklearns train_test_split method. Each element in the list is a group of videos that belong to the same class and group
    for cl in known_classes:
        videos_current_class = sorted([x for x in videos if x.split("/")[0] == cl])
        groups_current_class = np.unique(
            [x.split("_")[2] for x in videos_current_class]
        )
        for g in groups_current_class:
            videos_group_current_class = sorted(
                [x for x in videos_current_class if x.split("_")[2] == g]
            )
            merged_list.append(videos_group_current_class)

    merged_list_labels = [x[0].split("/")[0] for x in merged_list]

    # employ sklearns train test split method to generate train/val knowns
    (
        merged_knowns_train,
        merged_knowns_val,
        merged_knowns_train_labels,
        merged_knowns_val_labels,
    ) = train_test_split(
        merged_list,
        merged_list_labels,
        train_size=params["train_ratio"],
        random_state=params["seed"],
        shuffle=True,
        stratify=merged_list_labels,
    )

    # unmerge lists
    train = []
    train_labels = []
    for m, l in zip(merged_knowns_train, merged_knowns_train_labels):
        for n in m:
            train.append(n)
            train_labels.append(l)
    val = []
    val_labels = []
    for m, l in zip(merged_knowns_val, merged_knowns_val_labels):
        for n in m:
            val.append(n)
            val_labels.append(l)

    return train, train_labels, val, val_labels


def get_label(folder, class_dict):
    folder = folder.split("/")[0]
    return class_dict.get(folder)


class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        list_IDs,
        class_dict,
        batch_size=6,
        n_classes=101,
        flow=False,
        shuffle=True,
        is_training=True,
        num_frames=64,
    ):
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.flow = flow
        self.on_epoch_end()
        self.class_dict = class_dict
        self.is_training = is_training
        self.num_frames = num_frames

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp, self.flow)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, filenames, flow=False):
        batch = []
        labels = []
        logfile = open("log.txt", "a")
        for video in filenames:
            folder = video.split(".")[0]
            labels.append(get_label(folder, self.class_dict))
            if self.flow == False:
                files = sorted(glob(PATH_TO_FRAMES + folder + "/i_*"))
                if not files:
                    print(PATH_TO_FRAMES + folder)
                    logfile.write(PATH_TO_FRAMES + folder + "\n")
                if self.is_training == True:
                    files = random_temporal_crop(files, self.num_frames, 30)
                else:
                    files = center_temporal_crop(files, self.num_frames)
                images = read_rgb_images(files, self.is_training)
                batch.append(images)
            else:
                files = sorted(glob(PATH_TO_FRAMES + folder + "/m_*"))
                if self.is_training == True:
                    files = random_temporal_crop(files, self.num_frames, 30)
                else:
                    files = center_temporal_crop(files, self.num_frames)
                images = read_flow_images(files, self.is_training)
                batch.append(images)

        batch = np.vstack(batch)

        # labels need to be sequential from 0 to n_classes
        labels = np.array(labels).astype(np.int)
        new_labels_dict = {}
        for i, j in zip(
            np.unique(list(self.class_dict.values())),
            range(len(np.unique(list(self.class_dict.values())))),
        ):
            # print(i)
            new_labels_dict[i] = int(j)

        # print (new_labels_dict)
        labels = [new_labels_dict[i] for i in labels]
        cat_labels = to_categorical(labels, num_classes=self.n_classes)
        return batch, cat_labels
