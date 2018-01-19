from pathlib import Path
import numpy as np
import os
import cv2
from scipy.ndimage import rotate
from utils.image_utils import get_image_paths, load_images
import cv2
from umeyama import umeyama

def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
    h,w = image.shape[0:2]
    rotation = np.random.uniform( -rotation_range, rotation_range )
    scale = np.random.uniform( 1 - zoom_range, 1 + zoom_range )
    tx = np.random.uniform( -shift_range, shift_range ) * w
    ty = np.random.uniform( -shift_range, shift_range ) * h
    mat = cv2.getRotationMatrix2D( (w//2,h//2), rotation, scale )
    mat[:,2] += (tx,ty)
    result = cv2.warpAffine( image, mat, (w,h), borderMode=cv2.BORDER_REPLICATE )
    if np.random.random() < random_flip:
        result = result[:,::-1]
    return result

# get pair of random warped images from aligened face image
def random_warp(image):
    assert image.shape == (256,256,3)
    range_ = np.linspace( 128-80, 128+80, 5 )
    mapx = np.broadcast_to( range_, (5,5) )
    mapy = mapx.T

    mapx = mapx + np.random.normal( size=(5,5), scale=5 )
    mapy = mapy + np.random.normal( size=(5,5), scale=5 )

    interp_mapx = cv2.resize( mapx, (80,80) )[8:72,8:72].astype('float32')
    interp_mapy = cv2.resize( mapy, (80,80) )[8:72,8:72].astype('float32')

    warped_image = cv2.remap( image, interp_mapx, interp_mapy, cv2.INTER_LINEAR )

    src_points = np.stack( [ mapx.ravel(), mapy.ravel() ], axis=-1 )
    dst_points = np.mgrid[0:65:16,0:65:16].T.reshape(-1,2)
    mat = umeyama( src_points, dst_points, True )[0:2]

    target_image = cv2.warpAffine( image, mat, (64,64) )

    return warped_image, target_image

random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
    }

def get_training_data(images, batch_size):
    indices = np.random.randint(len(images), size=batch_size)
    for i,index in enumerate(indices):
        image = images[index]
        image = random_transform(image, **random_transform_args)
        warped_img, target_img = random_warp(image)

        if i == 0:
            warped_images = np.empty((batch_size,) + warped_img.shape, warped_img.dtype)
            target_images = np.empty((batch_size,) + target_img.shape, warped_img.dtype)

        warped_images[i] = warped_img
        target_images[i] = target_img

    return warped_images, target_images

class FaceDataset(object):
    def __init__(self, batch_size, train_ratio=0.95, num_gpus=1, resize=(64, 64)):

        self.iterations = 0
        self.num_gpus = num_gpus
        self.x_A, self.x_B = self.load_dataset()

        train_A_index = int(self.x_A.shape[0] * 0.95)
        train_B_index = int(self.x_B.shape[0] * 0.95)

        if self.x_A.shape[0]-train_A_index < batch_size:
            train_A_index = self.x_A.shape[0]-batch_size

        if self.x_B.shape[0]-train_B_index < batch_size:
            train_B_index = self.x_B.shape[0]-batch_size


        print(self.x_A.shape, self.x_B.shape, train_ratio, train_A_index, train_B_index)

        self.x_train_A, self.x_val_A = self.x_A[:train_A_index], self.x_A[train_A_index:]
        self.x_train_B, self.x_val_B = self.x_B[:train_B_index], self.x_B[train_B_index:]

        self.batch_size = batch_size

        self.train_index = 0
        self.val_index = 0
        self.resize = resize

        self.random_transform_args = {
        'rotation_range': 10,
        'zoom_range': 0.05,
        'shift_range': 0.05,
        'random_flip': 0.4,
        }

    def load_dataset(self):
        return NotImplementedError

    def preprocess_data(self, x):
        reverse_x = np.ones(shape=x.shape)
        
        for channel in range(x.shape[-1]):
            reverse_x[:, :, :, :, x.shape[-1] - 1 - channel] = x[:, :, :, :, channel]
        x = reverse_x
        
        x = 2 * x - 1
        
        return x

    def get_image_paths(self, directory):
        return [x.path for x in os.scandir(directory) if
                x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".jpeg") or x.name.endswith(
                    ".JPG")]

    def load_images(self, image_paths, convert=None):
        iter_all_images = (cv2.imread(fn) for fn in image_paths)
        if convert:
            iter_all_images = (convert(img) for img in iter_all_images)
        for i, image in enumerate(iter_all_images):
            if i == 0:
                all_images = np.empty(
                    (len(image_paths),) + image.shape, dtype=image.dtype)
            all_images[i] = image
        return all_images

    def reconstruct_original(self, x):
        x = (x+1) / 2
        return x

    def rotate_image_pair_batch(self, x_batch_A, x_batch_B):
        k = np.random.randint(low=0, high=359)
        x_batch_A = self.rotate_batch(x_batch=x_batch_A, k=k, axis=(0, 1))
        x_batch_B = self.rotate_batch(x_batch=x_batch_B, k=k, axis=(0, 1))
        return x_batch_A, x_batch_B



    def augment_batch(self, image_batch, augment=False):
        x_input = []
        x_target = []

        if augment:
            for image in image_batch:
                image = random_transform(image=image, rotation_range=self.random_transform_args['rotation_range'],
                                                      zoom_range=self.random_transform_args['zoom_range'],
                                                      random_flip=self.random_transform_args['random_flip'],
                                                      shift_range=self.random_transform_args['shift_range'])

                augmented_img, target_img = random_warp(image)
                x_input.append(augmented_img)
                x_target.append(target_img)
            x_input = np.array(x_input)
            x_target = np.array(x_target)
        else:
            x_input = image_batch
            x_target = image_batch

        return x_input, x_target

    def rotate_batch(self, x_batch, axis, k):
        x_batch = rotate(x_batch, k, reshape=False, axes=axis, mode="nearest")
        return x_batch

    def shuffle(self, x):

        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices]

        return x

    def get_train_batch(self, augment=True):
        x_train_input_B = np.ones(shape=(self.num_gpus, self.batch_size,
                                         self.resize[0],
                                         self.resize[1], self.x_train_A.shape[3]), dtype=np.float32)

        x_train_target_B = np.ones(shape=(self.num_gpus, self.batch_size,
                                          self.resize[0],
                                          self.resize[1], self.x_train_A.shape[3]), dtype=np.float32)

        x_train_input_A = np.ones(shape=(self.num_gpus, self.batch_size,
                                         self.resize[0],
                                         self.resize[1], self.x_train_A.shape[3]), dtype=np.float32)

        x_train_target_A = np.ones(shape=(self.num_gpus, self.batch_size,
                                          self.resize[0],
                                          self.resize[1], self.x_train_A.shape[3]), dtype=np.float32)

        for gpu in range(self.num_gpus):
            choose_samples_A = np.random.choice(np.arange(self.x_train_A.shape[0]), size=self.batch_size, replace=True)
            choose_samples_B = np.random.choice(np.arange(self.x_train_B.shape[0]), size=self.batch_size, replace=True)
            x_train_A = self.x_train_A[choose_samples_A]
            x_train_B = self.x_train_B[choose_samples_B]
            x_train_input_A[gpu], x_train_target_A[gpu] = self.augment_batch(x_train_A, augment=augment)
            x_train_input_B[gpu], x_train_target_B[gpu] = self.augment_batch(x_train_B, augment=augment)

        x_train_input_A, x_train_target_A = self.preprocess_data(np.array(x_train_input_A)), \
                                                      self.preprocess_data(np.array(x_train_target_A))
        x_train_input_B, x_train_target_B = self.preprocess_data(np.array(x_train_input_B)), \
                                                      self.preprocess_data(np.array(x_train_target_B))

        return x_train_input_A, x_train_target_A, x_train_input_B, x_train_target_B

    def get_val_batch(self):
        x_val_B = np.ones(shape=(self.num_gpus, self.batch_size,
                                 self.resize[0],
                                 self.resize[1], self.x_train_A.shape[3]), dtype=np.float32)

        x_val_A = np.ones(shape=(self.num_gpus, self.batch_size,
                                 self.resize[0],
                                 self.resize[1], self.x_train_A.shape[3]), dtype=np.float32)

        for gpu in range(self.num_gpus):
            for j in range(self.batch_size):
                choose_samples_A = np.random.choice(np.arange(self.x_val_A.shape[0]))
                choose_samples_B = np.random.choice(np.arange(self.x_val_B.shape[0]))
                _, x_val_A[gpu, j] = random_warp(self.x_val_A[choose_samples_A])
                _, x_val_B[gpu, j] = random_warp(self.x_val_B[choose_samples_B])

        x_val_A = self.preprocess_data(np.array(x_val_A))
        x_val_B = self.preprocess_data(np.array(x_val_B))

        return x_val_A, x_val_A, x_val_B, x_val_B

class AToBDataset(FaceDataset):
    def __init__(self, batch_size, path_images_A, path_images_B, train_ratio=0.88, num_gpus=1):
        self.path_images_A = path_images_A
        self.path_images_B = path_images_B
        super(AToBDataset, self).__init__(batch_size, train_ratio, num_gpus)


    def load_dataset(self):
        person_A_training_data = self.path_images_A#training_data_dir / Path("daisy_ridley")
        person_B_training_data = self.path_images_B#training_data_dir / Path("ryan_gosling")
        images_A = get_image_paths(str(person_A_training_data))
        images_B = get_image_paths(str(person_B_training_data))
        images_A = load_images(images_A)
        images_A = images_A / np.max(images_A)
        images_B = load_images(images_B)
        images_B = images_B / np.max(images_B)

        print(images_A.max(), images_B.max())

        return images_A, images_B


