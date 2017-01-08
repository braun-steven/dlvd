from time import time

import numpy as np
import pandas
import tensorflow as tf
import os
import utils
from vgg19 import Vgg19
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import minimize


class DFI:
    def __init__(self, k=10, alpha=0.4, lamb=0.001, beta=2, model_path="vgg19.npy", num_layers=3, gpu=True):
        # Set variables
        self._k = k
        self._alpha = alpha
        self._beta = beta
        self._lamb = lamb
        self._num_layers = num_layers
        self._model = self._load_model(model_path)
        self._gpu = gpu

        self._tensor_names = ['conv3_1/Relu:0', 'conv4_1/Relu:0', 'conv5_1/Relu:0']
        self._sess = None

        # Set device
        device = '/gpu:0' if self._gpu else '/cpu:0'

        # Setup
        with tf.device(device):
            self._graph = tf.Graph()
            with self._graph.as_default():
                self._nn = Vgg19(model=self._model)

    def run(self, feat='No Beard', person_index=0):
        # Config for gpu
        config = tf.ConfigProto()
        if self._gpu:
            config.gpu_options.allow_growth = False
            config.gpu_options.per_process_gpu_memory_fraction = 0.80

        # Run the graph in the session.
        with tf.Session(graph=self._graph, config=config) as self._sess:
            tf.global_variables_initializer().run()

            self._tensors = [self._graph.get_tensor_by_name(self._tensor_names[idx]) for idx in range(self._num_layers)]

            print('Initialized session')

            t_start = time()
            atts = self._load_discrete_lfw_attributes()
            imgs_path = atts['path'].values
            person_img = self._reduce_img_size(utils.load_images(*[imgs_path[0]]))[0]

            pos_paths, neg_paths = self._get_sets(atts, feat, person_index)

            pos_paths = pos_paths.as_matrix()
            neg_paths = neg_paths.as_matrix()

            pos_imgs = self._reduce_img_size(utils.load_images(*pos_paths))
            neg_imgs = self._reduce_img_size(utils.load_images(*neg_paths))

            pos_deep_features = self._phi(pos_imgs)
            neg_deep_features = self._phi(neg_imgs)

            w = np.mean(pos_deep_features, axis=0) - np.mean(neg_deep_features, axis=0)
            w /= np.linalg.norm(w)

            phi_x = self._phi(person_img)
            phi_z = phi_x + self._alpha * w

            initial_guess = np.array(person_img).reshape(-1)

            # Create bounds
            bounds = []
            for i in range(initial_guess.shape[0]):
                bounds.append((0, 255))

            print('Starting minimize function')
            opt_res = minimize(fun=self._minimize_z,
                               x0=initial_guess,
                               args=(phi_z, self._lamb, self._beta),
                               method='L-BFGS-B',
                               options={
                                   # 'maxfun': 10,
                                   'disp': True,
                                   'eps': 5,
                                   'maxiter': 1
                               },
                               bounds=bounds
                               )

            t_end = time()
            print(t_end - t_start)
            return opt_res.x

    def attributes(self):
        print('TODO: Implement')
        pass

    def _load_model(self, model_path):
        return np.load(model_path, encoding='latin1').item()

    def _phi(self, imgs):
        """Transform list of images into deep feature space

        :param imgs: input images
        :return: deep feature transformed images
        """

        if not isinstance(imgs, list):
            input_images = [imgs]
        else:
            input_images = imgs

        t0 = time()
        ret = self._sess.run(self._tensors,
                             feed_dict={
                                 self._nn.inputRGB: input_images
                             })
        t1 = time()
        print('Took {}'.format(t1 - t0))
        res = []

        # Create result list
        for img_idx in range(len(input_images)):
            phi_img = np.array([])

            # Append all layer results to a (M,) vector
            for layer_idx in range(self._num_layers):
                phi_img = np.append(phi_img, ret[layer_idx][img_idx].reshape(-1))

            res.append(phi_img)

        # Handle correct return type and normalize (L2)
        if not isinstance(imgs, list):
            return np.linalg.norm(res[0])  # Single image
        else:
            return [np.linalg.norm(x) for x in res]  # List of images

    def _load_lfw_attributes(self):
        """Loads the lfw attribute file

        :return: Pandas dataframe containing the lfw attributes for each image
        """
        path = './data/lfw_attributes.txt'
        df = pandas.read_csv(path, sep='\t')

        paths = []

        for idx, row in df.iterrows():
            name = row[0]
            img_idx = str(row[1])
            name = name.replace(' ', '_')

            while len(img_idx) < 4:
                img_idx = '0' + img_idx

            path = './data/lfw-deepfunneled/{0}/{0}_{1}.jpg'.format(name, img_idx)
            paths.append(path)
        df['path'] = paths
        del df['imagenum']
        return df

    def _load_discrete_lfw_attributes(self):
        """Loads the discretized lfw attributes

        :return: Discretized lfw attributes
        """
        df = self._load_lfw_attributes()

        for column in df:
            if column == 'person' or column == 'path':
                continue
            df[column] = df[column].apply(np.sign)

        return df

    def _minimize_z(self, z, phi_z, lamb, beta):
        # Reshape into image form

        z = z.reshape(224, 224, 3)

        loss = 0.5 * np.linalg.norm(phi_z - self._phi(z)) ** 2
        total_variation = lamb * self._R(z, beta)
        res = loss + total_variation
        print(loss)
        print(total_variation)
        # print(res)
        return res

    def _R(self, z, beta):
        """Total Variation regularizer

        :param z: objective
        :param beta: beta
        :return: R
        """
        result = 0
        for i in range(z.shape[0] - 1):
            for j in range(z.shape[1] - 1):
                var = np.linalg.norm(z[i][j + 1] - z[i][j]) ** 2 + \
                      np.linalg.norm(z[i + 1][j] - z[i][j]) ** 2

                result += var ** (beta * 0.5)

        # normalize R
        result /= np.prod(z.shape, dtype=np.float32)

        return result

    def _reduce_img_size(self, imgs):
        for idx, img in enumerate(imgs):
            imgs[idx] = img[13:-13, 13:-13]
        return imgs

    def _get_sets(self, atts, feat, person_index):
        person = atts.loc[person_index]
        del person['person']
        del person['path']

        # Remove person from df
        atts = atts.drop(person_index)

        # Split by feature
        pos_set = atts.loc[atts[feat] == 1]
        neg_set = atts.loc[atts[feat] == -1]
        pos_paths = self._get_k_neighbors(pos_set, person)
        neg_paths = self._get_k_neighbors(neg_set, person)

        return pos_paths, neg_paths

    def _get_k_neighbors(self, subset, person):
        del subset['person']
        paths = subset['path']
        del subset['path']

        knn = KNeighborsClassifier(n_jobs=4)
        dummy_target = [0 for x in range(subset.shape[0])]
        knn.fit(X=subset.as_matrix(), y=dummy_target)
        knn_indices = knn.kneighbors(person.as_matrix(), n_neighbors=self._k, return_distance=False)[0]

        neighbor_paths = paths.iloc[knn_indices]

        return neighbor_paths
