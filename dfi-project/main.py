from time import time

import numpy as np
import pandas
import tensorflow as tf
import os
import utils
from vgg19 import Vgg19
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import minimize
from scipy.optimize import  fmin_l_bfgs_b

model = np.load("vgg19.npy", encoding='latin1').item()

K = 10 ** 1
ALPHA = 0.4
LAMB = 0.001
BETA = 2

graph = None
sess = None
nn = None
tensors = None


def phi(imgs=[]):
    """Transform list of images into deep feature space

    :param imgs: input images
    :return: deep feature transformed images
    """
    # Design the graph.


    t0 = time()
    [conv3,
     #conv4,
     #conv5
     ] = sess.run(tensors,
                       feed_dict={
                           nn.inputRGB: imgs
                       })
    t1 = time()
    print('Took {}'.format(t1 - t0))
    res = []
    for idx in range(len(imgs)):
        phi_img = conv3[idx].reshape(-1)
        #phi_img = np.append(conv3[idx].reshape(-1),
        #                     np.append(conv4[idx].reshape(-1),
        #                             conv5[idx].reshape(-1))
        #                    )
        res.append(phi_img)
    return [np.linalg.norm(x) for x in res]


def main(feat='No Beard', person_index=0):
    t_start=time()
    with tf.device('/gpu:0'):

        global graph
        graph = tf.Graph()
        with graph.as_default():
            global nn
            nn = Vgg19(model=model)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False


    config.gpu_options.per_process_gpu_memory_fraction = 0.80

# Run the graph in the session.
    global sess
    with tf.Session(graph=graph, config=config) as sess:
        tf.global_variables_initializer().run()
        global tensors
        tensors = [graph.get_tensor_by_name('conv3_1/Relu:0')
            #, graph.get_tensor_by_name('conv4_1/Relu:0')
            #, graph.get_tensor_by_name('conv5_1/Relu:0')
                   ]

        print('Initialized session')

        atts = load_discrete_lfw_attributes()
        imgs_path = atts['path'].values
        person_img = reduce_img_size(utils.load_images(*[imgs_path[0]]))

        pos_paths, neg_paths = get_sets(atts, feat, person_index)

        pos_paths = pos_paths.as_matrix()
        neg_paths = neg_paths.as_matrix()

        pos_imgs = reduce_img_size(utils.load_images(*pos_paths))
        neg_imgs = reduce_img_size(utils.load_images(*neg_paths))

        pos_deep_features = phi(pos_imgs)
        neg_deep_features = phi(neg_imgs)

        w = np.mean(pos_deep_features, axis=0) - np.mean(neg_deep_features, axis=0)
        w /= np.linalg.norm(w)

        phi_x = phi(person_img)[0]
        phi_z = phi_x + ALPHA * w

        bounds = []

        initial_guess = np.array(person_img[0]).reshape(-1)

        for i in range(initial_guess.shape[0]):
            bounds.append((0,255))


        print('Starting minimize function')
        opt_res = minimize(fun=minimize_z,
                           x0=initial_guess,
                           args=(phi_z, LAMB, BETA),
                           method='L-BFGS-B',
                           options={#'maxfun': 10,
                               'disp': True,
                               'eps' : 5,
                               # 'factr':10**10,
                               'maxiter':1
                           },
                           bounds= bounds
                           )


        t_end=time()
        print(t_end-t_start)
        return opt_res.x


def minimize_z(z, phi_z, lamb, beta):
    # Reshape into image form

    z = z.reshape(-1, 224, 224, 3)

    first_term = 0.5 * np.linalg.norm(phi_z - phi(z)[0]) ** 2
    second_term = lamb * R(z, beta)
    res = first_term + second_term
    print(first_term)
    print(second_term)
    #print(res)
    return res


def R(z, beta):
    """Total Variation regularizer

    :param z: objective
    :param beta: beta
    :return: R
    """
    result = 0
    z=z[0]
    for i in range(z.shape[0] - 1):
        for j in range(z.shape[1] - 1):
            var = np.linalg.norm(z[i][j + 1] - z[i][j]) ** 2 + \
                  np.linalg.norm(z[i + 1][j] - z[i][j]) ** 2

            result += var ** (beta * 0.5)

    # normalize R
    result /= np.prod(z.shape, dtype=np.float32)

    return result


def reduce_img_size(imgs):
    for idx, img in enumerate(imgs):
        imgs[idx] = img[13:-13, 13:-13]
    return imgs


def get_sets(atts, feat, person_index):
    person = atts.loc[person_index]
    del person['person']
    del person['path']
    # Remove person from df
    atts = atts.drop(person_index)
    # Split by feature
    neg_set = atts.loc[atts[feat] == -1]
    pos_set = atts.loc[atts[feat] == 1]
    pos_paths = get_k_neighbors(pos_set, person)
    neg_paths = get_k_neighbors(neg_set, person)

    return pos_paths, neg_paths


def get_k_neighbors(subset, person):
    del subset['person']
    paths = subset['path']
    del subset['path']

    knn = KNeighborsClassifier(n_jobs=4)
    dummy_target = [0 for x in range(subset.shape[0])]
    knn.fit(X=subset.as_matrix(), y=dummy_target)
    knn_indices = knn.kneighbors(person.as_matrix(), n_neighbors=K, return_distance=False)[0]

    neighbor_paths = paths.iloc[knn_indices]

    return neighbor_paths


def load_lfw_attributes():
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


def load_discrete_lfw_attributes():
    """Loads the discretized lfw attributes

    :return: Discretized lfw attributes
    """
    df = load_lfw_attributes()

    for column in df:
        if column == 'person' or column == 'path':
            continue
        df[column] = df[column].apply(np.sign)

    return df


def load_lfw_images():
    images = []
    for subdir, dirs, files in os.walk('./data/lfw-deepfunneled'):
        for file in files:
            images.append(os.path.join(subdir, file))
    return images


if __name__ == '__main__':
    main()
