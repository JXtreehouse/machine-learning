import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import h5py
from random import shuffle
import os
import glob

def generate_h5(data_path, labels_to_classes_dictionary, outfile_path=None, shuffle_data=True, resize_height=224,
                resize_width=224, data_order='tf', train_dev_test_ratio=[0.6, 0.2, 0.2]):
    def label_to_class(label):
        return labels_to_classes_dictionary[str(label)]

    if outfile_path is None:
        outfile_path = "dataset_" + str(resize_height) + "_" + str(resize_width) + ".h5"

    # read all addresses and all 'label' folders
    all_addrs = []
    all_labels = []

    for label_int, label_str in labels_to_classes_dictionary.iteritems():
        label_dir = os.path.join(data_path, label_str)
        if os.path.isdir(label_dir) and os.path.exists(label_dir):
            # addrs = glob.glob(os.path.join(label_dir, "*." + data_file_ext))
            addrs = glob.glob(os.path.join(label_dir, "*.*"))
            labels = [int(label_int) for addr in addrs]

            all_addrs.extend(addrs)
            all_labels.extend(labels)

    # addrs = glob.glob(cat_dog_train_path)
    # labels = [0 if 'cat' in addr else 1 for addr in addrs]

    if shuffle_data:
        c = list(zip(all_addrs, all_labels))
        shuffle(c)
        all_addrs, all_labels = zip(*c)

    # Divide the data into 60% train, 20% validation, and 20% test
    train_ratio = train_dev_test_ratio[0]
    dev_ratio = train_dev_test_ratio[1]
    test_ratio = train_dev_test_ratio[2]

    train_addrs = all_addrs[0: int(train_ratio * len(all_addrs))]
    train_labels = all_labels[0: int(train_ratio * len(all_labels))]

    dev_addrs = all_addrs[int(train_ratio * len(all_addrs)):int((train_ratio + dev_ratio) * len(all_addrs))]
    dev_labels = all_labels[int(train_ratio * len(all_addrs)):int((train_ratio + dev_ratio) * len(all_addrs))]

    test_addrs = all_addrs[int((train_ratio + dev_ratio) * len(all_addrs)):]
    test_labels = all_labels[int((train_ratio + dev_ratio) * len(all_labels)):]

    if data_order == 'th':  # channel first
        train_shape = (len(train_addrs), 3, resize_height, resize_width)
        dev_shape = (len(dev_addrs), 3, resize_height, resize_width)
        test_shape = (len(test_addrs), 3, resize_height, resize_width)
    elif data_order == 'tf':
        train_shape = (len(train_addrs), resize_height, resize_width, 3)
        dev_shape = (len(dev_addrs), resize_height, resize_width, 3)
        test_shape = (len(test_addrs), resize_height, resize_width, 3)

    # open a hdf5 file and create earrays
    hdf5_file = h5py.File(outfile_path, mode='w')

    hdf5_file.create_dataset("train_set_x", train_shape, np.uint8)
    hdf5_file.create_dataset("dev_set_x", dev_shape, np.uint8)
    hdf5_file.create_dataset("test_set_x", test_shape, np.uint8)

    # hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)

    hdf5_file.create_dataset("train_set_y", (len(train_addrs),), np.uint8)
    hdf5_file["train_set_y"][...] = train_labels
    hdf5_file.create_dataset("dev_set_y", (len(dev_addrs),), np.uint8)
    hdf5_file["dev_set_y"][...] = dev_labels
    hdf5_file.create_dataset("test_set_y", (len(test_addrs),), np.uint8)
    hdf5_file["test_set_y"][...] = test_labels

    # list of classes
    list_classes = [""] * len(labels_to_classes_dictionary)
    for label_int, label_str in labels_to_classes_dictionary.iteritems():
        list_classes[int(label_int)] = label_str

    list_classes = [n.encode("ascii", "ignore") for n in list_classes]

    hdf5_file.create_dataset("list_classes", (len(list_classes),), 'S10')
    hdf5_file["list_classes"][...] = np.array(list_classes)

    # loop over train addresses
    for i in range(len(train_addrs)):
        # print how many images are saved every 1000 images
        if i % 1000 == 0 and i > 1:
            print
            'Train data: {}/{}'.format(i, len(train_addrs))
        # read an image and resize to (resize_height, resize_width)
        addr = train_addrs[i]
        img = ndimage.imread(addr, flatten=False)
        img = scipy.misc.imresize(img, size=(resize_height, resize_width))

        # add any image pre-processing here
        # if the data order is Theano, axis orders should change
        if data_order == 'th':
            img = np.rollaxis(img, 2)
        # save the image and calculate the mean so far

        hdf5_file["train_set_x"][i, ...] = img
        # mean += img / float(len(train_labels))

    # loop over dev addresses
    for i in range(len(dev_addrs)):
        # print how many images are saved every 1000 images
        if i % 1000 == 0 and i > 1:
            print
            'Validation data: {}/{}'.format(i, len(dev_addrs))
        # read an image and resize to (resize_height, resize_width)
        addr = dev_addrs[i]
        img = ndimage.imread(addr, flatten=False)
        img = scipy.misc.imresize(img, size=(resize_height, resize_width))

        # add any image pre-processing here
        # if the data order is Theano, axis orders should change
        if data_order == 'th':
            img = np.rollaxis(img, 2)
        # save the image

        hdf5_file["dev_set_x"][i, ...] = img

    # loop over test addresses
    for i in range(len(test_addrs)):
        # print how many images are saved every 1000 images
        if i % 1000 == 0 and i > 1:
            print
            'Test data: {}/{}'.format(i, len(test_addrs))
        # read an image and resize to (resize_height, resize_width)

        addr = test_addrs[i]
        img = ndimage.imread(addr, flatten=False)
        img = scipy.misc.imresize(img, size=(resize_height, resize_width))

        # add any image pre-processing here
        # if the data order is Theano, axis orders should change
        if data_order == 'th':
            img = np.rollaxis(img, 2)
        # save the image

        hdf5_file["test_set_x"][i, ...] = img

    # save and close the hdf5 file

    hdf5_file.close()


def load_all_data(hdf5_filename):
    dataset = h5py.File(hdf5_filename, "r")

    train_set_x_orig = np.array(dataset["train_set_x"][:])
    train_set_y_orig = np.array(dataset["train_set_y"][:])

    dev_set_x_orig = np.array(dataset["dev_set_x"][:])
    dev_set_y_orig = np.array(dataset["dev_set_y"][:])

    test_set_x_orig = np.array(dataset["test_set_x"][:])
    test_set_y_orig = np.array(dataset["test_set_y"][:])

    train_set_y_orig = train_set_y_orig.reshape((train_set_y_orig.shape[0], 1))
    dev_set_y_orig = dev_set_y_orig.reshape((dev_set_y_orig.shape[0], 1))
    test_set_y_orig = test_set_y_orig.reshape((test_set_y_orig.shape[0], 1))

    classes = dataset["list_classes"][:]

    return train_set_x_orig, train_set_y_orig, dev_set_x_orig, dev_set_y_orig, test_set_x_orig, test_set_y_orig, classes

train_set_x_orig, train_set_y,dev_set_x_orig, dev_set_y_orig, test_set_x_orig, test_set_y, classes = load_all_data()

num_px = train_set_x_orig.shape[1]
m = train_set_x_orig[2]

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


def sigmoid(z):
    s = 1 / (1 + np.power(np.e, -z))
    return s


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(w, b, X, Y):
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log (1-A),axis = 1, keepdims=True)

    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y, axis=1, keepdims=True)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw, "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=True):

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 99 == 0:
            costs.append(cost)

        if print_cost and i % 99 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    Y_prediction = np.rint(A)

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

    w, b = initialize_with_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, False)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


my_image = "my_image.jpg"

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.001, print_cost = False)
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")