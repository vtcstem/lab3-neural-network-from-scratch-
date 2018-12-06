import numpy as np
from data_util import DataUtils
from neuralnetwork import *
import time
from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt 

# define paths of data files
trainfile_X = 'MNIST_dataset/train-images.idx3-ubyte'
trainfile_y = 'MNIST_dataset/train-labels.idx1-ubyte'
testfile_X = 'MNIST_dataset/t10k-images.idx3-ubyte'
testfile_y = 'MNIST_dataset/t10k-labels.idx1-ubyte'

number_of_classes = 10 # because we have 0 - 9 digits / output cases

def outImg(arrX, arrY, num_out, out_path):
    """
    Display the mnist dataset as PNG image
    Input parameters: ndarray of input data X, the respective label y and number of PNG images to be generated
    Return: folder containing the PNG images
    Source: https://github.com/csuldw/MachineLearning/blob/master/utils/data_util.py
    """
    m, n = np.shape(arrX)
    if num_out > m:
        print("Invalid argument num_out. Must be <= total number of images, "+str(m))
        return 0

    for i in range(num_out):
        img = np.array(arrX[i])
        img = img.reshape(28,28)
        outfile = str(i) + "_" +  str(arrY[i]) + ".png"
        plt.figure()
        plt.imshow(img, cmap = 'binary') 
        plt.savefig(outpath + "/" + outfile)

        
def load_dataset(display = False, flag="train"):
    """
    Load MNIST hand-written digit dataset, which consists of 60000 samples of training set and 100000 samples of test set.
    Each set contains label Y (digit 0 to 9) and image of hand-written digit X
    Input parametesr: boolean "display" for controlling the display of loaded data in image format
    Return: X_data for inputting to the neural network and y_data for the labels of X_data
    """

    if flag =="train":
        print("Loading training set image X ...")
        train_X_data = DataUtils(filename=trainfile_X).getImage()
        print("Loading training set label y ...")
        train_y_data = DataUtils(filename=trainfile_y).getLabel()
        print("size of training set X = ", train_X_data.shape)
        print("size of training set y = ", train_y_data.shape)     

        if display:
            path_trainset = "MNIST_dataset/imgs_train"
            if not os.path.exists(path_trainset):
                os.mkdir(path_trainset)
            outImg(train_X_data, train_y_data, 30, out_path)
            DataUtils(outpath=path_trainset).outImg(train_X_data, train_y_data, 30)

        return train_X_data, train_y_data
        
    elif flag == "test":
        print("Loading test set image X ...")
        test_X_data = DataUtils(testfile_X).getImage()
        print("Loading test set label y ...")
        test_y_data = DataUtils(testfile_y).getLabel()
        print("size of test set X = ", test_X_data.shape)
        print("size of test set y = ", test_y_data.shape)

        if display:
            path_testset = "MNIST_dataset/imgs_test"
            if not os.path.exists(path_testset):
                os.mkdir(path_testset)
            DataUtils(outpath=path_testset).outImg(test_X_data, test_y_data, 30)

        return test_X_data, test_y_data

# convert 
def one_hot(a, num_classes):
    '''
    Convert an input data a into an "one-hot" representation
    Input Arguments:
    num_classes: Number of classes of the data
    (e.g. num_classes = 10 for 0 - 9 digits)
    a: 1-D array (vector) with shape (number of examples, )
    Output:
    a matrix with dimension (number of examples, num_classes)
    '''
    return None

def main(args):
    global nn

    n_h = args.n_h # number of hidden units
    n_y = number_of_classes 

    # load datasets
    train_X_data, train_y_data = load_dataset(args.display, "train")

    # reshape train_label Y into (m x 10) matrix
    one_hot_train_y_data = one_hot(train_y_data, number_of_classes)
    print("size of one-hot vector training label = ", one_hot_train_y_data.shape)

    # get the shape of the input and output layer of NN, depending on the data
    m, n_x = None

    # initialize neural network
    nn = None

    if not (os.path.isfile(args.weight_path)):
        # start training the nn
        print("begin training...")
        start_time = time.time()
        #### Call nn.train method to train the nn
        ####
        end_time = time.time()
        print("Training is complete. Total training time taken: {:10.2f}sec".format(end_time - start_time))

        print("Saving the trained weights into file...")
        # output weights file
        if nn.output() == 0:
            sys.exit(0)
        print("="*50)
    else:
        print("Found existing weights. Loading...")
        weights = open(args.weight_path, 'rb')
        nn.input(weights)
        print("Finish loading weights into the neural network")

    # run predictions on test dataset

    # load test data
    test_X_data, test_y_data = load_dataset(args.display, "test")

    # obtain sample number
    m, _ = None

    # run prediction
    predictions_one_hot=nn.predict_mnist(test_X_data.T)

    # transpose the output vector 
    predictions_one_hot = predictions_one_hot.T

    # convert one-hot vector into numbers
    y_hat = np.argmax(predictions_one_hot, axis=1)
    print("shape of prediction = ",y_hat.shape)
    print("first 10 sample of y_hat: ",y_hat[0:10])
    print("first 10 sample of test_y: ",test_y_data[0:10])

    # evaluate accuracy of our trained model
    diff = y_hat - test_y_data

    accuracy = 100 - (np.count_nonzero(diff) / m * 100)
    print("accuracy of the model {:3.2f}%".format(accuracy))

    cost = nn.getcostlist()
    plt.plot(cost)
    plt.title("Cost function over iteration")
    plt.ylabel('cost')
    plt.xlabel('iteration')
    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-w", "--weight", help="specify the path of pre-trained weights for skipping training and running predictions on test data directly", dest="weight_path", default="nn_weights.dat")
    parser.add_argument("-nh", "--hidden_size", help="define the size of the hidden unit, default = 50 units", dest="n_h", type=int, default=50)
    parser.add_argument("-lr", "--learningrate", help="define the learning rate, default = 0.15", dest="lr", type=float, default=0.15)
    parser.add_argument("-it", "--iteration", help="define the iterations for training, default = 2000 steps", dest="iter", type=int, default=2000)
    parser.add_argument("-d", "--display", help="display dataset after loading", dest="display", type=bool, default=False)
    args = parser.parse_args()
    main(args)
