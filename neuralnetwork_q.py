import numpy as np
import pickle

class neuralnetwork:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, learning_rate=0.12):
        self.params = {}
        np.random.seed(2)
        self.params['W1'] = weight_init_std * np.random.randn(hidden_size, input_size)
        self.params['b1'] = np.zeros((hidden_size, 1))
        self.params['W2'] = weight_init_std * np.random.randn(output_size, hidden_size)
        self.params['b2'] = np.zeros((output_size, 1))

        self.cost = []
        self.m = 0
        self.learning_rate = learning_rate

        self.cache = {}
        self.trained = False

        self.input_size = input_size
        self.hidden_size = hidden_size
        self. output_size = output_size

    def set_learning_rate(learning_rate = 0.12):
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return np.multiply(self.sigmoid(x) , (1 - self.sigmoid(x)))

    def tanh(self, x):
        return None

    def tanh_prime(self, x):
        return None

    def softmax(self, x):
        return None

    def forward(self, X): 
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2'] 

        # Forward propagation 
        Z1 = np.dot(W1, X) + b1 
        A1 = self.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        #A2 = self.sigmoid(Z2) 
        A2 = self.softmax(Z2)

        # save Z1, A1, Z2 and A2 in cache for later use
        self.cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    def cross_entropy_loss(self, Y):
        """
        computes cross entropy loss function
        input: Y, ground-truth of input data X
        output: J, cost value
        """
        # get y_hat
        y_hat = self.cache["A2"]
        assert self.m != 0

        ### Your code here (~1-2 lines) ###
        J = -1/self.m * np.sum( np.multiply(Y, np.log(y_hat)) + np.multiply((1 - Y), np.log(1 - y_hat) ) )
        
        # repack J into a scalar, e.g. not [[19.0]]
        J = np.squeeze(J)
       
        return J

    def multiclass_loss(self, Y):
        """
        computes loss function for multiple classes problem
        input: Y, ground-truth of input data X
        output: J, cost value
        """
        # get y_hat
        y_hat = self.cache["A2"]
        assert self.m != 0

        ### Your code here ###
        J = np.squeeze(J)
        return J

    def backward(self, X, Y):
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2'] 
        Z1 = self.cache['Z1']
        A1 = self.cache["A1"]
        A2 = self.cache["A2"]

        assert self.m != 0

        # Backward propagation
        ### Your code here (6 lines) ###
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / self.m
        db2 = np.sum(dZ2, axis = 1, keepdims = True) / self.m
        dZ1 = np.multiply(np.dot(W2.T, dZ2) , self.tanh_prime(Z1))
        dW1 = np.dot(dZ1, X.T) / self.m
        db1 = np.sum(dZ1, axis = 1, keepdims = True) / self.m

        return {"dW1":dW1, "db1":db1, "dW2":dW2, "db2":db2}

    def update(self, gradients):
        """
        update the parameters of W1, b1, W2 and b2 using the gradients 
        computed from back propagation and the learning rate alpha
        input: gradients dictionary from back propagation
        """
        # load parameters
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2'] 
        # load gradients from back propagation
        dW1, db1, dW2, db2 = gradients['dW1'], gradients['db1'], gradients['dW2'], gradients['db2'],

        # Update Parameters
        ### Your code here (4 lines) ###
        W1 = W1 - self.learning_rate * dW1
        b1 = b1 - self.learning_rate * db1
        W2 = W2 - self.learning_rate * dW2
        b2 = b2 - self.learning_rate * db2        
        
        self.params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def train(self, X, Y, m, num_iterations = 20000, debug = False):
        self.cost = []
        self.m = m
        assert self.m != 0

        print("Training start. Using learing rate %f"%(self.learning_rate))
        print("="*50)

        for i in range(0, num_iterations):
            self.forward(X)
            #J = self.cross_entropy_loss(Y)
            J = self.multiclass_loss(Y)
            gradients = self.backward(X, Y)
            self.update(gradients)

            self.cost.append(J)
            if debug and i%10==0:
                print("cost after iteration %i: %2.8f"%(i, J))

        print("Training is completed!")
        self.trained = True

    def predict(self, X):
        if self.trained:
            self.forward(X)
            prediction = (self.cache["A2"] > 0.5)
            print("Output is generated.")
            return prediction
        else:
            print("Please train the network first!")
            return None

    def predict_mnist(self, X):
        if self.trained:
            self.forward(X)
            prediction = self.cache["A2"]
            print("Output of MNIST prediction is generated.")
            return prediction
        else:
            print("Please train the network first!")
            return None

    def output(self):
        if self.trained:
            out = open('nn_weights.dat', 'wb')
            pickle.dump(self.params, out)
            print("Successfully save trained weights into file nn_weights.dat")
            return 1
        else:
            print("Please train the network with data first!")
            return 0

    def input(self, weight_file):
        self.params = pickle.load(weight_file)
        self.trained = True

    def getsize(self):
        return self.input_size, self.hidden_size, self.output_size

    def getcostlist(self):
        return self.cost
