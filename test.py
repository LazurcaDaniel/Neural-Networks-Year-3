import numpy as np
from torchvision.datasets import MNIST
from timed_decorator.simple_timed import timed



def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform = lambda x: np.array(x).flatten() / 255.0,
                    download = False,
                    train = is_train)
    mnist_data=[]
    mnist_labels=[]
    num_classes = 10
    for image,label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return np.array(mnist_data,dtype=np.float16), np.eye(num_classes)[np.array(mnist_labels,dtype=np.uint8)]





train_X, train_Y = download_mnist(True) 
"""
Train_X.shape = (60_000,784)
wheights.shape = (10,784)
Bias.shape = (10,)
Train_Y = lable-uri
"""

def process_batch(batch_x, batch_y, W_batch, Bias_batch, alpha=0.01):
    w_sum = (W_batch @ batch_x.T) + Bias_batch[:, np.newaxis]  
    
    exp_sum = np.exp(w_sum - np.max(w_sum, axis=0))
    output = exp_sum / np.sum(exp_sum, axis=0)  

    gradient = output - batch_y.T

    W_batch += alpha * gradient @ batch_x
    Bias_batch += alpha * np.sum(gradient, axis=1) 
    
    return W_batch, Bias_batch

@timed(use_seconds=True, show_args=True)
def perceptron(train_X, train_Y, alpha = 0.01):
    W = np.random.rand(10,784) * 0.01
    Bias = np.random.rand(10)   
    error_rate = 0.94
    batch_size = 100
    num_batches = len(train_X) // batch_size  # Calculate the number of full batches
    data_batches = np.array(np.array_split(train_X[:num_batches * batch_size], num_batches))
    label_batches = np.array(np.array_split(train_Y[:num_batches * batch_size], num_batches))
    for epoch in range(300):
        print(f'Epoch {epoch + 1}')
        for x,y in zip(data_batches, label_batches):
            W, Bias = process_batch(x, y, W, Bias, alpha)
        alpha *= error_rate
    return W, Bias
    
def predict(X,W,Bias):
    w_sum = W @ X + Bias
    output = np.exp(w_sum - np.max(w_sum,axis = 0)) / np.sum(np.exp(w_sum) - np.max(w_sum,axis = 0), axis = 0)
    return np.argmax(output)

test_X, test_Y = download_mnist(False)

w,bias = perceptron(train_X, train_Y)

predictions = [predict(x,w,bias) for x in test_X]
correct_predictions = sum([pred == true for pred, true in zip(predictions, test_Y)])
accuracy = correct_predictions / len(test_Y) * 100
print(sum(accuracy)/10)
with open('w_b.txt', 'w') as file:
    file.write(f'{w}')
    file.write(f'{bias}')
    file.write(f'{accuracy}')