import numpy as np
from torchvision.datasets import MNIST
from timed_decorator.simple_timed import timed

def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten() / 255.0,
                    download=False,
                    train=is_train)
    mnist_data, mnist_labels = [], []
    num_classes = 10
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return np.array(mnist_data, dtype=np.float32), np.eye(num_classes)[np.array(mnist_labels, dtype=np.uint8)]


W1 = np.random.randn(784, 100).astype(np.float32) * 0.01
B1 = np.zeros((100,), dtype=np.float32)
W2 = np.random.randn(100, 10).astype(np.float32) * 0.01
B2 = np.zeros((10,), dtype=np.float32)

train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)


def relu(x):
    return np.maximum(0, x)

def softmax_cross_entropy_loss(z2, Y):
    z2_shifted = z2 - np.max(z2, axis=1, keepdims=True) #shifting so that we don't get overflow with values 
    exp_z2 = np.exp(z2_shifted) #e^z
    softmax_output = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True) #e^z / sum(e^z) - softmax
    loss = -np.sum(Y * np.log(softmax_output + 1e-8)) / Y.shape[0] #Cost function: -1/N * sum(expected_values * softmax(computed_values))

    return loss, softmax_output

def dropout(X, drop_prob):
    """Apply dropout to the input data X."""
    if drop_prob < 0.0 or drop_prob >= 1.0:
        raise ValueError("drop_prob must be in the range [0, 1)")
    mask = (np.random.rand(*X.shape) > drop_prob).astype(np.float32) #creating a matrix of shape X with random values. Values < drop_prob -> 0, values > drop_prob -> 1
    return mask * X / (1.0 - drop_prob) #dividing to increase the values in the Neurons because they would be smaller than normal and mess with learning 

def forward_propagation(X, Y, drop_prob=0.01, train=True):
    global W1, B1, W2, B2
    z1 = X @ W1 + B1 #values for hidden layer
    h = relu(z1) #relu(values for hidden layer)

    if train:
        h = dropout(h, drop_prob) #applying dropout

    z2 = h @ W2 + B2 #values for output layer
    loss, out = softmax_cross_entropy_loss(z2, Y) #loss, computed_values(softmax(z2))
    return z1, h, z2, out, loss

def backwards_propagation(X, Y, z1, h, out, alpha=0.01):
    global W1, B1, W2, B2
    N = Y.shape[0]
    dz2 = out - Y   
    dW2 = h.T @ dz2 / N
    dB2 = np.sum(dz2, axis=0) / N
    
    dz1 = (dz2 @ W2.T) * (z1 > 0)
    dW1 = X.T @ dz1 / N
    dB1 = np.sum(dz1, axis=0) / N

    W2 -= alpha * dW2
    B2 -= alpha * dB2
    W1 -= alpha * dW1
    B1 -= alpha * dB1

@timed(use_seconds=True, show_args=True)
def training(X, Y, alpha=0.01, epochs=85, batch_size=100, drop_prob=0.01):
    num_batches = X.shape[0] // batch_size 
    for epoch in range(epochs):
        for i in range(num_batches):
            X_batch = X[i*batch_size:(i+1)*batch_size]
            Y_batch = Y[i*batch_size:(i+1)*batch_size]
            z1, h, z2, out, loss = forward_propagation(X_batch, Y_batch, drop_prob, train=True)
            backwards_propagation(X_batch, Y_batch, z1, h, out, alpha)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")


def test(test_X, test_Y):
    _, _, _, out, _ = forward_propagation(test_X, test_Y, train=False)
    predictions = np.argmax(out, axis=1)
    true_labels = np.argmax(test_Y, axis=1)
    accuracy = np.mean(predictions == true_labels) * 100
    print(f'Accuracy: {accuracy:.2f}%')

# Train and test
training(train_X, train_Y)
test(test_X, test_Y)
