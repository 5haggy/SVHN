import matplotlib.pyplot as plt

def plot_weights(weights, dim):
    n=10
    plt.figure(figsize=(30, 8))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(weights[i].reshape(dim, dim))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(weights[i+n].reshape(dim, dim))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(weights[n+n+i].reshape(dim, dim))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()