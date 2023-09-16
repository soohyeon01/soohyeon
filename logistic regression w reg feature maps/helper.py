import numpy as np
import matplotlib.pyplot as plt

def plot_exam(X, y, theta):

    b = theta[0]
    w1, w2 = theta[1], theta[2]
    c = -b/w1
    m = -w2/w1

    x1min, x1max = 30, 100
    x2min, x2max = 30, 100
    x1d = np.array([x1min, x1max])
    x2d = m*x1d + c
    
    plt.rcParams["figure.figsize"] = (10,8)

    colormap = np.array(['tab:blue', 'tab:red'])
    plt.plot(x1d, x2d, 'k', lw=1, ls='--')
    plt.scatter(X[:,1], X[:,2], c=colormap[y.astype(int)].reshape(-1))
    plt.xlabel('Exam 1')
    plt.ylabel('Exam 2')
    
    
def plot_micro(X, y, theta, degree):

    num_samp = 50
    xrange = np.linspace(-1, 1.5, num_samp)

    Z = np.zeros((num_samp, num_samp))
    for i, u in enumerate(xrange):
        for j, v in enumerate(xrange):
            Z[i, j] = np.matmul(mapFeature(np.array([[u, v]]), degree), theta)

    #Z = np.around(Z, decimals=1)
    xx1, xx2 = np.meshgrid(xrange,xrange)

    plt.rcParams["figure.figsize"] = (5,5)

    colormap = np.array(['tab:blue', 'tab:red'])
    plt.contour(xx1, xx2, Z, colors='g', levels=[0], alpha=0.5,
                   linestyles=['-'])
    plt.scatter(X[:,1], X[:,2], c=colormap[y.astype(int)].reshape(-1))
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    
def plot_micro2(X, y, theta, degree):

    num_samp = 50
    xmin = -2.1
    xmax = 2.1
    xrange = np.linspace(xmin, xmax, num_samp)

    Z = np.zeros((num_samp, num_samp))
    for i, u in enumerate(xrange):
        for j, v in enumerate(xrange):
            Z[i, j] = np.matmul(mapFeature(np.array([[u, v]]), degree), theta)

    #Z = np.around(Z, decimals=1)
    xx1, xx2 = np.meshgrid(xrange,xrange)
    xx1 = xx1.reshape(-1)
    xx2 = xx2.reshape(-1)
    plt.rcParams["figure.figsize"] = (5,5)
    Zlabel = np.zeros_like(Z)
    Zlabel[Z>0] = 1
    Zlabel = Zlabel.reshape(-1)
    colormap = np.array(['tab:blue', 'tab:red'])
    plt.scatter(xx1, xx2, c=colormap[Zlabel.astype(int)].reshape(-1), alpha=0.1)
    plt.scatter(X[:,1], X[:,2], c=colormap[y.astype(int)].reshape(-1))
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')

    
def mapFeature(X, degree):
    output = []
    for i in range(degree+1):
        for j in range(int(i)+1):
            output.append((X[:,0]**(i-j))*(X[:,1]**(j)))
            
    return np.array(output).T