import scipy.io

def fetch(filename):
    mat = scipy.io.loadmat(filename)
    Y_train = mat['Ct'].T
    X_train = mat['Yt']
    Y_test = mat['Cv'].T
    X_test = mat['Yv']
    return Y_train, X_train, Y_test, X_test