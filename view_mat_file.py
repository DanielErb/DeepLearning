import scipy.io
mat = scipy.io.loadmat('GMMData.mat')
print(mat)
x = mat['Cv']
print(x[1])
print(x.shape)