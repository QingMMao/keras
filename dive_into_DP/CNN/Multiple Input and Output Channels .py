from mxnet import nd
import Tools

def corr2d_multi_in(X,K):
    return nd.add_n(*[Tools.corr2d(x,k) for x,k in zip(X,K)])

X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    
K = nd.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

corr2d_multi_in(X, K)

def corr2d_multi_in_out(X,K):
    return nd.stack(*[corr2d_multi_in(X,k) for k in K])

K = nd.stack(K,K+1,K+2)

def corr2d_multi_in_out_1x1(X,K):
    c_i,h,w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i,h*w))
    K = K.reshape((c_o,c_i))
    Y = nd.dot(K,X)
    return Y.reshape((c_o,h,w))

X = nd.random.uniform(shape=(3, 3, 3))
K = nd.random.uniform(shape=(2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

#Y1==Y2:True