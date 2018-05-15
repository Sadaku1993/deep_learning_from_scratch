#coding:utf-8

from layers import *

relu = Relu()
sigmoid = Sigmoid()

x = np.array([-0.2, 0.4, 0.5, 0.2])
dout = np.array([0.2, 0.3, -0.2, -0.3])

relu_forward = relu.forward(x)
relu_backward = relu.backward(dout)

sigmoid_forward = sigmoid.forward(x)
sigmoid_backward = sigmoid.backward(x)

print("x : "+str(x))
print("relu F : "+str(relu_forward))
print("relu B : "+str(relu_backward))
print("sigm F : "+str(sigmoid_forward))
print("sigm B : "+str(sigmoid_backward))

