# Training an MNIST classifier with numpy

# Motivation

The exercise here is to build out a training loop for an MNIST classifier using only numpy (vs. pytorch, etc.) to gain an appreciation and understanding of how neural networks actually work from the math up.

All of it fits nicely into the [mnist-notebook.ipynb](mnist-notebook.ipynb).

I added some graphs at the bottom of the notebook for flavor.



#### Reference Videos : 
- https://www.youtube.com/watch?v=w8yWXqWQYmU&t=13s
- [3Blue1Brown neural net series](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Karpathy : NN's from Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)


#### Tools
- ![Dataset: MNIST](https://img.shields.io/badge/dataset-MNIST-blue)
- ![NumPy](https://img.shields.io/badge/built%20with-NumPy-013243?logo=numpy)


---



# MATH

## 1.) forward pass

$m$ = the number of examples in our batch

Input Layer : 

$A^{[0]} = X$  

shape(784, m) : 784 rows, m columns


### Activation of Layer 1 : 

$W^{[1]} \in \mathbb{R}^{128 \times 784}$

$A^{[0]} \in \mathbb{R}^{784 \times m}$

$b^{[1]} \in \mathbb{R}^{128 \times 1}$

we have : 

$Z^{[1]} = W^{[1]}A^{[0]} + b^{[1]}$

shapes : ``` (128,784) * (784, m) = (128, m) ``` 

then for the non-linearity : 

$A^{[1]} = g(Z^{[1]}) = ReLU(Z^{[1]})$



### Activation of Layer 2 : 

$Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}$

$A^{[2]} = g(Z^{[1]}) = softmax(Z^{[1]})$


## 2.) back propagation

the derivatives starting from the right (Layer 2) 

$dZ^{[2]} = A^{[2]} - Y$

$dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}$

$db^{[2]} = \frac{1}{m} \sum{dZ^{[2]}}$

$dZ^{[1]} = W^{[2]T} dZ^{[2]} g'(Z^{[1]})$ 

where g' is the derivative of the activation function : ReLU in this case

$dW^{[1]} = \frac{1}{m} dZ^{[1]} X^{T}$

$db^{[1]} = \frac{1}{m} \sum{dZ^{[1]}}$


## 3.) update the parameters

$W^{[1]} = W^{[1]} - \alpha dW^{[1]}$

$b^{[1]} = b^{[1]} - \alpha db^{[1]}$ 

$W^{[2]} = W^{[2]} - \alpha dW^{[2]}$

$b^{[2]} = b^{[2]} - \alpha db^{[2]}$ 


GOTO the [mnist-notebook.ipynb](mnist-notebook.ipynb)