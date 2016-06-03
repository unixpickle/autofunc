# autofunc

This package makes it easy to compute gradients of complicated, deeply nested functions. It is designed for Machine Learning, wherein it is common to represent Neural Networks as mathematical functions and take their derivatives.

In addition to basic [backpropagation](https://en.wikipedia.org/wiki/Backpropagation), a form of reverse automatic differentiation, **autofunc** can perform forward automatic differentiation with respect to one variable, an operation known as the [R operator](http://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf). As a result, **autofunc** is suitable for approximating various aspects of a function's Hessian, such as the Hessian's rows' magnitudes (as suggested in [this paper](http://arxiv.org/pdf/1502.04390v2.pdf)).

# Automatic Differentiation

Many Machine Learning models (specifically Neural Networks) are mathematical functions. To train these models, people traditionally use some improved version of [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent). To use SGD or a variant thereof, one must be able to compute gradients of large functions with respect to millions of variables. This is done via .
