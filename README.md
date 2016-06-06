# autofunc [![GoDoc](https://godoc.org/github.com/unixpickle/autofunc?status.svg)](https://godoc.org/github.com/unixpickle/autofunc)

This package makes it easy to compute gradients of complicated, deeply nested functions. It is designed for Machine Learning, wherein it is common practice to compute gradients of complex Neural Networks.

In addition to basic [backpropagation](https://en.wikipedia.org/wiki/Backpropagation), a form of reverse automatic differentiation, **autofunc** can perform forward automatic differentiation with respect to one variable, an operation known as the [R operator](http://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf). As a result, **autofunc** is suitable for approximating various aspects of a function's Hessian, such as the Hessian's rows' magnitudes (as suggested in [this paper](http://arxiv.org/pdf/1502.04390v2.pdf)).

# Usage

Installation is easy:

```
$ go get github.com/unixpickle/autofunc
```

To see how you might create something like a Multilayer Perceptron network, checkout [bench/mlp.go](bench/mlp.go).
