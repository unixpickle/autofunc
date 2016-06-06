// Package autofunc performs forward and backward
// automatic differentation, allowing you to
// evaluate functions of millions of variables
// without ever having to worry about computing
// their gradients by hand.
//
// Package autofunc works by providing a suite
// of built-in mathematical functions that it
// knows how to differentiate. As long as you
// stick to these built-in functions (or add
// some of your own), differentation will be
// completely taken care of.
//
// Variables in autofunc are represented by
// *Variable values.
// A Variable is simply a struct containing a
// vector of floating points.
// Embedding the slice inside a struct makes it
// possible to get pointers to the struct and
// thus to use the *Variable as a map key, etc.
//
// All results from computations in autofunc
// implement the Result or RResult interface.
// Both Results and RResults have an output,
// the vector resulting from a computation.
// In addition, an RResult has an ROutput, the
// derivatives of each of the output components
// with respect to a variable r.
// Finally, both a Result and an RResult can
// perform back propagation, computing the
// gradient of some output value with respect to
// all the variables used in the computation.
//
// Objects with an Apply() method that takes a
// Result and returns a Result implement the
// Func interface.
// Similarly, there is an RFunc interface which
// requires an ApplyR() method.
// Package autofunc includes many RFuncs out of
// the box, including LinTran for matrix algebra,
// Exp{} to exponentiate values, etc.
//
// There are other operations which don't behave
// like Funcs or RFuncs.
// For example, autofunc provides a Mul() function
// to multiply two Results component-wise.
//
// Package autofunc uses github.com/gonum/blas to
// optimize various linear algebra operations.
// As a result, it is a very good choice for things
// like neural networks which are matrix-heavy.
//
// Package autofunc makes no attempt to simplify the
// dependency graph of a Result or RResult.
// If a Result x is used multiple times to compute a
// Result y, backpropagation on y will run through x
// multiple times.
//
// To avoid exponential backpropagation work for things
// like Recurrent Neural Networks, you can "pool" the
// gradient with respect to a Result before backpropagating
// through said Result.
// To do this, put the Result you wish to pool into a
// *Variable, then use that variable for all the
// computations that depend on said Result.
// During backpropagation, you would first backpropagate
// with respect to the *Variable, thus accumulating the
// total output gradient with respect to said *Variable.
// Next, you would backpropagate through the Result itself,
// passing the *Variable's gradient as the upstream
// argument of PropagateGradient().
package autofunc
