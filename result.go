package autofunc

import "github.com/unixpickle/num-analysis/linalg"

// Result represents the output of a function.
type Result interface {
	// Output is the output from the function.
	Output() linalg.Vector

	// PropagateGradient performs back propagation
	// through the function.
	// The upstream argument provides partials of
	// a value with respect to each of the outputs.
	// The gradient of the result is added to grad.
	PropagateGradient(upstream linalg.Vector, grad *Gradient)
}

// RResult is like a Result, but is used propagate
// R-operators rather than plain old gradients.
type RResult interface {
	// Output is the output from the function.
	Output() linalg.Vector

	// ROutput is a vector containing the derivatives
	// of the output components with respect to r.
	ROutput() linalg.Vector

	// PropagateRGradient performs back propagation
	// to compute the RGradient of the output.
	// The upstream argument specifies the gradient
	// of the error function with respect to the output.
	// The upstreamR argument specifies the derivative
	// of upstream with respect to r.
	// The r-gradient is added to grad.
	PropagateRGradient(upstream, upstreamR linalg.Vector, grad RGradient)
}
