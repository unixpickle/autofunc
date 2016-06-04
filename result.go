package autofunc

import "github.com/unixpickle/num-analysis/linalg"

// Result represents the output of a function.
type Result interface {
	// Output is the output from the function.
	Output() linalg.Vector

	// Constant returns true if this Result is
	// constant with respect to all the variables
	// in a given Gradient.
	Constant(g Gradient) bool

	// PropagateGradient performs back propagation
	// through the function.
	// The upstream argument provides partials of
	// a value with respect to each of the outputs.
	// The gradient of the result is added to grad.
	PropagateGradient(upstream linalg.Vector, grad Gradient)
}

// RResult is like a Result, but is used propagate
// R-operators rather than plain old gradients.
type RResult interface {
	// Output is the output from the function.
	Output() linalg.Vector

	// ROutput is a vector containing the derivatives
	// of the output components with respect to r.
	ROutput() linalg.Vector

	// Constant returns true if this Result is
	// constant with respect to all the variables
	// in both rg and g.
	//
	// The g argument may be nil, in which case it
	// is completely ignored.
	Constant(rg RGradient, g Gradient) bool

	// PropagateRGradient performs back propagation
	// to compute the RGradient of the output.
	// The upstream argument specifies the gradient
	// of the error function with respect to the output.
	// The upstreamR argument specifies the derivative
	// of upstream with respect to r.
	// The r-gradient is added to rgrad.
	//
	// If the grad argument is non-nil, it will be updated
	// like the grad argument for Result.PropagateGradient,
	// thus taking advantage of the fact that r-propagation
	// could easily compute the regular gradient as well.
	PropagateRGradient(upstream, upstreamR linalg.Vector, rgrad RGradient, grad Gradient)
}
