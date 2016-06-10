package autofunc

import "github.com/unixpickle/num-analysis/linalg"

// Result represents the output of an operation.
//
// A Result is only valid so long as the Results,
// Variables, and functions it relies on are not
// modified in any way.
type Result interface {
	// Output is the output from the operation.
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
	//
	// This routine may modify the contents of upstream and
	// upstreamR internally, especially if doing so helps
	// with performance.
	// However, the caller regains ownership of upstream
	// after this returns, so Results should not retain
	// references to their upstream argument.
	PropagateGradient(upstream linalg.Vector, grad Gradient)

	// Release releases any data used by this Result back
	// to the cache from which it was obtained.
	// It also propagates the release signal back to
	// Results on which this Result depends.
	//
	// Release may be called multiple times.
	// Every call after the first should do nothing.
	//
	// You should not use a Result after releasing it unless
	// the Result is a *Variable.
	// Since a Result releases its dependencies, you should
	// take care not to use a Result that has been Released
	// indirectly through some other result.
	//
	// Release is not concurrency-safe (i.e. thread-safe).
	Release()
}

// RResult is like a Result, but each value in an
// RResult is associated with a derivative with
// respect to some variable r.
//
// An RResult is only valid so long as the Results,
// Variables, and functions it relies on are not
// modified in any way.
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
	//
	// This routine may modify the contents of upstream and
	// upstreamR internally, especially if doing so helps
	// with performance.
	// However, the caller regains ownership of upstream and
	// upstreamR after this returns, so RResults should not
	// retain references to their upstream arguments.
	PropagateRGradient(upstream, upstreamR linalg.Vector, rgrad RGradient, grad Gradient)

	// Release is like Result.Release().
	Release()
}
