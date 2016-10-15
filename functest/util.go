package functest

import "github.com/unixpickle/autofunc"

// AddTwice adds an input to itself.
// This makes back-propagation propagate through the input
// twice, thus testing that gradients are properly added
// to rather than being overwritten.
type addTwice struct{}

func (_ addTwice) Apply(r autofunc.Result) autofunc.Result {
	return autofunc.Add(r, r)
}

func (_ addTwice) ApplyR(v autofunc.RVector, r autofunc.RResult) autofunc.RResult {
	return autofunc.AddR(r, r)
}

// mulTwice multiplies an input by itself.
//
// This is useful for testing back-propagation with a
// non-zero upstreamR parameter.
// Since the upstream derivative for x in x*y is y, the
// upstream derivative's R-derivative is the R-derivative
// of y, which may be non-zero.
type mulTwice struct{}

func (_ mulTwice) Apply(r autofunc.Result) autofunc.Result {
	return autofunc.Mul(r, r)
}

func (_ mulTwice) ApplyR(v autofunc.RVector, r autofunc.RResult) autofunc.RResult {
	return autofunc.MulR(r, r)
}
