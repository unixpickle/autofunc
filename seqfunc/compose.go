package seqfunc

import "github.com/unixpickle/autofunc"

// A ComposedFunc applies a set of Funcs in order.
type ComposedFunc []Func

// ApplySeqs applies the functions in c in order, starting
// with the first one.
func (c ComposedFunc) ApplySeqs(in Result) Result {
	for _, f := range c {
		in = f.ApplySeqs(in)
	}
	return in
}

// A ComposedRFunc applies a set of RFuncs in order.
type ComposedRFunc []RFunc

// ApplySeqs applies the functions in c in order, starting
// with the first one.
func (c ComposedRFunc) ApplySeqs(in Result) Result {
	for _, f := range c {
		in = f.ApplySeqs(in)
	}
	return in
}

// ApplySeqsR applies the functions in c in order,
// starting with the first one.
func (c ComposedRFunc) ApplySeqsR(rv autofunc.RVector, in RResult) RResult {
	for _, f := range c {
		in = f.ApplySeqsR(rv, in)
	}
	return in
}
