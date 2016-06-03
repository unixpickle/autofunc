package autofunc

// A Func is an operation which can be applied to
// the result of any other operation.
// The Func must be able to perform back-propagation
// of gradients.
type Func interface {
	Apply(input Result) Result
}

// An RFunc is like a Func, but it must be able
// to propagate RResults as well as Results.
type RFunc interface {
	Func
	ApplyR(v RVector, input RResult) RResult
}
