package autofunc

// A Func is an operation which can be applied to
// the result of any other operation.
// The Func must be able to perform back-propagation
// of gradients.
type Func interface {
	Apply(input Result) Result
}

// A ComposedFunc feeds input from each Func
// to the next Func in a list.
type ComposedFunc []Func

func (c ComposedFunc) Apply(input Result) Result {
	for _, f := range c {
		input = f.Apply(input)
	}
	return input
}

// An RFunc is like a Func, but it must be able
// to propagate RResults as well as Results.
type RFunc interface {
	Func
	ApplyR(v RVector, input RResult) RResult
}

// A ComposedRFunc is like a ComposedFunc, but
// for RFuncs.
type ComposedRFunc []RFunc

func (c ComposedRFunc) Apply(input Result) Result {
	for _, f := range c {
		input = f.Apply(input)
	}
	return input
}

func (c ComposedRFunc) ApplyR(v RVector, input RResult) RResult {
	for _, f := range c {
		input = f.ApplyR(v, input)
	}
	return input
}
