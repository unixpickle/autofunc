// Package seqfunc provides abstractions for dealing with
// functions that operate on vector sequences.
package seqfunc

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// A Func can map lists of vector sequences (Results) to
// new output sequences in a differentiable manner.
type Func interface {
	ApplySeqs(r Result) Result
}

// An RFunc is like a Func, but can operate on RResults.
type RFunc interface {
	Func
	ApplySeqsR(rv autofunc.RVector, r RResult) RResult
}

// A Result represents a differentiable list of vector
// sequences.
type Result interface {
	// OutputSeqs returns the list of vector sequences.
	OutputSeqs() [][]linalg.Vector

	// PropagateGradient performs back-propagation through
	// the sequences, given an upstream vector which matches
	// the structure of the output sequence list.
	//
	// This should not modify the upstream gradient.
	PropagateGradient(upstream [][]linalg.Vector, g autofunc.Gradient)
}

// An RResult is like a Result, but with forward derivative
// values ("r" values) as well as regular values.
type RResult interface {
	// OutputSeqs returns the list of vector sequences.
	OutputSeqs() [][]linalg.Vector

	// ROutputSeqs returns the derivatives of OutputSeqs
	// with respect to some variable r.
	ROutputSeqs() [][]linalg.Vector

	// PropagateRGradient performs back-propagation.
	// The g parameter may be nil, indicating no desire to
	// compute raw gradients.
	//
	// This should not modify the upstream gradient.
	PropagateRGradient(upstream, upstreamR [][]linalg.Vector,
		rg autofunc.RGradient, g autofunc.Gradient)
}
