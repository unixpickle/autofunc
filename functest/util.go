package functest

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
)

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

// addTwiceSeq is a seqfunc.RFunc which adds an input to
// itself, thus baking it possible to test back-prop
// gradient accumulation.
type addTwiceSeq struct{}

func (_ addTwiceSeq) ApplySeqs(r seqfunc.Result) seqfunc.Result {
	var out [][]linalg.Vector
	for _, outSeq := range r.OutputSeqs() {
		var newOut []linalg.Vector
		for _, outVec := range outSeq {
			newOut = append(newOut, outVec.Copy().Scale(2))
		}
		out = append(out, newOut)
	}
	return &addTwiceResult{
		Output: out,
		Input:  r,
	}
}

func (_ addTwiceSeq) ApplySeqsR(rv autofunc.RVector, r seqfunc.RResult) seqfunc.RResult {
	var out [][]linalg.Vector
	var outR [][]linalg.Vector
	for i, outSeq := range r.OutputSeqs() {
		var newOut []linalg.Vector
		var newOutR []linalg.Vector
		for j, outVec := range outSeq {
			newOut = append(newOut, outVec.Copy().Scale(2))
			newOutR = append(newOutR, r.ROutputSeqs()[i][j].Copy().Scale(2))
		}
		out = append(out, newOut)
		outR = append(outR, newOutR)
	}
	return &addTwiceRResult{
		Output:  out,
		ROutput: outR,
		Input:   r,
	}
}

type addTwiceResult struct {
	Output [][]linalg.Vector
	Input  seqfunc.Result
}

func (a *addTwiceResult) OutputSeqs() [][]linalg.Vector {
	return a.Output
}

func (a *addTwiceResult) PropagateGradient(u [][]linalg.Vector, g autofunc.Gradient) {
	for i := 0; i < 2; i++ {
		a.Input.PropagateGradient(u, g)
	}
}

type addTwiceRResult struct {
	Output  [][]linalg.Vector
	ROutput [][]linalg.Vector
	Input   seqfunc.RResult
}

func (a *addTwiceRResult) OutputSeqs() [][]linalg.Vector {
	return a.Output
}

func (a *addTwiceRResult) ROutputSeqs() [][]linalg.Vector {
	return a.ROutput
}

func (a *addTwiceRResult) PropagateRGradient(u, uR [][]linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	for i := 0; i < 2; i++ {
		a.Input.PropagateRGradient(u, uR, rg, g)
	}
}
