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

// mulTwiceSeq is a seqfunc.RFunc which multiplies an
// input by itself, thus getting a non-zero r-upstream.
type mulTwiceSeq struct{}

func (_ mulTwiceSeq) ApplySeqs(r seqfunc.Result) seqfunc.Result {
	var out [][]linalg.Vector
	for _, outSeq := range r.OutputSeqs() {
		var newOut []linalg.Vector
		for _, outVec := range outSeq {
			resVec := make(linalg.Vector, len(outVec))
			for i, x := range outVec {
				resVec[i] = x * x
			}
			newOut = append(newOut, resVec)
		}
		out = append(out, newOut)
	}
	return &mulTwiceResult{
		Output: out,
		Input:  r,
	}
}

func (_ mulTwiceSeq) ApplySeqsR(rv autofunc.RVector, r seqfunc.RResult) seqfunc.RResult {
	var out [][]linalg.Vector
	var outR [][]linalg.Vector
	for i, outSeq := range r.OutputSeqs() {
		var newOut []linalg.Vector
		var newOutR []linalg.Vector
		for j, outVec := range outSeq {
			resVec := make(linalg.Vector, len(outVec))
			resVecR := make(linalg.Vector, len(outVec))
			for k, x := range outVec {
				resVec[i] = x * x
				resVecR[i] = 2 * x * r.ROutputSeqs()[i][j][k]
			}
			newOut = append(newOut, resVec)
			newOutR = append(newOutR, resVecR)
		}
		out = append(out, newOut)
		outR = append(outR, newOutR)
	}
	return &mulTwiceRResult{
		Output:  out,
		ROutput: outR,
		Input:   r,
	}
}

type mulTwiceResult struct {
	Output [][]linalg.Vector
	Input  seqfunc.Result
}

func (a *mulTwiceResult) OutputSeqs() [][]linalg.Vector {
	return a.Output
}

func (a *mulTwiceResult) PropagateGradient(u [][]linalg.Vector, g autofunc.Gradient) {
	var newUpstream [][]linalg.Vector
	for i, seq := range u {
		var upSeq []linalg.Vector
		for j, vec := range seq {
			newVec := make(linalg.Vector, len(vec))
			for k, x := range vec {
				newVec[k] = x * a.Input.OutputSeqs()[i][j][k]
			}
			upSeq = append(upSeq, newVec)
		}
		newUpstream = append(newUpstream, upSeq)
	}
	for i := 0; i < 2; i++ {
		a.Input.PropagateGradient(newUpstream, g)
	}
}

type mulTwiceRResult struct {
	Output  [][]linalg.Vector
	ROutput [][]linalg.Vector
	Input   seqfunc.RResult
}

func (a *mulTwiceRResult) OutputSeqs() [][]linalg.Vector {
	return a.Output
}

func (a *mulTwiceRResult) ROutputSeqs() [][]linalg.Vector {
	return a.ROutput
}

func (a *mulTwiceRResult) PropagateRGradient(u, uR [][]linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	var newUpstream [][]linalg.Vector
	var newUpstreamR [][]linalg.Vector
	for i, seq := range u {
		var upSeq []linalg.Vector
		var upSeqR []linalg.Vector
		for j, vec := range seq {
			newVec := make(linalg.Vector, len(vec))
			newVecR := make(linalg.Vector, len(vec))
			for k, x := range vec {
				newVec[k] = x * a.Input.OutputSeqs()[i][j][k]
				newVecR[k] = x*a.Input.ROutputSeqs()[i][j][k] +
					uR[i][j][k]*a.Input.OutputSeqs()[i][j][k]
			}
			upSeq = append(upSeq, newVec)
			upSeqR = append(upSeqR, newVecR)
		}
		newUpstream = append(newUpstream, upSeq)
		newUpstreamR = append(newUpstreamR, upSeqR)
	}
	for i := 0; i < 2; i++ {
		a.Input.PropagateRGradient(newUpstream, newUpstreamR, rg, g)
	}
}
