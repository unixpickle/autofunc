package seqfunc

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

type concatInnerResult struct {
	Inputs []Result
	Output [][]linalg.Vector
}

// ConcatInner joins the vectors in lists of sequences
// to one another.
//
// The sequence lists must have the same shapes, i.e.
// the same number of sequences and the same number of
// vectors per sequence.
// However, the sizes of the vectors may differ.
//
// Each vector v in the result will be v1+v2+...+vn where
// vi is the vector from the i-th Result.
func ConcatInner(seqs ...Result) Result {
	if len(seqs) == 0 {
		return ConstResult(nil)
	}
	out := copySeqs(seqs[0].OutputSeqs())
	for _, x := range seqs[1:] {
		if !shapesEqual(out, x.OutputSeqs()) {
			panic("input shapes do not match")
		}
		appendSeqs(out, x.OutputSeqs())
	}
	return &concatInnerResult{
		Inputs: seqs,
		Output: out,
	}
}

func (c *concatInnerResult) OutputSeqs() [][]linalg.Vector {
	return c.Output
}

func (c *concatInnerResult) PropagateGradient(u [][]linalg.Vector, g autofunc.Gradient) {
	upstreams := make([][][]linalg.Vector, len(c.Inputs))
	for i := range upstreams {
		upstreams[i] = make([][]linalg.Vector, len(u))
		for j, seq := range u {
			upstreams[i][j] = make([]linalg.Vector, len(seq))
		}
	}

	for i, seq := range u {
		for j, vec := range seq {
			var start int
			for inIdx, in := range c.Inputs {
				inVec := in.OutputSeqs()[i][j]
				upstreams[inIdx][i][j] = vec[start : start+len(inVec)]
				start += len(inVec)
			}
			if start != len(vec) {
				panic("invalid upstream vector length")
			}
		}
	}

	for i, ups := range upstreams {
		c.Inputs[i].PropagateGradient(ups, g)
	}
}

type concatInnerRResult struct {
	Inputs  []RResult
	Output  [][]linalg.Vector
	ROutput [][]linalg.Vector
}

// ConcatInnerR is like ConcatInner but for RResults.
func ConcatInnerR(seqs ...RResult) RResult {
	if len(seqs) == 0 {
		return ConstRResult(nil)
	}
	out := copySeqs(seqs[0].OutputSeqs())
	outR := copySeqs(seqs[0].ROutputSeqs())
	for _, x := range seqs[1:] {
		if !shapesEqual(out, x.OutputSeqs()) {
			panic("input shapes do not match")
		}
		appendSeqs(out, x.OutputSeqs())
		appendSeqs(outR, x.ROutputSeqs())
	}
	return &concatInnerRResult{
		Inputs:  seqs,
		Output:  out,
		ROutput: outR,
	}
}

func (c *concatInnerRResult) OutputSeqs() [][]linalg.Vector {
	return c.Output
}

func (c *concatInnerRResult) ROutputSeqs() [][]linalg.Vector {
	return c.ROutput
}

func (c *concatInnerRResult) PropagateRGradient(u, uR [][]linalg.Vector,
	rg autofunc.RGradient, g autofunc.Gradient) {
	upstreams := make([][][]linalg.Vector, len(c.Inputs))
	upstreamsR := make([][][]linalg.Vector, len(c.Inputs))
	for i := range upstreams {
		upstreams[i] = make([][]linalg.Vector, len(u))
		upstreamsR[i] = make([][]linalg.Vector, len(u))
		for j, seq := range u {
			upstreams[i][j] = make([]linalg.Vector, len(seq))
			upstreamsR[i][j] = make([]linalg.Vector, len(seq))
		}
	}

	for i, seq := range u {
		for j, vec := range seq {
			rVec := uR[i][j]

			var start int
			for inIdx, in := range c.Inputs {
				inVec := in.OutputSeqs()[i][j]
				upstreams[inIdx][i][j] = vec[start : start+len(inVec)]
				upstreamsR[inIdx][i][j] = rVec[start : start+len(inVec)]
				start += len(inVec)
			}

			if start != len(vec) {
				panic("invalid upstream vector length")
			}
		}
	}

	for i, ups := range upstreams {
		c.Inputs[i].PropagateRGradient(ups, upstreamsR[i], rg, g)
	}
}

type concatAllResult struct {
	Input  Result
	OutVec linalg.Vector
}

// ConcatAll joins all of the timesteps in all of the
// sequences in to one packed autofunc.Result.
// The packing is done as follows: first timesteps from
// the same sequence are packed left to right, then the
// packed vectors from each sequence are joined together
// from the first sequence to the last.
func ConcatAll(in Result) autofunc.Result {
	var joined linalg.Vector
	for _, seq := range in.OutputSeqs() {
		for _, vec := range seq {
			joined = append(joined, vec...)
		}
	}
	return &concatAllResult{Input: in, OutVec: joined}
}

func (a *concatAllResult) Output() linalg.Vector {
	return a.OutVec
}

func (a *concatAllResult) Constant(g autofunc.Gradient) bool {
	return false
}

func (a *concatAllResult) PropagateGradient(u linalg.Vector, g autofunc.Gradient) {
	var idx int
	var splitUpstream [][]linalg.Vector
	for _, outSeq := range a.Input.OutputSeqs() {
		var splitSeq []linalg.Vector
		for _, step := range outSeq {
			splitSeq = append(splitSeq, u[idx:idx+len(step)])
			idx += len(step)
		}
		splitUpstream = append(splitUpstream, splitSeq)
	}
	a.Input.PropagateGradient(splitUpstream, g)
}

type concatAllRResult struct {
	Input   RResult
	OutVec  linalg.Vector
	ROutVec linalg.Vector
}

// ConcatAllR is like ConcatAll for RResults.
func ConcatAllR(in RResult) autofunc.RResult {
	var joined, joinedR linalg.Vector
	rOut := in.ROutputSeqs()
	for i, seq := range in.OutputSeqs() {
		for j, vec := range seq {
			joined = append(joined, vec...)
			joinedR = append(joinedR, rOut[i][j]...)
		}
	}
	return &concatAllRResult{Input: in, OutVec: joined, ROutVec: joinedR}
}

func (a *concatAllRResult) Output() linalg.Vector {
	return a.OutVec
}

func (a *concatAllRResult) ROutput() linalg.Vector {
	return a.ROutVec
}

func (a *concatAllRResult) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	return false
}

func (a *concatAllRResult) PropagateRGradient(u, uR linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	var idx int
	var splitUpstream, splitUpstreamR [][]linalg.Vector
	for _, outSeq := range a.Input.OutputSeqs() {
		var splitSeq, splitSeqR []linalg.Vector
		for _, step := range outSeq {
			splitSeq = append(splitSeq, u[idx:idx+len(step)])
			splitSeqR = append(splitSeqR, uR[idx:idx+len(step)])
			idx += len(step)
		}
		splitUpstream = append(splitUpstream, splitSeq)
		splitUpstreamR = append(splitUpstreamR, splitSeqR)
	}
	a.Input.PropagateRGradient(splitUpstream, splitUpstreamR, rg, g)
}

type concatLastResult struct {
	OutVec linalg.Vector
	In     Result
}

// ConcatLast concatenates the last timestep outputs of
// all the sequences.
//
// Empty input sequences are ignored.
func ConcatLast(in Result) autofunc.Result {
	var joined linalg.Vector
	for _, x := range in.OutputSeqs() {
		if len(x) > 0 {
			joined = append(joined, x[len(x)-1]...)
		}
	}
	return &concatLastResult{
		OutVec: joined,
		In:     in,
	}
}

func (c *concatLastResult) Output() linalg.Vector {
	return c.OutVec
}

func (c *concatLastResult) Constant(g autofunc.Gradient) bool {
	return false
}

func (c *concatLastResult) PropagateGradient(u linalg.Vector, g autofunc.Gradient) {
	up := make([][]linalg.Vector, len(c.In.OutputSeqs()))
	var idx int
	for i, seq := range c.In.OutputSeqs() {
		up[i] = make([]linalg.Vector, len(seq))
		if len(seq) == 0 {
			continue
		}
		for j, x := range seq[:len(seq)-1] {
			up[i][j] = make(linalg.Vector, len(x))
		}
		last := seq[len(seq)-1]
		up[i][len(seq)-1] = u[idx : idx+len(last)]
		idx += len(last)
	}
	c.In.PropagateGradient(up, g)
}

type concatLastRResult struct {
	OutVec  linalg.Vector
	ROutVec linalg.Vector
	In      RResult
}

// ConcatLastR is like ConcatLast for RResults.
func ConcatLastR(in RResult) autofunc.RResult {
	var joined, joinedR linalg.Vector
	outSeqsR := in.ROutputSeqs()
	for i, x := range in.OutputSeqs() {
		if len(x) > 0 {
			xR := outSeqsR[i]
			joined = append(joined, x[len(x)-1]...)
			joinedR = append(joinedR, xR[len(xR)-1]...)
		}
	}
	return &concatLastRResult{
		OutVec:  joined,
		ROutVec: joinedR,
		In:      in,
	}
}

func (c *concatLastRResult) Output() linalg.Vector {
	return c.OutVec
}

func (c *concatLastRResult) ROutput() linalg.Vector {
	return c.ROutVec
}

func (c *concatLastRResult) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	return false
}

func (c *concatLastRResult) PropagateRGradient(u, uR linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	up := make([][]linalg.Vector, len(c.In.OutputSeqs()))
	upR := make([][]linalg.Vector, len(c.In.OutputSeqs()))
	var idx int
	for i, seq := range c.In.OutputSeqs() {
		up[i] = make([]linalg.Vector, len(seq))
		upR[i] = make([]linalg.Vector, len(seq))
		if len(seq) == 0 {
			continue
		}
		for j, x := range seq[:len(seq)-1] {
			up[i][j] = make(linalg.Vector, len(x))
			upR[i][j] = make(linalg.Vector, len(x))
		}
		last := seq[len(seq)-1]
		up[i][len(seq)-1] = u[idx : idx+len(last)]
		upR[i][len(seq)-1] = uR[idx : idx+len(last)]
		idx += len(last)
	}
	c.In.PropagateRGradient(up, upR, rg, g)
}

func shapesEqual(s1, s2 [][]linalg.Vector) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, x := range s1 {
		if len(x) != len(s2[i]) {
			return false
		}
	}
	return true
}

func copySeqs(seqs [][]linalg.Vector) [][]linalg.Vector {
	res := make([][]linalg.Vector, len(seqs))
	for i, seq := range seqs {
		res[i] = make([]linalg.Vector, len(seq))
		for j, vec := range seq {
			res[i][j] = make(linalg.Vector, len(vec))
			copy(res[i][j], vec)
		}
	}
	return res
}

func appendSeqs(seqs, addition [][]linalg.Vector) {
	for i, seq := range seqs {
		for j := range seq {
			seq[j] = append(seq[j], addition[i][j]...)
		}
	}
}
