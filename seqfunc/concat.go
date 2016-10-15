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
