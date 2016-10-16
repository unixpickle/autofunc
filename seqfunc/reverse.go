package seqfunc

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// Reverse reverses the sequences inside a sequence list,
// but does not change the order in which the sequences
// themselves appear in the list.
func Reverse(in Result) Result {
	return &reverseResult{
		Input:  in,
		Output: reverseSequences(in.OutputSeqs()),
	}
}

// ReverseR is like Reverse for RResults.
func ReverseR(in RResult) RResult {
	return &reverseRResult{
		Input:   in,
		Output:  reverseSequences(in.OutputSeqs()),
		ROutput: reverseSequences(in.ROutputSeqs()),
	}
}

type reverseResult struct {
	Input  Result
	Output [][]linalg.Vector
}

func (r *reverseResult) OutputSeqs() [][]linalg.Vector {
	return r.Output
}

func (r *reverseResult) PropagateGradient(u [][]linalg.Vector, g autofunc.Gradient) {
	r.Input.PropagateGradient(reverseSequences(u), g)
}

type reverseRResult struct {
	Input   RResult
	Output  [][]linalg.Vector
	ROutput [][]linalg.Vector
}

func (r *reverseRResult) OutputSeqs() [][]linalg.Vector {
	return r.Output
}

func (r *reverseRResult) ROutputSeqs() [][]linalg.Vector {
	return r.ROutput
}

func (r *reverseRResult) PropagateRGradient(u, uR [][]linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	r.Input.PropagateRGradient(reverseSequences(u), reverseSequences(uR), rg, g)
}

func reverseSequences(u [][]linalg.Vector) [][]linalg.Vector {
	reversed := make([][]linalg.Vector, len(u))
	for i, seq := range u {
		reversed[i] = make([]linalg.Vector, len(seq))
		for j, x := range seq {
			reversed[i][len(seq)-(j+1)] = x
		}
	}
	return reversed
}
