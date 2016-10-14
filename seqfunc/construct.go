package seqfunc

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// ConstResult constructs a Result with the given constant
// output seqs.
// The input is not copied, so it should not be modified
// as long as the result needs to remain valid.
func ConstResult(vecs [][]linalg.Vector) Result {
	return &constResult{Output: vecs}
}

// ConstRResult is like ConstResult, but produces an
// RResult with zero r-outputs.
func ConstRResult(vecs [][]linalg.Vector) RResult {
	zeroOut := make([][]linalg.Vector, len(vecs))
	for i, s := range vecs {
		zeroOut[i] = make([]linalg.Vector, len(s))
		for j, x := range s {
			zeroOut[i][j] = make(linalg.Vector, len(x))
		}
	}
	return &constRResult{Output: vecs, ROutput: zeroOut}
}

type constResult struct {
	Output [][]linalg.Vector
}

func (c *constResult) OutputSeqs() [][]linalg.Vector {
	return c.Output
}

func (c *constResult) PropagateGradient(u [][]linalg.Vector, g autofunc.Gradient) {
}

type constRResult struct {
	Output  [][]linalg.Vector
	ROutput [][]linalg.Vector
}

func (c *constRResult) OutputSeqs() [][]linalg.Vector {
	return c.Output
}

func (c *constRResult) ROutputSeqs() [][]linalg.Vector {
	return c.ROutput
}

func (c *constRResult) PropagateRGradient(u, uR [][]linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
}
