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

func (c *constRResult) OutputSeqs() [][]linalg.Vector {
	return c.Output
}

func (c *constRResult) ROutputSeqs() [][]linalg.Vector {
	return c.ROutput
}

func (c *constRResult) PropagateRGradient(u, uR [][]linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
}

type varResult struct {
	Output [][]linalg.Vector
	Vars   [][]*autofunc.Variable
}

// VarResult creates a Result from an underlying list of
// variable sequences.
// The result is only valid as long as none of the
// variables are modified.
func VarResult(vars [][]*autofunc.Variable) Result {
	res := &varResult{
		Output: make([][]linalg.Vector, len(vars)),
		Vars:   vars,
	}
	for i, seq := range vars {
		res.Output[i] = make([]linalg.Vector, len(seq))
		for j, x := range seq {
			res.Output[i][j] = x.Vector
		}
	}
	return res
}

func (v *varResult) OutputSeqs() [][]linalg.Vector {
	return v.Output
}

func (v *varResult) PropagateGradient(u [][]linalg.Vector, g autofunc.Gradient) {
	for i, seq := range u {
		for j, vec := range seq {
			if variable := v.Vars[i][j]; !variable.Constant(g) {
				us := make(linalg.Vector, len(vec))
				copy(us, vec)
				variable.PropagateGradient(us, g)
			}
		}
	}
}

type varRResult struct {
	Output  [][]linalg.Vector
	ROutput [][]linalg.Vector
	RVars   [][]*autofunc.RVariable
}

// VarRResult is like VarResult but for RResults.
func VarRResult(rv autofunc.RVector, vars [][]*autofunc.Variable) RResult {
	res := &varRResult{
		Output:  make([][]linalg.Vector, len(vars)),
		ROutput: make([][]linalg.Vector, len(vars)),
		RVars:   make([][]*autofunc.RVariable, len(vars)),
	}
	for i, seq := range vars {
		res.Output[i] = make([]linalg.Vector, len(seq))
		res.ROutput[i] = make([]linalg.Vector, len(seq))
		res.RVars[i] = make([]*autofunc.RVariable, len(seq))
		for j, x := range seq {
			rVar := autofunc.NewRVariable(x, rv)
			res.RVars[i][j] = rVar
			res.Output[i][j] = rVar.Output()
			res.ROutput[i][j] = rVar.ROutput()
		}
	}
	return res
}

func (v *varRResult) OutputSeqs() [][]linalg.Vector {
	return v.Output
}

func (v *varRResult) ROutputSeqs() [][]linalg.Vector {
	return v.ROutput
}

func (v *varRResult) PropagateRGradient(u, uR [][]linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	for i, seq := range u {
		for j, vec := range seq {
			if variable := v.RVars[i][j]; !variable.Constant(rg, g) {
				us := make(linalg.Vector, len(vec))
				copy(us, vec)
				usR := make(linalg.Vector, len(vec))
				copy(usR, uR[i][j])
				variable.PropagateRGradient(us, usR, rg, g)
			}
		}
	}
}
