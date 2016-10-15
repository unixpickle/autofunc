package functest

import (
	"fmt"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// A SeqFuncChecker is a Checker for a seqfunc.Func.
//
// It also implements the FullCheck helper, which is
// similar to FuncChecker.FullCheck.
type SeqFuncChecker struct {
	// F is the function to check.
	F seqfunc.Func

	// Vars are the variables whose gradients are checked.
	Vars []*autofunc.Variable

	// Input is the input to pass the function.
	// If the Input is to be gradient checked, it should
	// appear in Vars.
	Input [][]*autofunc.Variable

	// Delta is the delta used for gradient approximation.
	// If it is 0, DefaultDelta is used.
	Delta float64

	// Prec is the precision to use when comparing values.
	// If it is 0, DefaultPrec is used.
	Prec float64
}

// FullCheck checks gradients on f and its variations.
// For example, it will verify that the function computes
// the correct gradients even when only one variable's
// gradient is requested.
func (s *SeqFuncChecker) FullCheck(t *testing.T) {
	t.Run("Standard", func(t *testing.T) {
		Check(t, s)
	})
	allVars := s.Vars
	for i := range allVars {
		t.Run(fmt.Sprintf("Vars[%d]", i), func(t *testing.T) {
			newS := *s
			newS.Vars = allVars[i : i+1]
			Check(t, &newS)
		})
	}
	t.Run("Accumulation", func(t *testing.T) {
		newS := *s
		newS.F = seqfunc.ComposedFunc{s.F, addTwiceSeq{}}
		Check(t, &newS)
	})
}

// TestPrec returns s.Prec or DefaultPrec.
func (s *SeqFuncChecker) TestPrec() float64 {
	if s.Prec == 0 {
		return DefaultPrec
	}
	return s.Prec
}

// Variables returns s.Vars.
func (s *SeqFuncChecker) Variables() []*autofunc.Variable {
	return s.Vars
}

// ApproxPartials uses gradient checking to approximate
// the partial derivatives of the outputs with respect to
// the parameter.
func (s *SeqFuncChecker) ApproxPartials(v *autofunc.Variable, idx int) linalg.Vector {
	param := &v.Vector[idx]
	old := *param
	*param = old + s.delta()
	val1 := copyFlatOut(s.F.ApplySeqs(s.input()).OutputSeqs())
	*param = old - s.delta()
	val2 := copyFlatOut(s.F.ApplySeqs(s.input()).OutputSeqs())
	*param = old
	return val1.Add(val2.Scale(-1)).Scale(0.5 / s.delta())
}

// Jacobian uses s.F to compute the jacobian of the output.
func (s *SeqFuncChecker) Jacobian() []autofunc.Gradient {
	var jacobian []autofunc.Gradient
	output := s.F.ApplySeqs(s.input())

	upstream := make([][]linalg.Vector, len(output.OutputSeqs()))
	for i, outSeq := range output.OutputSeqs() {
		upstream[i] = make([]linalg.Vector, len(outSeq))
		for j, outVec := range outSeq {
			upstream[i][j] = make(linalg.Vector, len(outVec))
		}
	}

	for i, outSeq := range output.OutputSeqs() {
		for j, vec := range outSeq {
			for k := range vec {
				grad := autofunc.NewGradient(s.Vars)
				upstream[i][j][k] = 1
				output.PropagateGradient(upstream, grad)
				upstream[i][j][k] = 0
				jacobian = append(jacobian, grad)
			}
		}
	}

	return jacobian
}

func (s *SeqFuncChecker) delta() float64 {
	if s.Delta == 0 {
		return DefaultDelta
	}
	return s.Delta
}

func (s *SeqFuncChecker) input() seqfunc.Result {
	return seqfunc.VarResult(s.Input)
}

func copyFlatOut(seqs [][]linalg.Vector) linalg.Vector {
	var res linalg.Vector
	for _, seq := range seqs {
		for _, v := range seq {
			res = append(res, v...)
		}
	}
	return res
}
