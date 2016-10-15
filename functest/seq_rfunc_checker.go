package functest

import (
	"fmt"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// A SeqRFuncChecker is a Checker for a seqfunc.Func.
//
// It also implements the FullCheck helper, which is
// similar to FuncChecker.FullCheck.
type SeqRFuncChecker struct {
	// F is the function to check.
	F seqfunc.RFunc

	// Vars are the variables whose gradients are checked.
	Vars []*autofunc.Variable

	// Input is the input to pass the function.
	// If the Input is to be gradient checked, it should
	// appear in Vars.
	Input [][]*autofunc.Variable

	// RV stores the first derivatives of any relevant
	// variables and is passed to F.ApplySeqsR.
	RV autofunc.RVector

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
func (s *SeqRFuncChecker) FullCheck(t *testing.T) {
	t.Run("Standard", func(t *testing.T) {
		CheckR(t, s)
		s.testConsistency(t)
	})
	allVars := s.Vars
	for i := range allVars {
		t.Run(fmt.Sprintf("Vars[%d]", i), func(t *testing.T) {
			newS := *s
			newS.Vars = allVars[i : i+1]
			CheckR(t, &newS)
			newS.testConsistency(t)
		})
	}
	t.Run("Accumulation", func(t *testing.T) {
		newS := *s
		newS.F = seqfunc.ComposedRFunc{s.F, addTwiceSeq{}}
		CheckR(t, &newS)
		newS.testConsistency(t)
	})
	t.Run("Square", func(t *testing.T) {
		newS := *s
		newS.F = seqfunc.ComposedRFunc{s.F, mulTwiceSeq{}}
		CheckR(t, &newS)
		newS.testConsistency(t)
	})
}

// TestPrec returns s.Prec or DefaultPrec.
func (s *SeqRFuncChecker) TestPrec() float64 {
	if s.Prec == 0 {
		return DefaultPrec
	}
	return s.Prec
}

// Variables returns s.Vars.
func (s *SeqRFuncChecker) Variables() []*autofunc.Variable {
	return s.Vars
}

// ApproxPartials uses gradient checking to approximate
// the partial derivatives of the outputs with respect to
// the parameter.
func (s *SeqRFuncChecker) ApproxPartials(v *autofunc.Variable, idx int) linalg.Vector {
	param := &v.Vector[idx]
	old := *param
	*param = old + s.delta()
	val1 := copyFlatOut(s.F.ApplySeqsR(s.RV, s.input()).OutputSeqs())
	*param = old - s.delta()
	val2 := copyFlatOut(s.F.ApplySeqsR(s.RV, s.input()).OutputSeqs())
	*param = old
	return val1.Add(val2.Scale(-1)).Scale(0.5 / s.delta())
}

// Jacobian uses s.F to compute the jacobian of the output.
func (s *SeqRFuncChecker) Jacobian() []autofunc.Gradient {
	var jacobian []autofunc.Gradient
	output := s.F.ApplySeqsR(s.RV, s.input())

	upstream := zeroSeqList(output.OutputSeqs())
	upstreamR := zeroSeqList(output.ROutputSeqs())

	for i, outSeq := range output.OutputSeqs() {
		for j, vec := range outSeq {
			for k := range vec {
				grad := autofunc.NewGradient(s.Vars)
				rgrad := autofunc.NewRGradient(s.Vars)
				upstream[i][j][k] = 1
				output.PropagateRGradient(upstream, upstreamR, rgrad, grad)
				upstream[i][j][k] = 0
				jacobian = append(jacobian, grad)
			}
		}
	}

	return jacobian
}

// ApproxPartialsR approximates the second derivatives of
// the function based on exact first derivatives.
func (s *SeqRFuncChecker) ApproxPartialsR(v *autofunc.Variable, idx int) linalg.Vector {
	exactPartials := func() linalg.Vector {
		var partials linalg.Vector
		out := s.F.ApplySeqsR(s.RV, s.input())
		upstream := zeroSeqList(out.OutputSeqs())
		zeroUp := zeroSeqList(out.OutputSeqs())
		for i, upSeq := range upstream {
			for j, upVec := range upSeq {
				for k := range upVec {
					grad := autofunc.NewGradient(s.Vars)
					rgrad := autofunc.NewRGradient(s.Vars)
					upstream[i][j][k] = 1
					out.PropagateRGradient(upstream, zeroUp, rgrad, grad)
					upstream[i][j][k] = 0
					partials = append(partials, grad[v][idx])
				}
			}
		}
		return partials
	}

	varBackups := map[*autofunc.Variable]linalg.Vector{}
	for v, rv := range s.RV {
		varBackups[v] = v.Vector.Copy()
		v.Vector.Add(rv.Copy().Scale(-s.delta()))
	}
	partials1 := exactPartials()
	for v, rv := range s.RV {
		v.Vector.Add(rv.Copy().Scale(2 * s.delta()))
	}
	partials2 := exactPartials()
	for v, backup := range varBackups {
		copy(v.Vector, backup)
	}
	return partials2.Add(partials1.Scale(-1)).Scale(0.5 / s.delta())
}

// JacobianR computes the derivatives of Jacobian() with
// respect to the variable R.
func (s *SeqRFuncChecker) JacobianR() []autofunc.RGradient {
	var jacobian []autofunc.RGradient
	out := s.F.ApplySeqsR(s.RV, s.input())

	upstream := zeroSeqList(out.OutputSeqs())
	zeroUp := zeroSeqList(out.OutputSeqs())
	for i, upSeq := range upstream {
		for j, upVec := range upSeq {
			for k := range upVec {
				grad := autofunc.NewGradient(s.Vars)
				rgrad := autofunc.NewRGradient(s.Vars)
				upstream[i][j][k] = 1
				out.PropagateRGradient(upstream, zeroUp, rgrad, grad)
				upstream[i][j][k] = 0
				jacobian = append(jacobian, rgrad)
			}
		}
	}

	return jacobian
}

// ApproxOutputR approximates the derivative of the output
// with respect to R.
func (s *SeqRFuncChecker) ApproxOutputR() linalg.Vector {
	eval := func() linalg.Vector {
		return copyFlatOut(s.F.ApplySeqsR(s.RV, s.input()).OutputSeqs())
	}

	varBackups := map[*autofunc.Variable]linalg.Vector{}
	for v, rv := range s.RV {
		varBackups[v] = v.Vector.Copy()
		v.Vector.Add(rv.Copy().Scale(-s.delta()))
	}
	output1 := eval()
	for v, rv := range s.RV {
		v.Vector.Add(rv.Copy().Scale(2 * s.delta()))
	}
	output2 := eval()
	for v, backup := range varBackups {
		copy(v.Vector, backup)
	}
	return output2.Add(output1.Scale(-1)).Scale(0.5 / s.delta())
}

// OutputR computes the exact derivative of the output
// with respect to R.
func (s *SeqRFuncChecker) OutputR() linalg.Vector {
	return copyFlatOut(s.F.ApplySeqsR(s.RV, s.input()).ROutputSeqs())
}

func (s *SeqRFuncChecker) delta() float64 {
	if s.Delta == 0 {
		return DefaultDelta
	}
	return s.Delta
}

func (s *SeqRFuncChecker) input() seqfunc.RResult {
	return seqfunc.VarRResult(s.RV, s.Input)
}

func (s *SeqRFuncChecker) testConsistency(t *testing.T) {
	s.outputConsistency(t)
	s.jacobianConsistency(t)
}

func (s *SeqRFuncChecker) outputConsistency(t *testing.T) {
	nonR := s.F.ApplySeqs(seqfunc.VarResult(s.Input)).OutputSeqs()
	r := s.F.ApplySeqsR(s.RV, s.input()).OutputSeqs()

	if !s.seqsConsistent(nonR, r) {
		t.Errorf("output inconsistency: ApplySeqs gave %v ApplySeqsR gave %v", nonR, r)
	}
}

func (s *SeqRFuncChecker) jacobianConsistency(t *testing.T) {
	sc := SeqFuncChecker{F: s.F, Vars: s.Vars, Input: s.Input, Delta: s.Delta, Prec: s.Prec}
	nonR := sc.Jacobian()
	r := s.Jacobian()

	if len(nonR) != len(r) {
		t.Errorf("jacobian counts: R gave %d non-R gave %d", len(r), len(nonR))
		return
	}

	for i, x := range r {
		y := nonR[i]
		for varIdx, variable := range s.Vars {
			v1 := x[variable]
			v2 := y[variable]
			if !s.vecsConsistent(v1, v2) {
				t.Errorf("gradient %d var %d: ApplySeqs gave %v ApplySeqsR gave %v",
					i, varIdx, v2, v1)
			}
		}
	}
}

func (s *SeqRFuncChecker) seqsConsistent(s1, s2 [][]linalg.Vector) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, seq1 := range s1 {
		seq2 := s2[i]
		if len(seq1) != len(seq2) {
			return false
		}
		for j, x := range seq1 {
			y := seq2[j]
			if !s.vecsConsistent(x, y) {
				return false
			}
		}
	}
	return true
}

func (s *SeqRFuncChecker) vecsConsistent(v1, v2 linalg.Vector) bool {
	f := &SeqRFuncChecker{Prec: s.Prec}
	return f.vecsConsistent(v1, v2)
}

func zeroSeqList(model [][]linalg.Vector) [][]linalg.Vector {
	upstream := make([][]linalg.Vector, len(model))
	for i, outSeq := range model {
		upstream[i] = make([]linalg.Vector, len(outSeq))
		for j, outVec := range outSeq {
			upstream[i][j] = make(linalg.Vector, len(outVec))
		}
	}
	return upstream
}
