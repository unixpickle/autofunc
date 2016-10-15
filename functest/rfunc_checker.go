package functest

import (
	"fmt"
	"math"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// An RFuncChecker is an RChecker for an autofunc.RFunc.
//
// It also implements the FullCheck helper, which runs
// tests targeted specifically at autofunc.RFuncs.
type RFuncChecker struct {
	// F is the function to check.
	F autofunc.RFunc

	// Vars are the variables whose gradients are checked.
	Vars []*autofunc.Variable

	// Input is the input which is fed into F.
	Input *autofunc.Variable

	// RV stores the first derivatives of any relevant
	// variables and is passed to F.ApplyR.
	RV autofunc.RVector

	// Delta is the delta used for gradient approximation.
	// If it is 0, DefaultDelta is used.
	Delta float64

	// Prec is the precision to use when comparing values.
	// If it is 0, DefaultPrec is used.
	Prec float64
}

// FullCheck checks gradients on f and its variations.
// It also verifies that the gradients produced by
// PropagateRGradient match those from PropagateGradient.
func (f *RFuncChecker) FullCheck(t *testing.T) {
	t.Run("Standard", func(t *testing.T) {
		CheckR(t, f)
		f.testConsistency(t)
	})
	allVars := f.Vars
	for i := range allVars {
		t.Run(fmt.Sprintf("Vars[%d]", i), func(t *testing.T) {
			newF := *f
			newF.Vars = allVars[i : i+1]
			CheckR(t, &newF)
			newF.testConsistency(t)
		})
	}
	t.Run("Accumulation", func(t *testing.T) {
		newF := *f
		newF.F = autofunc.ComposedRFunc{f.F, addTwice{}}
		CheckR(t, &newF)
		newF.testConsistency(t)
	})
	t.Run("Square", func(t *testing.T) {
		newF := *f
		newF.F = autofunc.ComposedRFunc{f.F, mulTwice{}}
		CheckR(t, &newF)
		newF.testConsistency(t)
	})
}

// TestPrec returns f.Prec or DefaultPrec.
func (f *RFuncChecker) TestPrec() float64 {
	if f.Prec == 0 {
		return DefaultPrec
	}
	return f.Prec
}

// Variables returns f.Vars.
func (f *RFuncChecker) Variables() []*autofunc.Variable {
	return f.Vars
}

// ApproxPartials approximates the partials using
// f.F.ApplyR.
func (f *RFuncChecker) ApproxPartials(v *autofunc.Variable, idx int) linalg.Vector {
	param := &v.Vector[idx]
	inputRV := f.rInput()
	old := *param
	*param = old + f.delta()
	val1 := f.F.ApplyR(f.RV, inputRV).Output().Copy()
	*param = old - f.delta()
	val2 := f.F.ApplyR(f.RV, inputRV).Output().Copy()
	*param = old
	return val1.Add(val2.Scale(-1)).Scale(0.5 / f.delta())
}

// Jacobian computes the exact jacobian using f.F.ApplyR.
func (f *RFuncChecker) Jacobian() []autofunc.Gradient {
	var jacobian []autofunc.Gradient
	output := f.F.ApplyR(f.RV, f.rInput())

	for i := range output.Output() {
		grad := autofunc.NewGradient(f.Vars)
		rgrad := autofunc.NewRGradient(f.Vars)
		outGrad := make(linalg.Vector, len(output.Output()))
		zeroVec := make(linalg.Vector, len(outGrad))
		outGrad[i] = 1
		output.PropagateRGradient(outGrad, zeroVec, rgrad, grad)
		jacobian = append(jacobian, grad)
	}

	return jacobian
}

// ApproxPartialsR approximates the second derivatives of
// the function based on exact first derivatives.
func (f *RFuncChecker) ApproxPartialsR(v *autofunc.Variable, idx int) linalg.Vector {
	exactPartials := func() linalg.Vector {
		var partials linalg.Vector
		out := f.F.ApplyR(f.RV, f.rInput())
		for i := range out.Output() {
			grad := autofunc.NewGradient(f.Vars)
			rgrad := autofunc.NewRGradient(f.Vars)
			zeroUp := make(linalg.Vector, len(out.Output()))
			upstream := make(linalg.Vector, len(out.Output()))
			upstream[i] = 1
			out.PropagateRGradient(upstream, zeroUp, rgrad, grad)
			partials = append(partials, grad[v][idx])
		}
		return partials
	}

	varBackups := map[*autofunc.Variable]linalg.Vector{}
	for v, rv := range f.RV {
		varBackups[v] = v.Vector.Copy()
		v.Vector.Add(rv.Copy().Scale(-f.delta()))
	}
	partials1 := exactPartials()
	for v, rv := range f.RV {
		v.Vector.Add(rv.Copy().Scale(2 * f.delta()))
	}
	partials2 := exactPartials()
	for v, backup := range varBackups {
		copy(v.Vector, backup)
	}
	return partials2.Add(partials1.Scale(-1)).Scale(0.5 / f.delta())
}

// JacobianR computes the derivatives of Jacobian() with
// respect to the variable R.
func (f *RFuncChecker) JacobianR() []autofunc.RGradient {
	var jacobian []autofunc.RGradient
	output := f.F.ApplyR(f.RV, f.rInput())

	for i := range output.Output() {
		grad := autofunc.NewGradient(f.Vars)
		rgrad := autofunc.NewRGradient(f.Vars)
		outGrad := make(linalg.Vector, len(output.Output()))
		zeroVec := make(linalg.Vector, len(outGrad))
		outGrad[i] = 1
		output.PropagateRGradient(outGrad, zeroVec, rgrad, grad)
		jacobian = append(jacobian, rgrad)
	}

	return jacobian
}

// ApproxOutputR approximates the derivative of the output
// with respect to R.
func (f *RFuncChecker) ApproxOutputR() linalg.Vector {
	eval := func() linalg.Vector {
		return f.F.ApplyR(f.RV, f.rInput()).Output().Copy()
	}

	varBackups := map[*autofunc.Variable]linalg.Vector{}
	for v, rv := range f.RV {
		varBackups[v] = v.Vector.Copy()
		v.Vector.Add(rv.Copy().Scale(-f.delta()))
	}
	output1 := eval()
	for v, rv := range f.RV {
		v.Vector.Add(rv.Copy().Scale(2 * f.delta()))
	}
	output2 := eval()
	for v, backup := range varBackups {
		copy(v.Vector, backup)
	}
	return output2.Add(output1.Scale(-1)).Scale(0.5 / f.delta())
}

// OutputR computes the exact derivative of the output
// with respect to R.
func (f *RFuncChecker) OutputR() linalg.Vector {
	return f.F.ApplyR(f.RV, f.rInput()).ROutput().Copy()
}

func (f *RFuncChecker) rInput() autofunc.RResult {
	return autofunc.NewRVariable(f.Input, f.RV)
}

func (f *RFuncChecker) delta() float64 {
	if f.Delta == 0 {
		return DefaultDelta
	}
	return f.Delta
}

func (f *RFuncChecker) testConsistency(t *testing.T) {
	f.outputConsistency(t)
	f.jacobianConsistency(t)
}

func (f *RFuncChecker) outputConsistency(t *testing.T) {
	nonR := f.F.Apply(f.Input).Output()
	r := f.F.ApplyR(f.RV, f.rInput()).Output()

	if !f.vecsConsistent(nonR, r) {
		t.Errorf("output inconsistency: Apply gave %v ApplyR gave %v", nonR, r)
	}
}

func (f *RFuncChecker) jacobianConsistency(t *testing.T) {
	fc := FuncChecker{F: f.F, Vars: f.Vars, Input: f.Input, Delta: f.Delta, Prec: f.Prec}
	nonR := fc.Jacobian()
	r := f.Jacobian()

	if len(nonR) != len(r) {
		t.Errorf("jacobian counts: R gave %d non-R gave %d", len(r), len(nonR))
		return
	}

	for i, x := range r {
		y := nonR[i]
		for varIdx, variable := range f.Vars {
			v1 := x[variable]
			v2 := y[variable]
			if !f.vecsConsistent(v1, v2) {
				t.Errorf("gradient %d var %d: Apply gave %v ApplyR gave %v",
					i, varIdx, v2, v1)
			}
		}
	}
}

func (f *RFuncChecker) vecsConsistent(v1, v2 linalg.Vector) bool {
	if len(v1) != len(v2) {
		return false
	}
	for i, x := range v1 {
		y := v2[i]
		if math.IsNaN(x) && math.IsNaN(y) {
			continue
		}
		if math.IsNaN(x) || math.IsNaN(y) || math.Abs(x-y) > f.TestPrec() {
			return false
		}
	}
	return true
}
