// Package functest provides various gradient checking
// helpers to verify your differentiable functions.
package functest

import (
	"fmt"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

const (
	// DefaultDelta is the default difference in input
	// used to approximate a partial derivative.
	DefaultDelta = 1e-5

	// DefaultPrec is the default precision used to tell
	// whether two outputs match.
	DefaultPrec = 1e-5
)

// A FuncChecker is a Checker for an autofunc.Func.
//
// It also implements the FullCheck helper, which runs
// tests targeted specifically at autofunc.Funcs.
type FuncChecker struct {
	// F is the function to check.
	F autofunc.Func

	// Vars are the variables whose gradients are checked.
	Vars []*autofunc.Variable

	// Input is the input to pass the function.
	// If the Input is to be gradient checked, it should
	// appear in Vars.
	Input *autofunc.Variable

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
func (f *FuncChecker) FullCheck(t *testing.T) {
	t.Run("Standard", func(t *testing.T) {
		Check(t, f)
	})
	allVars := f.Vars
	for i := range allVars {
		t.Run(fmt.Sprintf("Vars[%d]", i), func(t *testing.T) {
			newF := *f
			newF.Vars = allVars[i : i+1]
			Check(t, &newF)
		})
	}
	t.Run("Accumulation", func(t *testing.T) {
		newF := *f
		newF.F = autofunc.ComposedFunc{f.F, addTwice{}}
		Check(t, &newF)
	})
}

// TestPrec returns f.Prec or DefaultPrec.
func (f *FuncChecker) TestPrec() float64 {
	if f.Prec == 0 {
		return DefaultPrec
	}
	return f.Prec
}

// Variables returns f.Vars.
func (f *FuncChecker) Variables() []*autofunc.Variable {
	return f.Vars
}

// ApproxPartials uses gradient checking to approximate
// the partial derivatives of the output with respect to
// the parameter.
func (f *FuncChecker) ApproxPartials(v *autofunc.Variable, idx int) linalg.Vector {
	param := &v.Vector[idx]
	old := *param
	*param = old + f.delta()
	val1 := f.F.Apply(f.Input).Output().Copy()
	*param = old - f.delta()
	val2 := f.F.Apply(f.Input).Output().Copy()
	*param = old
	return val1.Add(val2.Scale(-1)).Scale(1.0 / (2 * f.delta()))
}

// Jacobian uses f.F to compute the jacobian of the output.
func (f *FuncChecker) Jacobian() []autofunc.Gradient {
	var jacobian []autofunc.Gradient
	output := f.F.Apply(f.Input)

	for i := range output.Output() {
		grad := autofunc.NewGradient(f.Vars)
		outGrad := make(linalg.Vector, len(output.Output()))
		outGrad[i] = 1
		output.PropagateGradient(outGrad, grad)
		jacobian = append(jacobian, grad)
	}

	return jacobian
}

func (f *FuncChecker) delta() float64 {
	if f.Delta == 0 {
		return DefaultDelta
	}
	return f.Delta
}
