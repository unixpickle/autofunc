package functest

import (
	"math"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// A Checker can compute exact derivatives and find
// approximations of them.
type Checker interface {
	// TestPrec returns the desired test precision
	// for comparing
	TestPrec() float64

	// Variables returns the variables of interest.
	Variables() []*autofunc.Variable

	// ApproxPartials approximates partial derivatives of
	// some vector function with respect to a component in
	// a variable.
	ApproxPartials(v *autofunc.Variable, idx int) linalg.Vector

	// Jacobian computes the exact jacobian of a function.
	// There is one autofunc.Gradient per output in the
	// function's output vector.
	Jacobian() []autofunc.Gradient
}

// Check performs gradient checking on a Checker.
func Check(t *testing.T, c Checker) {
	jacobian := c.Jacobian()
	for varIdx, variable := range c.Variables() {
		for elementIdx := range variable.Vector {
			approxVec := c.ApproxPartials(variable, elementIdx)
			for outputIdx, grad := range jacobian {
				actual := grad[variable][elementIdx]
				expected := approxVec[outputIdx]
				if math.IsNaN(actual) {
					t.Errorf("var %d, output %d, entry %d: expected %f got %f",
						varIdx, outputIdx, elementIdx, expected, actual)
				} else if math.Abs(actual-expected) > c.TestPrec() {
					t.Errorf("var %d, output %d, entry %d: expected %f got %f",
						varIdx, outputIdx, elementIdx, expected, actual)
				}
			}
		}
	}
}

// An RChecker adds r-operator checking capabilities to
// the Checker interface.
type RChecker interface {
	Checker

	// ApproxPartialsR computes the approximate derivative
	// of the exact partial derivatives.
	// Thus, it is not a numerical second derivative of
	// ApproxPartials, but rather a numerical derivative of
	// an exact version of ApproxPartials.
	ApproxPartialsR(v *autofunc.Variable, idx int) linalg.Vector

	// JacobianR computes the derivative of Jacobian() with
	// respect to some variable R.
	JacobianR() []autofunc.RGradient

	// ApproxOutputR computes the approximate derivative
	// of the function output with respect to R.
	ApproxOutputR() linalg.Vector

	// OutputR computes the exact derivative of the function
	// output with respect to R.
	OutputR() linalg.Vector
}

// CheckR performs gradient checking, r-output checking,
// and r-gradient checking.
// It includes all of the checks done by Check.
func CheckR(t *testing.T, c RChecker) {
	Check(t, c)
	checkROutput(t, c)
	checkRGradient(t, c)
}

func checkROutput(t *testing.T, c RChecker) {
	expected := c.ApproxOutputR()
	actual := c.OutputR()

	for i, a := range actual {
		x := expected[i]
		if math.IsNaN(a) {
			t.Errorf("r-output %d: got NaN", i)
			continue
		} else if math.Abs(x-a) > c.TestPrec() {
			t.Errorf("r-output %d: expected %f but got %f", i, x, a)
		}
	}
}

func checkRGradient(t *testing.T, c RChecker) {
	jacobian := c.JacobianR()
	for varIdx, variable := range c.Variables() {
		for elementIdx := range variable.Vector {
			approxVec := c.ApproxPartialsR(variable, elementIdx)
			for outputIdx, grad := range jacobian {
				actual := grad[variable][elementIdx]
				expected := approxVec[outputIdx]
				if math.IsNaN(actual) {
					t.Errorf("r-gradient: var %d, output %d, entry %d: expected %f got %f",
						varIdx, outputIdx, elementIdx, expected, actual)
				} else if math.Abs(actual-expected) > c.TestPrec() {
					t.Errorf("r-gradient: var %d, output %d, entry %d: expected %f got %f",
						varIdx, outputIdx, elementIdx, expected, actual)
				}
			}
		}
	}
}
