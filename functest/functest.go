package functest

import (
	"math"
	"testing"

	. "github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

const (
	funcTestDelta = 1e-5
	funcTestPrec  = 1e-5
)

type AddTwice struct{}

func (_ AddTwice) Apply(r Result) Result {
	return Add(r, r)
}

func (_ AddTwice) ApplyR(v RVector, r RResult) RResult {
	return AddR(r, r)
}

type GradientTest interface {
	Variables() []*Variable
	ApproxPartials(param *float64) linalg.Vector
	Jacobian() []Gradient
}

type FuncTest struct {
	F     Func
	Vars  []*Variable
	Input Result
}

func (f *FuncTest) Variables() []*Variable {
	return f.Vars
}

func (f *FuncTest) ApproxPartials(param *float64) linalg.Vector {
	old := *param
	*param = old + funcTestDelta
	val1 := f.F.Apply(f.Input).Output()
	*param = old - funcTestDelta
	val2 := f.F.Apply(f.Input).Output()
	*param = old
	return val1.Add(val2.Scale(-1)).Scale(1.0 / (2 * funcTestDelta))
}

func (f *FuncTest) Jacobian() []Gradient {
	var jacobian []Gradient
	output := f.F.Apply(f.Input)

	for i := range output.Output() {
		grad := NewGradient(f.Vars)
		outGrad := make(linalg.Vector, len(output.Output()))
		outGrad[i] = 1
		output.PropagateGradient(outGrad, grad)
		jacobian = append(jacobian, grad)
	}

	return jacobian
}

func (f *FuncTest) Run(t *testing.T) {
	testFuncGradient(t, f)
}

type RFuncTest struct {
	F     RFunc
	Vars  []*Variable
	Input RResult
	RV    RVector
}

func (f *RFuncTest) Variables() []*Variable {
	return f.Vars
}

func (f *RFuncTest) ApproxPartials(param *float64) linalg.Vector {
	old := *param
	*param = old + funcTestDelta
	val1 := f.F.ApplyR(f.RV, f.Input).Output()
	*param = old - funcTestDelta
	val2 := f.F.ApplyR(f.RV, f.Input).Output()
	*param = old
	return val1.Add(val2.Scale(-1)).Scale(1.0 / (2 * funcTestDelta))
}

func (f *RFuncTest) Jacobian() []Gradient {
	var jacobian []Gradient
	output := f.F.ApplyR(f.RV, f.Input)

	for i := range output.Output() {
		grad := NewGradient(f.Vars)
		rgrad := NewRGradient(f.Vars)
		outGrad := make(linalg.Vector, len(output.Output()))
		zeroVec := make(linalg.Vector, len(outGrad))
		outGrad[i] = 1
		output.PropagateRGradient(outGrad, zeroVec, rgrad, grad)
		jacobian = append(jacobian, grad)
	}

	return jacobian
}

func (f *RFuncTest) Run(t *testing.T) {
	testFuncGradient(t, f)

	output := f.F.ApplyR(f.RV, f.Input)

	approxGrads, outGrads := f.approximateR()

	for outIdx, approxOut := range outGrads {
		actual := output.ROutput()[outIdx]
		if math.Abs(actual-approxOut) > funcTestPrec {
			t.Errorf("output %d: expected ROutput %f got %f", outIdx, approxOut, actual)
		}
	}

	for outIdx, approxGrad := range approxGrads {
		actualGrad := NewRGradient(f.Vars)
		outVec := make(linalg.Vector, len(approxGrads))
		outVec[outIdx] = 1
		zeroVec := make(linalg.Vector, len(outVec))
		output.PropagateRGradient(outVec, zeroVec, actualGrad, nil)
		for varIdx, variable := range f.Vars {
			actual := actualGrad[variable]
			expected := approxGrad[variable]
			for i, x := range expected {
				a := actual[i]
				if math.Abs(a-x) > funcTestPrec {
					t.Errorf("var %d, output %d, entry %d: expected %f got %f (r-gradient)",
						varIdx, outIdx, i, x, a)
				}
			}
		}
	}
}

func (f *RFuncTest) approximateR() ([]RGradient, linalg.Vector) {
	for v, add := range f.RV {
		v.Vector.Add(add.Copy().Scale(funcTestDelta))
	}

	unusedRGrad := NewRGradient(f.Variables())

	res1 := f.F.ApplyR(f.RV, f.Input)
	var grads1 []Gradient

	for outputIdx := range res1.Output() {
		grad := NewGradient(f.Variables())
		outGrads := make(linalg.Vector, len(res1.Output()))
		outGrads[outputIdx] = 1
		zeroVec := make(linalg.Vector, len(outGrads))
		res1.PropagateRGradient(outGrads, zeroVec, unusedRGrad, grad)
		grads1 = append(grads1, grad)
	}

	for v, add := range f.RV {
		v.Vector.Add(add.Copy().Scale(-2 * funcTestDelta))
	}

	res2 := f.F.ApplyR(f.RV, f.Input)
	var grads2 []Gradient

	for outputIdx := range res2.Output() {
		grad := NewGradient(f.Variables())
		outGrads := make(linalg.Vector, len(res2.Output()))
		outGrads[outputIdx] = 1
		zeroVec := make(linalg.Vector, len(outGrads))
		res2.PropagateRGradient(outGrads, zeroVec, unusedRGrad, grad)
		grads2 = append(grads2, grad)
	}

	for v, add := range f.RV {
		v.Vector.Add(add.Copy().Scale(funcTestDelta))
	}

	resGrads := make([]RGradient, len(grads1))
	for i, grad1 := range grads1 {
		grad2 := grads2[i]
		for v, vec1 := range grad1 {
			vec2 := grad2[v]
			vec1.Add(vec2.Scale(-1)).Scale(0.5 / funcTestDelta)
		}
		resGrads[i] = RGradient(grad1)
	}

	outGrads := res1.Output().Add(res2.Output().Scale(-1)).Scale(0.5 / funcTestDelta)

	return resGrads, outGrads
}

func testFuncGradient(t *testing.T, f GradientTest) {
	jacobian := f.Jacobian()
	for varIdx, variable := range f.Variables() {
		for elementIdx := range variable.Vector {
			varPtr := &variable.Vector[elementIdx]
			approxVec := f.ApproxPartials(varPtr)
			for outputIdx, grad := range jacobian {
				actual := grad[variable][elementIdx]
				expected := approxVec[outputIdx]
				if math.Abs(actual-expected) > funcTestPrec {
					t.Errorf("var %d, output %d, entry %d: expected %f got %f (gradient)",
						varIdx, outputIdx, elementIdx, expected, actual)
				}
			}
		}
	}
}
