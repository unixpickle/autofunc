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
	Input *Variable
	Cache *VectorCache
}

func (f *FuncTest) Variables() []*Variable {
	return f.Vars
}

func (f *FuncTest) ApproxPartials(param *float64) linalg.Vector {
	old := *param
	*param = old + funcTestDelta
	val1 := f.F.Apply(f.Input).Output().Copy()
	*param = old - funcTestDelta
	val2 := f.F.Apply(f.Input).Output().Copy()
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

	testFuncCacheUsage(t, f.Cache, func() interface{} {
		output := f.F.Apply(f.Input)
		for i := range output.Output() {
			grad := NewGradient(f.Vars)
			outGrad := make(linalg.Vector, len(output.Output()))
			outGrad[i] = 1
			output.PropagateGradient(outGrad, grad)
		}
		return output
	}, func(res interface{}) {
		res.(Result).Release()
	})
}

type RFuncTest struct {
	F     RFunc
	Vars  []*Variable
	Input *Variable
	RV    RVector
	Cache *VectorCache
}

func (f *RFuncTest) Variables() []*Variable {
	return f.Vars
}

func (f *RFuncTest) ApproxPartials(param *float64) linalg.Vector {
	inputRV := NewRVariableCache(f.Input, f.RV, f.Cache)
	old := *param
	*param = old + funcTestDelta
	val1 := f.F.ApplyR(f.RV, inputRV).Output().Copy()
	*param = old - funcTestDelta
	val2 := f.F.ApplyR(f.RV, inputRV).Output().Copy()
	*param = old
	return val1.Add(val2.Scale(-1)).Scale(1.0 / (2 * funcTestDelta))
}

func (f *RFuncTest) Jacobian() []Gradient {
	var jacobian []Gradient
	output := f.F.ApplyR(f.RV, f.rInput())

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

	output := f.F.ApplyR(f.RV, f.rInput())

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
					t.Errorf("var %d, output %d, entry %d: expected %f got %f",
						varIdx, outIdx, i, x, a)
				}
			}
		}
	}

	testFuncCacheUsage(t, f.Cache, func() interface{} {
		output := f.F.ApplyR(f.RV, f.rInput())
		for i := range output.Output() {
			grad := NewGradient(f.Vars)
			rgrad := NewRGradient(f.Vars)
			outGrad := make(linalg.Vector, len(output.Output()))
			outGrad[i] = 1
			outGradR := make(linalg.Vector, len(output.Output()))
			output.PropagateRGradient(outGrad, outGradR, rgrad, grad)
		}
		return output
	}, func(res interface{}) {
		res.(RResult).Release()
	})
}

func (f *RFuncTest) approximateR() ([]RGradient, linalg.Vector) {
	for v, add := range f.RV {
		v.Vector.Add(add.Copy().Scale(funcTestDelta))
	}

	unusedRGrad := NewRGradient(f.Variables())

	res1 := f.F.ApplyR(f.RV, f.rInput())
	var grads1 []Gradient

	for outputIdx := range res1.Output() {
		grad := NewGradient(f.Variables())
		outGrads := make(linalg.Vector, len(res1.Output()))
		outGrads[outputIdx] = 1
		zeroVec := make(linalg.Vector, len(outGrads))
		res1.PropagateRGradient(outGrads, zeroVec, unusedRGrad, grad)
		grads1 = append(grads1, grad)
	}

	res1Out := res1.Output().Copy()

	for v, add := range f.RV {
		v.Vector.Add(add.Copy().Scale(-2 * funcTestDelta))
	}

	res2 := f.F.ApplyR(f.RV, f.rInput())
	var grads2 []Gradient

	for outputIdx := range res2.Output() {
		grad := NewGradient(f.Variables())
		outGrads := make(linalg.Vector, len(res2.Output()))
		outGrads[outputIdx] = 1
		zeroVec := make(linalg.Vector, len(outGrads))
		res2.PropagateRGradient(outGrads, zeroVec, unusedRGrad, grad)
		grads2 = append(grads2, grad)
	}

	res2Out := res2.Output().Copy()

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

	outGrads := res1Out.Add(res2Out.Scale(-1)).Scale(0.5 / funcTestDelta)

	return resGrads, outGrads
}

func (f *RFuncTest) rInput() RResult {
	return NewRVariableCache(f.Input, f.RV, f.Cache)
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
					t.Errorf("var %d, output %d, entry %d: expected %f got %f",
						varIdx, outputIdx, elementIdx, expected, actual)
				}
			}
		}
	}
}

func testFuncCacheUsage(t *testing.T, c *VectorCache, run func() interface{},
	release func(interface{})) {
	c.Clear()

	run()
	release(run())

	// Give the cache an excess of every slice size in
	// order to catch some leaks.
	for key := range c.UsageHistogram() {
		c.Free(make(linalg.Vector, key))
	}

	baseAmount := c.FloatCount()

	res := run()
	preAmount := c.FloatCount()
	release(res)
	postAmount := c.FloatCount()
	release(res)
	postAmountTwo := c.FloatCount()
	if postAmountTwo != postAmount {
		t.Fatalf("second release went from count %d to count %d", postAmount, postAmountTwo)
	}

	if postAmount != baseAmount {
		t.Errorf("before run+release had %d free, now have %d free", baseAmount, postAmount)
	}

	res = run()
	newPreAmount := c.FloatCount()
	if newPreAmount != preAmount {
		t.Errorf("after release+run, size went from %d to %d", preAmount, newPreAmount)
	}
	release(res)
	newPostAmount := c.FloatCount()
	if newPostAmount != postAmount {
		t.Errorf("After run+release, size went from %d to %d", postAmount, newPostAmount)
	}
}
