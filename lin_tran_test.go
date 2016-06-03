package autofunc

import (
	"math"
	"testing"

	"github.com/unixpickle/num-analysis/linalg"
)

const (
	linTranTestDelta = 1e-5
	linTranTestPrec  = 1e-5
)

var linTranTestMat1 = &LinTran{
	Data: &Variable{
		Vector: linalg.Vector{
			1, 2, 3, 3,
			4, 5, -6, 1,
			7, 8, -10, 4,
		},
	},
	Rows: 3,
	Cols: 4,
}

var linTranTestMat2 = &LinTran{
	Data: &Variable{Vector: linalg.Vector{4, 2, 3}},
	Rows: 1,
	Cols: 3,
}

var linTranTestVec = &Variable{
	Vector: linalg.Vector{4, 3, 2, 1},
}

var linTranTestRVec = RVector{
	linTranTestMat1.Data: linalg.Vector{
		0.40350, 0.75874, 0.87843, 0.86287,
		0.97869, 0.27141, 0.84472, 0.43952,
		0.49841, 0.46748, 0.71821, 0.57590,
	},
	linTranTestMat2.Data: linalg.Vector{
		0.45823, 0.66692, 0.14662,
	},
	linTranTestVec: linalg.Vector{0, -1, 3, -7},
}

func TestLinTranGradient(t *testing.T) {
	res := linTranTestFunc()
	gradVars := []*Variable{linTranTestMat1.Data, linTranTestMat2.Data, linTranTestVec}
	grad := NewGradient(gradVars)
	res.PropagateGradient(linalg.Vector{1}, grad)

	for varIdx, v := range gradVars {
		for i, exact := range grad[v] {
			approx := approximateLinTranPartial(&v.Vector[i])
			if math.Abs(approx-exact) > linTranTestPrec {
				t.Errorf("var %d: got %f expected %f (idx %d)", varIdx, exact, approx, i)
				break
			}
		}
	}
}

func TestLinTranRGradient(t *testing.T) {
	gradVars := []*Variable{linTranTestMat1.Data, linTranTestMat2.Data, linTranTestVec}
	grad := NewGradient(gradVars)
	rgrad := NewRGradient(gradVars)
	res := linTranTestFuncR()
	res.PropagateRGradient(linalg.Vector{1}, linalg.Vector{0}, rgrad, grad)

	for varIdx, v := range gradVars {
		for i, exact := range grad[v] {
			approx := approximateLinTranPartial(&v.Vector[i])
			if math.Abs(approx-exact) > linTranTestPrec {
				t.Errorf("var %d: got %f expected %f (idx %d)", varIdx, exact, approx, i)
				break
			}
		}
	}

	approxR, approxOutDiff := approximateRGrad()
	if math.Abs(res.ROutput()[0]-approxOutDiff) > linTranTestPrec {
		t.Errorf("invalid ROutput: expected %f got %f", approxOutDiff, res.ROutput()[0])
	}
	for varIdx, variable := range gradVars {
		vec := rgrad[variable]
		approx := approxR[variable]
		for i, approxVal := range approx {
			exact := vec[i]
			if math.Abs(approxVal-exact) > linTranTestPrec {
				t.Errorf("var %d: expected %f but got %f (idx %d)", varIdx, approxVal, exact, i)
				break
			}
		}
	}
}

func linTranTestFunc() Result {
	return linTranTestMat2.Apply(linTranTestMat1.Apply(linTranTestVec))
}

func linTranTestFuncR() RResult {
	rVar := NewRVariable(linTranTestVec, linTranTestRVec)
	return linTranTestMat2.ApplyR(linTranTestRVec,
		linTranTestMat1.ApplyR(linTranTestRVec, rVar))
}

func approximateLinTranPartial(param *float64) float64 {
	old := *param
	*param = old + linTranTestDelta
	val1 := linTranTestFunc().Output()[0]
	*param = old - linTranTestDelta
	val2 := linTranTestFunc().Output()[0]
	*param = old
	return (val1 - val2) / (2 * linTranTestDelta)
}

func approximateRGrad() (RGradient, float64) {
	gradVars := []*Variable{linTranTestMat1.Data, linTranTestMat2.Data, linTranTestVec}

	for v, add := range linTranTestRVec {
		v.Vector.Add(add.Copy().Scale(linTranTestDelta))
	}

	grad1 := NewGradient(gradVars)
	res1 := linTranTestFunc()
	res1.PropagateGradient(linalg.Vector{1}, grad1)

	for v, add := range linTranTestRVec {
		v.Vector.Add(add.Copy().Scale(-2 * linTranTestDelta))
	}

	grad2 := NewGradient(gradVars)
	res2 := linTranTestFunc()
	res2.PropagateGradient(linalg.Vector{1}, grad2)

	for v, add := range linTranTestRVec {
		v.Vector.Add(add.Copy().Scale(linTranTestDelta))
	}

	for v, vec := range grad1 {
		vec1 := grad2[v]
		vec.Add(vec1.Scale(-1)).Scale(0.5 / linTranTestDelta)
	}

	outDelta := (res1.Output()[0] - res2.Output()[0]) / (2 * linTranTestDelta)
	return RGradient(grad1), outDelta
}
