package autofunc

import (
	"math"
	"testing"
)

type matMulVecTest struct{}

func (_ matMulVecTest) Apply(in Result) Result {
	return MatMulVec(linTranTestMat1.Data, 3, 4, in)
}

func (_ matMulVecTest) ApplyR(rv RVector, in RResult) RResult {
	mat := NewRVariable(linTranTestMat1.Data, rv)
	return MatMulVecR(mat, 3, 4, in)
}

func TestMatMulVecOutput(t *testing.T) {
	res := matMulVecTest{}.Apply(linTranTestVec)
	expected := []float64{19, 20, 36}
	for i, x := range expected {
		if math.Abs(x-res.Output()[i]) > funcTestPrec {
			t.Errorf("bad output %d: expected %f got %f", i, x, res.Output()[i])
		}
	}
}

func TestMatMulVecGradient(t *testing.T) {
	funcTest := &FuncTest{
		F:     ComposedFunc{matMulVecTest{}, AddTwice{}},
		Vars:  linTranTestVariables,
		Input: linTranTestVec,
	}
	funcTest.Run(t)
}

func TestMatMulVecRGradient(t *testing.T) {
	funcTest := &RFuncTest{
		F:     ComposedRFunc{matMulVecTest{}, AddTwice{}},
		Vars:  linTranTestVariables,
		Input: linTranTestVec,
		RV:    linTranTestRVec,
	}
	funcTest.Run(t)
}
