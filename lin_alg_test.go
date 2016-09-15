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

type outerProductTest struct{}

func (_ outerProductTest) Apply(in Result) Result {
	return OuterProduct(linTranTestVec, in)
}

func (_ outerProductTest) ApplyR(rv RVector, in RResult) RResult {
	vec1 := NewRVariable(linTranTestVec, rv)
	return OuterProductR(vec1, in)
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

func TestOuterProductOutput(t *testing.T) {
	res := outerProductTest{}.Apply(linTranTestMat2.Data)
	expected := []float64{
		16, 8, 12,
		12, 6, 9,
		8, 4, 6,
		4, 2, 3,
	}
	for i, x := range expected {
		if math.Abs(x-res.Output()[i]) > funcTestPrec {
			t.Errorf("bad output %d: expected %f got %f", i, x, res.Output()[i])
		}
	}
}

func TestOuterProductGradient(t *testing.T) {
	funcTest := &FuncTest{
		F:     ComposedFunc{outerProductTest{}, AddTwice{}},
		Vars:  linTranTestVariables,
		Input: linTranTestMat2.Data,
	}
	funcTest.Run(t)
}

func TestOuterProductRGradient(t *testing.T) {
	funcTest := &RFuncTest{
		F:     ComposedRFunc{outerProductTest{}, AddTwice{}},
		Vars:  linTranTestVariables,
		Input: linTranTestMat2.Data,
		RV:    linTranTestRVec,
	}
	funcTest.Run(t)
}
