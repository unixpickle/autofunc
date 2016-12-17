package autofunc

import (
	"math"
	"testing"

	. "github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
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

type transposeTest struct{}

func (_ transposeTest) Apply(in Result) Result {
	return Transpose(in, 3, 4)
}

func (_ transposeTest) ApplyR(rv RVector, in RResult) RResult {
	return TransposeR(in, 3, 4)
}

func TestMatMulVecOutput(t *testing.T) {
	res := matMulVecTest{}.Apply(linTranTestVec)
	expected := []float64{19, 20, 36}
	for i, x := range expected {
		if math.Abs(x-res.Output()[i]) > functest.DefaultPrec {
			t.Errorf("bad output %d: expected %f got %f", i, x, res.Output()[i])
		}
	}
}

func TestMatMulVecChecks(t *testing.T) {
	f := &functest.RFuncChecker{
		F:     matMulVecTest{},
		Vars:  linTranTestVariables,
		Input: linTranTestVec,
		RV:    linTranTestRVec,
	}
	f.FullCheck(t)
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
		if math.Abs(x-res.Output()[i]) > functest.DefaultPrec {
			t.Errorf("bad output %d: expected %f got %f", i, x, res.Output()[i])
		}
	}
}

func TestOuterProductChecks(t *testing.T) {
	f := &functest.RFuncChecker{
		F:     outerProductTest{},
		Vars:  linTranTestVariables,
		Input: linTranTestMat2.Data,
		RV:    linTranTestRVec,
	}
	f.FullCheck(t)
}

func TestTransposeOutput(t *testing.T) {
	trans := Transpose(linTranTestMat1.Data, 3, 4).Output()
	expected := []float64{
		1, 4, 7,
		2, 5, 8,
		3, -6, -10,
		3, 1, 4,
	}
	if trans.Copy().Scale(-1).Add(expected).MaxAbs() > 1e-5 {
		t.Errorf("expected %v got %v", expected, trans)
	}
}

func TestTransposeChecks(t *testing.T) {
	f := &functest.RFuncChecker{
		F:     transposeTest{},
		Vars:  linTranTestVariables,
		Input: linTranTestMat1.Data,
		RV:    linTranTestRVec,
	}
	f.FullCheck(t)
}
