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
		Vector: linalg.Vector([]float64{
			1, 2, 3, 3,
			4, 5, -6, 1,
			7, 8, -10, 4,
		}),
	},
	Rows: 3,
	Cols: 4,
}

var linTranTestMat2 = &LinTran{
	Data: &Variable{Vector: linalg.Vector([]float64{4, 2, 3})},
	Rows: 1,
	Cols: 3,
}

var linTranTestVec = &Variable{
	Vector: linalg.Vector([]float64{4, 3, 2, 1}),
}

var linTranTestVariables = []*Variable{
	linTranTestMat1.Data, linTranTestMat2.Data, linTranTestVec,
}

var linTranTestRVec = RVector{
	linTranTestMat1.Data: linalg.Vector([]float64{
		0.40350, 0.75874, 0.87843, 0.86287,
		0.97869, 0.27141, 0.84472, 0.43952,
		0.49841, 0.46748, 0.71821, 0.57590,
	}),
	linTranTestMat2.Data: linalg.Vector([]float64{
		0.45823, 0.66692, 0.14662,
	}),
	linTranTestVec: linalg.Vector([]float64{0, -1, 3, -7}),
}

func TestLinTranOutput(t *testing.T) {
	res := linTranTestMat1.Apply(linTranTestVec)
	expected := []float64{19, 20, 36}
	for i, x := range expected {
		if math.Abs(x-res.Output()[i]) > linTranTestPrec {
			t.Errorf("bad output %d: expected %f got %f", i, x, res.Output()[i])
		}
	}
}

func TestLinTranGradient(t *testing.T) {
	funcTest := &FuncTest{
		F:     ComposedFunc{linTranTestMat1, linTranTestMat2, AddTwice{}},
		Vars:  linTranTestVariables,
		Input: linTranTestVec,
	}
	funcTest.Run(t)
}

func TestLinTranRGradient(t *testing.T) {
	funcTest := &RFuncTest{
		F:     ComposedRFunc{linTranTestMat1, linTranTestMat2, AddTwice{}},
		Vars:  linTranTestVariables,
		Input: NewRVariable(linTranTestVec, linTranTestRVec),
		RV:    linTranTestRVec,
	}
	funcTest.Run(t)
}
