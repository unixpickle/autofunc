package autofunc

import (
	"testing"

	"github.com/unixpickle/num-analysis/linalg"
)

var (
	sigmoidTestVec  = &Variable{Vector: linalg.Vector([]float64{1, -0.5, 0.3, 0.7})}
	sigmoidTestVars = []*Variable{sigmoidTestVec}
	sigmoidTestRVec = RVector{
		sigmoidTestVec: linalg.Vector([]float64{0.5, -10, 5, 3.14}),
	}
)

func TestSigmoidGradient(t *testing.T) {
	funcTest := &FuncTest{
		F:     ComposedFunc{Sigmoid{}, Sigmoid{}, Sigmoid{}},
		Vars:  sigmoidTestVars,
		Input: sigmoidTestVec,
	}
	funcTest.Run(t)
}

func TestSigmoidRGradient(t *testing.T) {
	funcTest := &RFuncTest{
		F:     ComposedRFunc{Sigmoid{}, Sigmoid{}, Sigmoid{}},
		Vars:  sigmoidTestVars,
		Input: NewRVariable(sigmoidTestVec, sigmoidTestRVec),
		RV:    sigmoidTestRVec,
	}
	funcTest.Run(t)
}
