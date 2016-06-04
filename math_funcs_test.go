package autofunc

import (
	"testing"

	"github.com/unixpickle/num-analysis/linalg"
)

var (
	mathFuncTestVec  = &Variable{Vector: linalg.Vector([]float64{1, -0.5, 0.3, 0.7})}
	mathFuncTestVars = []*Variable{mathFuncTestVec}
	mathFuncTestRVec = RVector{
		mathFuncTestVec: linalg.Vector([]float64{0.5, -10, 5, 3.14}),
	}
)

func TestExpGradient(t *testing.T) {
	funcTest := &FuncTest{
		F:     Exp{},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
	}
	funcTest.Run(t)
}

func TestExpRGradient(t *testing.T) {
	funcTest := &RFuncTest{
		F:     Exp{},
		Vars:  mathFuncTestVars,
		Input: NewRVariable(mathFuncTestVec, mathFuncTestRVec),
		RV:    mathFuncTestRVec,
	}
	funcTest.Run(t)
}

func TestSigmoidGradient(t *testing.T) {
	funcTest := &FuncTest{
		F:     ComposedFunc{Sigmoid{}, Sigmoid{}, Sigmoid{}},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
	}
	funcTest.Run(t)
}

func TestSigmoidRGradient(t *testing.T) {
	funcTest := &RFuncTest{
		F:     ComposedRFunc{Sigmoid{}, Sigmoid{}, Sigmoid{}},
		Vars:  mathFuncTestVars,
		Input: NewRVariable(mathFuncTestVec, mathFuncTestRVec),
		RV:    mathFuncTestRVec,
	}
	funcTest.Run(t)
}
