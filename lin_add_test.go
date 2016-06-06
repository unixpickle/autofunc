package autofunc

import (
	"testing"

	"github.com/unixpickle/num-analysis/linalg"
)

var (
	linAddTestVec1 = &Variable{Vector: linalg.Vector([]float64{1, 3, 2, 1})}
	linAddTestVec2 = &Variable{Vector: linalg.Vector([]float64{5, -3, 5, 4})}
	linAddTestVars = []*Variable{linAddTestVec1, linAddTestVec2}
	linAddTestRVec = RVector{
		linAddTestVec1: linalg.Vector([]float64{0.5, -10, 5, 3.14}),
		linAddTestVec2: linalg.Vector([]float64{0, 5, -15, -30}),
	}
)

func TestLinAddGradient(t *testing.T) {
	addFunc := ComposedFunc{&LinAdd{Var: linAddTestVec1}, &LinAdd{Var: linAddTestVec2}, AddTwice{}}
	f := &FuncTest{
		F:     addFunc,
		Vars:  linAddTestVars,
		Input: linAddTestVec2,
	}
	f.Run(t)
}

func TestLinAddRGradient(t *testing.T) {
	addFunc := ComposedRFunc{&LinAdd{Var: linAddTestVec1}, &LinAdd{Var: linAddTestVec2}, AddTwice{}}
	f := &RFuncTest{
		F:     addFunc,
		Vars:  linAddTestVars,
		Input: NewRVariable(linAddTestVec2, linAddTestRVec),
		RV:    linAddTestRVec,
	}
	f.Run(t)
}
