package seqfunctest

import (
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/autofunc/seqfunc"
)

var (
	TestVars = []*autofunc.Variable{
		&autofunc.Variable{Vector: []float64{1, -3, 2, 0.5}},
		&autofunc.Variable{Vector: []float64{3, -3, 0, 0.5}},
		&autofunc.Variable{Vector: []float64{2, -3, 2, -0.5}},
		&autofunc.Variable{Vector: []float64{0, -3, -5, 0.5}},
	}
	TestSeqs = [][]*autofunc.Variable{
		{TestVars[0], TestVars[1], TestVars[2]},
		{TestVars[2], TestVars[0], TestVars[3]},
		{TestVars[1]},
		{TestVars[3]},
		{TestVars[0], TestVars[3]},
	}
	TestRV = autofunc.RVector{
		TestVars[0]: []float64{-1, 1, 0.33, -0.33},
		TestVars[3]: []float64{2, 0.33, -5, 0.33},
	}
)

func TestMapFunc(t *testing.T) {
	mf := &seqfunc.MapFunc{F: autofunc.Sin{}}
	r := &functest.SeqFuncChecker{
		F:     mf,
		Vars:  TestVars,
		Input: TestSeqs,
	}
	r.FullCheck(t)
}

func TestMapRFunc(t *testing.T) {
	mf := &seqfunc.MapRFunc{F: autofunc.Sin{}}
	r := &functest.SeqRFuncChecker{
		F:     mf,
		Vars:  TestVars,
		Input: TestSeqs,
		RV:    TestRV,
	}
	r.FullCheck(t)
}
