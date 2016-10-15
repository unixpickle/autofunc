package seqfunctest

import (
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/autofunc/seqfunc"
)

var (
	TranMatrix = &autofunc.Variable{
		Vector: []float64{
			0.35771, 0.87735, 0.82339, -0.11739,
			-0.56811, 0.56496, 0.41425, 0.14371,
			-0.82190, 0.24597, -0.23267, 0.50681,
			-0.82619, -0.59787, -0.76291, -0.96163,
		},
	}
	TestVars = []*autofunc.Variable{
		&autofunc.Variable{Vector: []float64{1, -3, 2, 0.5}},
		&autofunc.Variable{Vector: []float64{3, -3, 0, 0.5}},
		&autofunc.Variable{Vector: []float64{2, -3, 2, -0.5}},
		&autofunc.Variable{Vector: []float64{0, -3, -5, 0.5}},
		&autofunc.Variable{Vector: []float64{-2, 0.5, 3}},
		TranMatrix,
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
		TestVars[4]: []float64{1, 2, 3},
		TranMatrix: []float64{
			-0.31895, -0.47222, 0.58130, 0.54394,
			-0.99254, -0.42368, -0.56713, 0.15739,
			0.28701, 0.92360, -0.33617, -0.42437,
			-0.56957, -0.97235, 0.66386, 0.22420,
		},
	}
)

func TestMapFunc(t *testing.T) {
	linTran := &autofunc.LinTran{
		Rows: 4,
		Cols: 4,
		Data: TranMatrix,
	}
	mf := &seqfunc.MapFunc{F: linTran}
	r := &functest.SeqFuncChecker{
		F:     mf,
		Vars:  TestVars,
		Input: TestSeqs,
	}
	r.FullCheck(t)
}

func TestMapRFunc(t *testing.T) {
	linTran := &autofunc.LinTran{
		Rows: 4,
		Cols: 4,
		Data: TranMatrix,
	}
	mf := &seqfunc.MapRFunc{F: linTran}
	r := &functest.SeqRFuncChecker{
		F:     mf,
		Vars:  TestVars,
		Input: TestSeqs,
		RV:    TestRV,
	}
	r.FullCheck(t)
}
