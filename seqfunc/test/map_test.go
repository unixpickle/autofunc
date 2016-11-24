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
	TestLinTran = &autofunc.LinTran{
		Rows: 4,
		Cols: 4,
		Data: TranMatrix,
	}
)

type MapNTestFunc struct{}

func (_ MapNTestFunc) ApplySeqs(in seqfunc.Result) seqfunc.Result {
	return seqfunc.MapN(func(ins ...autofunc.Result) autofunc.Result {
		return autofunc.Add(ins[0], autofunc.Scale(ins[1], -2))
	}, in, seqfunc.VarResult(TestSeqs))
}

func (_ MapNTestFunc) ApplySeqsR(rv autofunc.RVector, in seqfunc.RResult) seqfunc.RResult {
	return seqfunc.MapNR(func(ins ...autofunc.RResult) autofunc.RResult {
		return autofunc.AddR(ins[0], autofunc.ScaleR(ins[1], -2))
	}, in, seqfunc.VarRResult(rv, TestSeqs))
}

func TestMapFunc(t *testing.T) {
	mf := &seqfunc.MapFunc{F: TestLinTran}
	r := &functest.SeqFuncChecker{
		F:     mf,
		Vars:  TestVars,
		Input: TestSeqs,
	}
	r.FullCheck(t)
}

func TestMapRFunc(t *testing.T) {
	mf := &seqfunc.MapRFunc{F: TestLinTran}
	r := &functest.SeqRFuncChecker{
		F:     mf,
		Vars:  TestVars,
		Input: TestSeqs,
		RV:    TestRV,
	}
	r.FullCheck(t)
}

func TestMapBatcherOutput(t *testing.T) {
	mf := &seqfunc.MapFunc{F: TestLinTran}
	mb := &seqfunc.MapBatcher{B: TestLinTran}

	expected := mf.ApplySeqs(seqfunc.VarResult(TestSeqs)).OutputSeqs()
	actual := mb.ApplySeqs(seqfunc.VarResult(TestSeqs)).OutputSeqs()

	equal := func() bool {
		if len(expected) != len(actual) {
			return false
		}
		for i, x := range expected {
			y := actual[i]
			if len(x) != len(y) {
				return false
			}
			for j, xVec := range x {
				yVec := y[j]
				if len(xVec) != len(yVec) {
					return false
				}
				if xVec.Copy().Scale(-1).Add(yVec).MaxAbs() > 1e-5 {
					return false
				}
			}
		}
		return true
	}()

	if !equal {
		t.Errorf("expected %v got %v", expected, actual)
	}
}

func TestMapBatcher(t *testing.T) {
	mf := &seqfunc.MapBatcher{B: TestLinTran}
	r := &functest.SeqFuncChecker{
		F:     mf,
		Vars:  TestVars,
		Input: TestSeqs,
	}
	r.FullCheck(t)
}

func TestFixedMapBatcher(t *testing.T) {
	mf := &seqfunc.FixedMapBatcher{B: TestLinTran, BatchSize: 2}
	r := &functest.SeqFuncChecker{
		F:     mf,
		Vars:  TestVars,
		Input: TestSeqs,
	}
	r.FullCheck(t)
}

func TestMapRBatcher(t *testing.T) {
	mf := &seqfunc.MapRBatcher{B: TestLinTran}
	r := &functest.SeqRFuncChecker{
		F:     mf,
		Vars:  TestVars,
		Input: TestSeqs,
		RV:    TestRV,
	}
	r.FullCheck(t)
}

func TestFixedMapRBatcher(t *testing.T) {
	mf := &seqfunc.FixedMapRBatcher{B: TestLinTran, BatchSize: 2}
	r := &functest.SeqRFuncChecker{
		F:     mf,
		Vars:  TestVars,
		Input: TestSeqs,
		RV:    TestRV,
	}
	r.FullCheck(t)
}

func TestMapMulti(t *testing.T) {
	r := &functest.SeqRFuncChecker{
		F:     MapNTestFunc{},
		Vars:  TestVars,
		Input: TestSeqs,
		RV:    TestRV,
	}
	r.FullCheck(t)
}

func TestMixedMatcher(t *testing.T) {
	m := &mixedMapBatcher{
		fb: &seqfunc.FixedMapBatcher{B: TestLinTran, BatchSize: 2},
		br: &seqfunc.MapRBatcher{B: TestLinTran},
	}
	r := &functest.SeqRFuncChecker{
		F:     m,
		Vars:  TestVars,
		Input: TestSeqs,
		RV:    TestRV,
	}
	r.FullCheck(t)
}

type mixedMapBatcher struct {
	fb *seqfunc.FixedMapBatcher
	br *seqfunc.MapRBatcher
}

func (m *mixedMapBatcher) ApplySeqs(r seqfunc.Result) seqfunc.Result {
	return m.fb.ApplySeqs(r)
}

func (m *mixedMapBatcher) ApplySeqsR(rv autofunc.RVector, r seqfunc.RResult) seqfunc.RResult {
	return m.br.ApplySeqsR(rv, r)
}
