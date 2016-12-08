package seqfunctest

import (
	"reflect"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
)

type AddAllTestFunc struct{}

func (p *AddAllTestFunc) Apply(in autofunc.Result) autofunc.Result {
	return seqfunc.AddAll(seqfunc.VarResult(TestSeqs))
}

func (p *AddAllTestFunc) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	return seqfunc.AddAllR(seqfunc.VarRResult(rv, TestSeqs))
}

type ConcatAllTestFunc struct{}

func (p *ConcatAllTestFunc) Apply(in autofunc.Result) autofunc.Result {
	s := seqfunc.VarResult(TestSeqs)
	return autofunc.Concat(in, seqfunc.ConcatAll(s))
}

func (p *ConcatAllTestFunc) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	s := seqfunc.VarRResult(rv, TestSeqs)
	return autofunc.ConcatR(in, seqfunc.ConcatAllR(s))
}

func TestAddAllOutput(t *testing.T) {
	in := [][]linalg.Vector{
		{{1, 2}, {3, -1}},
		{{-1.5, 0.3}},
	}
	actual := seqfunc.AddAll(seqfunc.ConstResult(in)).Output()
	expected := linalg.Vector([]float64{2.5, 1.3})
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected %v but got %v", expected, actual)
	}
}

func TestAddAll(t *testing.T) {
	checker := &functest.RFuncChecker{
		F:     &AddAllTestFunc{},
		Input: TestVars[0],
		Vars:  TestVars,
		RV:    TestRV,
	}
	checker.FullCheck(t)
}

func TestConcatAllOutput(t *testing.T) {
	in := [][]linalg.Vector{
		{{1, 2}, {3, 4}},
		{{5}, {6, 7, 8}, {9}},
		{},
		{{10, 11, 12, 13, 14, 15}},
	}
	inSeq := seqfunc.ConstResult(in)
	actual := seqfunc.ConcatAll(inSeq).Output()
	expected := make(linalg.Vector, 15)
	for i := range expected {
		expected[i] = float64(i) + 1
	}
	if expected.Copy().Scale(-1).Add(actual).MaxAbs() > 1e-6 {
		t.Errorf("expected %v got %v", expected, actual)
	}
}

func TestConcatAll(t *testing.T) {
	checker := &functest.RFuncChecker{
		F:     &ConcatAllTestFunc{},
		Input: TestVars[0],
		Vars:  TestVars,
		RV:    TestRV,
	}
	checker.FullCheck(t)
}
