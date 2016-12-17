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
