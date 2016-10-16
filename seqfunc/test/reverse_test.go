package seqfunctest

import (
	"reflect"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
)

type ReverseTestFunc struct{}

func (r *ReverseTestFunc) ApplySeqs(in seqfunc.Result) seqfunc.Result {
	return seqfunc.Reverse(in)
}

func (r *ReverseTestFunc) ApplySeqsR(rv autofunc.RVector, in seqfunc.RResult) seqfunc.RResult {
	return seqfunc.ReverseR(in)
}

func TestReverseOutput(t *testing.T) {
	in := seqfunc.ConstResult([][]linalg.Vector{
		{{1, 2}, {4, -1, 2}, {3, 4, 5}},
		{{1, 2}},
		{},
	})
	actual := seqfunc.Reverse(in).OutputSeqs()
	expected := [][]linalg.Vector{
		{{3, 4, 5}, {4, -1, 2}, {1, 2}},
		{{1, 2}},
		{},
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected %v got %v", expected, actual)
	}
}

func TestReverse(t *testing.T) {
	checker := &functest.SeqRFuncChecker{
		F:     &ReverseTestFunc{},
		Input: TestSeqs,
		Vars:  TestVars,
		RV:    TestRV,
	}
	checker.FullCheck(t)
}
