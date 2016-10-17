package seqfunctest

import (
	"reflect"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
)

type PoolTestFunc struct{}

func (p *PoolTestFunc) ApplySeqs(in seqfunc.Result) seqfunc.Result {
	return seqfunc.Pool(in, func(in seqfunc.Result) seqfunc.Result {
		return seqfunc.Map(seqfunc.ConcatInner(in, in), autofunc.Square)
	})
}

func (p *PoolTestFunc) ApplySeqsR(rv autofunc.RVector, in seqfunc.RResult) seqfunc.RResult {
	return seqfunc.PoolR(in, func(in seqfunc.RResult) seqfunc.RResult {
		return seqfunc.MapR(seqfunc.ConcatInnerR(in, in), autofunc.SquareR)
	})
}

func TestPoolOutput(t *testing.T) {
	in := [][]linalg.Vector{
		{{1, 2}, {3, -1}},
		{{-1}},
	}
	ptf := PoolTestFunc{}
	actual := ptf.ApplySeqs(seqfunc.ConstResult(in)).OutputSeqs()
	expected := [][]linalg.Vector{
		{{1, 4, 1, 4}, {9, 1, 9, 1}},
		{{1, 1}},
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected %v but got %v", expected, actual)
	}
}

func TestPool(t *testing.T) {
	checker := &functest.SeqRFuncChecker{
		F:     &PoolTestFunc{},
		Input: TestSeqs,
		Vars:  TestVars,
		RV:    TestRV,
	}
	checker.FullCheck(t)
}
