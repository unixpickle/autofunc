package seqfunctest

import (
	"reflect"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
)

var ConcatSeqs = [][]*autofunc.Variable{
	{TestVars[2], TestVars[4], TestVars[3]},
	{TestVars[0], TestVars[1], TestVars[2]},
	{TestVars[3]},
	{TestVars[4]},
	{TestVars[2], TestVars[4]},
}

type ConcatTestFunc struct{}

func (c *ConcatTestFunc) ApplySeqs(in seqfunc.Result) seqfunc.Result {
	return seqfunc.ConcatInner(in, seqfunc.VarResult(ConcatSeqs), in)
}

func (c *ConcatTestFunc) ApplySeqsR(rv autofunc.RVector, in seqfunc.RResult) seqfunc.RResult {
	return seqfunc.ConcatInnerR(in, seqfunc.VarRResult(rv, ConcatSeqs), in)
}

func TestConcat(t *testing.T) {
	sc := &functest.SeqRFuncChecker{
		F:     &ConcatTestFunc{},
		Vars:  TestVars,
		Input: TestSeqs,
		RV:    TestRV,
	}
	sc.FullCheck(t)
}

func TestConcatOutput(t *testing.T) {
	seqs1 := seqfunc.ConstResult([][]linalg.Vector{
		{{1, 2}, {3}, {4, 5, 6}},
		{{1}},
		{},
		{{-2, -3}},
	})
	seqs2 := seqfunc.ConstResult([][]linalg.Vector{
		{{3}, {-5, 4}, {}},
		{{-3}},
		{},
		{{4, 3, 2, 1}},
	})
	actual := seqfunc.ConcatInner(seqs1, seqs2).OutputSeqs()
	expected := [][]linalg.Vector{
		{{1, 2, 3}, {3, -5, 4}, {4, 5, 6}},
		{{1, -3}},
		{},
		{{-2, -3, 4, 3, 2, 1}},
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("expected %v got %v", expected, actual)
	}
}
