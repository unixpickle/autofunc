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

type ConcatAllTestFunc struct{}

func (p *ConcatAllTestFunc) Apply(in autofunc.Result) autofunc.Result {
	s := seqfunc.VarResult(TestSeqs)
	return autofunc.Concat(in, seqfunc.ConcatAll(s))
}

func (p *ConcatAllTestFunc) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	s := seqfunc.VarRResult(rv, TestSeqs)
	return autofunc.ConcatR(in, seqfunc.ConcatAllR(s))
}

type ConcatLastTestFunc struct{}

func (c *ConcatLastTestFunc) Apply(in autofunc.Result) autofunc.Result {
	seqs := append([][]*autofunc.Variable{{}}, TestSeqs...)
	s := seqfunc.VarResult(seqs)
	return autofunc.Concat(in, seqfunc.ConcatLast(s))
}

func (c *ConcatLastTestFunc) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	seqs := append([][]*autofunc.Variable{{}}, TestSeqs...)
	s := seqfunc.VarRResult(rv, seqs)
	return autofunc.ConcatR(in, seqfunc.ConcatLastR(s))
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

func TestConcatLastOutput(t *testing.T) {
	in := [][]linalg.Vector{
		{{1, 2}, {3, 4}},
		{{5}, {6, 7, 8}, {9}},
		{},
		{{10, 11, 12, 13, 14, 15}},
	}
	inSeq := seqfunc.ConstResult(in)
	actual := seqfunc.ConcatLast(inSeq).Output()
	expected := linalg.Vector([]float64{3, 4, 9, 10, 11, 12, 13, 14, 15})
	if expected.Copy().Scale(-1).Add(actual).MaxAbs() > 1e-6 {
		t.Errorf("expected %v got %v", expected, actual)
	}
}

func TestConcatLast(t *testing.T) {
	checker := &functest.RFuncChecker{
		F:     &ConcatLastTestFunc{},
		Input: TestVars[0],
		Vars:  TestVars,
		RV:    TestRV,
	}
	checker.FullCheck(t)
}
