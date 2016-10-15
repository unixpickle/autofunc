package functest

import (
	"testing"

	"github.com/unixpickle/autofunc"
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

func TestAddTwiceSeq(t *testing.T) {
	fc := &SeqFuncChecker{
		F:     addTwiceSeq{},
		Vars:  TestVars,
		Input: TestSeqs,
	}
	Check(t, fc)
}

func TestAddTwiceSeqR(t *testing.T) {
	fc := &SeqRFuncChecker{
		F:     addTwiceSeq{},
		Vars:  TestVars,
		Input: TestSeqs,
		RV:    TestRV,
	}
	CheckR(t, fc)
}

func TestMulTwiceSeq(t *testing.T) {
	fc := &SeqFuncChecker{
		F:     mulTwiceSeq{},
		Vars:  TestVars,
		Input: TestSeqs,
	}
	Check(t, fc)
}

func TestMulTwiceSeqR(t *testing.T) {
	fc := &SeqRFuncChecker{
		F:     mulTwiceSeq{},
		Vars:  TestVars,
		Input: TestSeqs,
		RV:    TestRV,
	}
	CheckR(t, fc)
}
