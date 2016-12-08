package autofunc

import (
	"math/rand"
	"testing"

	. "github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
)

type foldTestFunc struct {
	initState *Variable
}

func (f *foldTestFunc) Apply(in Result) Result {
	ins := Split(len(in.Output())/2, in)
	s := f.initState
	return Fold(s, ins, func(state Result, in Result) Result {
		return Mul(Inverse(state), in)
	})
}

func (f *foldTestFunc) ApplyR(rv RVector, in RResult) RResult {
	ins := SplitR(len(in.Output())/2, in)
	s := NewRVariable(f.initState, rv)
	return FoldR(s, ins, func(state RResult, in RResult) RResult {
		return MulR(InverseR(state), in)
	})
}

func TestFoldOut(t *testing.T) {
	vars := []*Variable{
		&Variable{Vector: []float64{1, 2, 2.5, 1.5, 3, 0.5}},
		&Variable{Vector: []float64{2, 1}},
	}
	f := &foldTestFunc{initState: vars[1]}
	actual := f.Apply(vars[0]).Output()
	expected := []float64{0.6, 2.0 / 3.0}
	if actual.Copy().Scale(-1).Add(expected).MaxAbs() > 1e-5 {
		t.Errorf("expected %v but got %v", expected, actual)
	}
}

func TestFold(t *testing.T) {
	vars := []*Variable{
		&Variable{Vector: []float64{1, 2, 2.5, 1.5, 3, 0.5}},
		&Variable{Vector: []float64{2, 1}},
	}
	rv := RVector{}
	for _, v := range vars {
		rv[v] = make(linalg.Vector, len(v.Vector))
		for i := range rv[v] {
			rv[v][i] = rand.NormFloat64()
		}
	}
	ch := &functest.RFuncChecker{
		F:     &foldTestFunc{initState: vars[1]},
		Vars:  vars,
		Input: vars[0],
		RV:    rv,
	}
	ch.FullCheck(t)
}
