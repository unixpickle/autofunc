package autofunc

import (
	"testing"

	"github.com/unixpickle/num-analysis/linalg"
)

var (
	slicesTestVec1 = &Variable{
		Vector: linalg.Vector([]float64{1, 2, -4, 3, -2}),
	}
	slicesTestVec2 = &Variable{
		Vector: linalg.Vector([]float64{4, -2, 3, -0.5, 0.3}),
	}
	slicesTestVec3 = &Variable{
		Vector: linalg.Vector([]float64{0.99196, 0.48826, 0.51066, 0.66715, 0.44423}),
	}
	slicesTestVars = []*Variable{
		slicesTestVec1, slicesTestVec2, slicesTestVec3,
	}
	slicesTestRVec = RVector{
		slicesTestVec1: linalg.Vector([]float64{
			0.340162, -0.325063, 0.179612, 0.056463, -0.812274,
		}),

		// Vector 2 intentionally left out to see what happens
		// when a vector isn't in the RVector.

		slicesTestVec3: linalg.Vector([]float64{
			0.59824, -0.63322, 0.13379, 0.99559, 0.53748,
		}),
	}
)

type concatTestFunc struct{}

func (_ concatTestFunc) Apply(r Result) Result {
	return Concat(slicesTestVec1, r, slicesTestVec3)
}

func (_ concatTestFunc) ApplyR(v RVector, r RResult) RResult {
	v1 := NewRVariable(slicesTestVec1, slicesTestRVec)
	v3 := NewRVariable(slicesTestVec3, slicesTestRVec)
	return ConcatR(v1, r, v3)
}

func TestConcatGradients(t *testing.T) {
	f := &FuncTest{
		F:     concatTestFunc{},
		Vars:  slicesTestVars,
		Input: slicesTestVec2,
	}
	f.Run(t)
}

func TestConcatRGradients(t *testing.T) {
	f := &RFuncTest{
		F:     concatTestFunc{},
		Vars:  slicesTestVars,
		Input: slicesTestVec2,
		RV:    slicesTestRVec,
	}
	f.Run(t)
}

type sliceTestFunc struct {
	WrapInput bool
}

func (s *sliceTestFunc) Apply(r Result) Result {
	if s.WrapInput {
		r = Add(r, Scale(r, 0))
	}
	return Slice(r, 1, 3)
}

func (s *sliceTestFunc) ApplyR(v RVector, r RResult) RResult {
	if s.WrapInput {
		r = AddR(r, ScaleR(r, 0))
	}
	return SliceR(r, 1, 3)
}

func TestSliceGradients(t *testing.T) {
	f := &FuncTest{
		F:     &sliceTestFunc{},
		Vars:  slicesTestVars,
		Input: slicesTestVec1,
	}
	f.Run(t)
}

func TestSliceRGradients(t *testing.T) {
	f := &RFuncTest{
		F:     &sliceTestFunc{},
		Vars:  slicesTestVars,
		Input: slicesTestVec1,
		RV:    slicesTestRVec,
	}
	f.Run(t)
}

func TestWrappedSliceGradients(t *testing.T) {
	f := &FuncTest{
		F:     &sliceTestFunc{WrapInput: true},
		Vars:  slicesTestVars,
		Input: slicesTestVec1,
	}
	f.Run(t)
}

func TestWrappedSliceRGradients(t *testing.T) {
	f := &RFuncTest{
		F:     &sliceTestFunc{WrapInput: true},
		Vars:  slicesTestVars,
		Input: slicesTestVec1,
		RV:    slicesTestRVec,
	}
	f.Run(t)
}
