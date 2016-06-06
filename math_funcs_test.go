package autofunc

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/num-analysis/linalg"
)

var (
	mathFuncTestVec  = &Variable{Vector: linalg.Vector([]float64{1, -0.5, 0.3, 0.7})}
	mathFuncTestVars = []*Variable{mathFuncTestVec}
	mathFuncTestRVec = RVector{
		mathFuncTestVec: linalg.Vector([]float64{0.5, -10, 5, 3.14}),
	}
)

func TestExpGradient(t *testing.T) {
	funcTest := &FuncTest{
		F:     Exp{},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
	}
	funcTest.Run(t)
}

func TestExpRGradient(t *testing.T) {
	funcTest := &RFuncTest{
		F:     Exp{},
		Vars:  mathFuncTestVars,
		Input: NewRVariable(mathFuncTestVec, mathFuncTestRVec),
		RV:    mathFuncTestRVec,
	}
	funcTest.Run(t)
}

func TestSigmoidGradient(t *testing.T) {
	funcTest := &FuncTest{
		F:     ComposedFunc{Sigmoid{}, Sigmoid{}, Sigmoid{}},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
	}
	funcTest.Run(t)
}

func TestSigmoidRGradient(t *testing.T) {
	funcTest := &RFuncTest{
		F:     ComposedRFunc{Sigmoid{}, Sigmoid{}, Sigmoid{}},
		Vars:  mathFuncTestVars,
		Input: NewRVariable(mathFuncTestVec, mathFuncTestRVec),
		RV:    mathFuncTestRVec,
	}
	funcTest.Run(t)
}

func TestSoftmaxGradient(t *testing.T) {
	funcTest := &FuncTest{
		F:     &Softmax{},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
	}
	funcTest.Run(t)
	funcTest.F = &Softmax{3}
	funcTest.Run(t)
}

func TestSoftmaxRGradient(t *testing.T) {
	funcTest := &RFuncTest{
		F:     &Softmax{},
		Vars:  mathFuncTestVars,
		Input: NewRVariable(mathFuncTestVec, mathFuncTestRVec),
		RV:    mathFuncTestRVec,
	}
	funcTest.Run(t)
	funcTest.F = &Softmax{3}
	funcTest.Run(t)
}

func BenchmarkSoftmaxTemp(b *testing.B) {
	rand.Seed(123)
	inputVec := make(linalg.Vector, 3000)
	for i := range inputVec {
		inputVec[i] = rand.Float64()*5 - 2.5
	}
	inputVar := &Variable{Vector: inputVec}
	bogusGrad := NewGradient([]*Variable{inputVar})

	b.ResetTimer()

	s := Softmax{Temperature: 15}
	for i := 0; i < b.N; i++ {
		s.Apply(inputVar).PropagateGradient(inputVec, bogusGrad)
	}
}

func BenchmarkSoftmaxNoTemp(b *testing.B) {
	rand.Seed(123)
	inputVec := make(linalg.Vector, 3000)
	for i := range inputVec {
		inputVec[i] = rand.Float64()*5 - 2.5
	}
	inputVar := &Variable{Vector: inputVec}
	bogusGrad := NewGradient([]*Variable{inputVar})

	b.ResetTimer()

	s := Softmax{}
	for i := 0; i < b.N; i++ {
		s.Apply(inputVar).PropagateGradient(inputVec, bogusGrad)
	}
}
