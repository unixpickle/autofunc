package autofunc

import (
	"math"
	"math/rand"
	"testing"

	. "github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
)

var (
	mathFuncTestVec    = &Variable{Vector: linalg.Vector([]float64{1, -0.5, 0.3, 0.7})}
	mathFuncTestVecPos = &Variable{Vector: linalg.Vector([]float64{1, 0.5, 0.3, 0.7})}
	mathFuncTestVars   = []*Variable{mathFuncTestVec}
	mathFuncTestRVec   = RVector{
		mathFuncTestVec: linalg.Vector([]float64{0.5, -10, 5, 3.14}),
	}
)

func TestExp(t *testing.T) {
	f := &functest.RFuncChecker{
		F:     Exp{},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
		RV:    mathFuncTestRVec,
	}
	f.FullCheck(t)
}

func TestLog(t *testing.T) {
	f := &functest.RFuncChecker{
		F:     Log{},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVecPos,
		RV:    mathFuncTestRVec,
	}
	f.FullCheck(t)
}

func TestSquaredNorm(t *testing.T) {
	f := &functest.RFuncChecker{
		F:     SquaredNorm{},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
		RV:    mathFuncTestRVec,
	}
	f.FullCheck(t)
}

func TestSigmoid(t *testing.T) {
	f := &functest.RFuncChecker{
		F:     ComposedRFunc{Sigmoid{}, Sigmoid{}, Sigmoid{}},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
		RV:    mathFuncTestRVec,
	}
	f.FullCheck(t)
}

func TestLogSigmoid(t *testing.T) {
	f := &functest.RFuncChecker{
		F:     ComposedRFunc{LogSigmoid{}, LogSigmoid{}, LogSigmoid{}},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
		RV:    mathFuncTestRVec,
	}
	f.FullCheck(t)
}

func TestSoftmax(t *testing.T) {
	f := &functest.RFuncChecker{
		F:     &Softmax{},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
		RV:    mathFuncTestRVec,
	}
	f.FullCheck(t)
}

func TestSoftmax3(t *testing.T) {
	f := &functest.RFuncChecker{
		F:     &Softmax{3},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
		RV:    mathFuncTestRVec,
	}
	f.FullCheck(t)
}

func TestSin(t *testing.T) {
	f := &functest.RFuncChecker{
		F:     Sin{},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
		RV:    mathFuncTestRVec,
	}
	f.FullCheck(t)
}

func TestCosOutput(t *testing.T) {
	cosFunc := Cos{}
	for i := 0; i < 10; i++ {
		arg := rand.Float64()*20 - 10
		output := cosFunc.Apply(&Variable{Vector: []float64{arg}}).Output()[0]
		if math.Abs(output-math.Cos(arg)) > 1e-5 {
			t.Error("argument", arg, "gave", output, "but expected", math.Cos(arg))
		}
	}
}

func TestNormOutput(t *testing.T) {
	vec := &Variable{Vector: []float64{3, 4}}
	expected := 5.0
	actual := Norm{}.Apply(vec)
	if math.Abs(actual.Output()[0]-expected) > 1e-5 {
		t.Errorf("expected %v got %v", expected, actual)
	}
}

func TestNorm(t *testing.T) {
	f := &functest.RFuncChecker{
		F:     Norm{},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
		RV:    mathFuncTestRVec,
	}
	f.FullCheck(t)
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
		res := s.Apply(inputVar)
		res.PropagateGradient(inputVec, bogusGrad)
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
		res := s.Apply(inputVar)
		res.PropagateGradient(inputVec, bogusGrad)
	}
}

func BenchmarkSquaredNorm(b *testing.B) {
	vec := make(linalg.Vector, 500)
	for i := range vec {
		vec[i] = rand.NormFloat64()
	}
	v := &Variable{Vector: vec}
	g := NewGradient([]*Variable{v})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SquaredNorm{}.Apply(v).PropagateGradient([]float64{1}, g)
	}
}

func BenchmarkNorm(b *testing.B) {
	vec := make(linalg.Vector, 500)
	for i := range vec {
		vec[i] = rand.NormFloat64()
	}
	v := &Variable{Vector: vec}
	g := NewGradient([]*Variable{v})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Norm{}.Apply(v).PropagateGradient([]float64{1}, g)
	}
}
