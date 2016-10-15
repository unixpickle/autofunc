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

func TestExpGradient(t *testing.T) {
	f := &functest.FuncChecker{
		F:     Exp{},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
	}
	f.FullCheck(t)
}

func TestExpRGradient(t *testing.T) {
	f := &functest.RFuncChecker{
		F:     Exp{},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
		RV:    mathFuncTestRVec,
	}
	f.FullCheck(t)
}

func TestLogGradient(t *testing.T) {
	f := &functest.FuncChecker{
		F:     Log{},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVecPos,
	}
	f.FullCheck(t)
}

func TestLogRGradient(t *testing.T) {
	f := &functest.RFuncChecker{
		F:     Log{},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVecPos,
		RV:    mathFuncTestRVec,
	}
	f.FullCheck(t)
}

func TestSquaredNormGradient(t *testing.T) {
	f := &functest.FuncChecker{
		F:     SquaredNorm{},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
	}
	f.FullCheck(t)
}

func TestSquaredNormRGradient(t *testing.T) {
	f := &functest.RFuncChecker{
		F:     SquaredNorm{},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
		RV:    mathFuncTestRVec,
	}
	f.FullCheck(t)
}

func TestSigmoidGradient(t *testing.T) {
	f := &functest.FuncChecker{
		F:     ComposedFunc{Sigmoid{}, Sigmoid{}, Sigmoid{}},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
	}
	f.FullCheck(t)
}

func TestSigmoidRGradient(t *testing.T) {
	f := &functest.RFuncChecker{
		F:     ComposedRFunc{Sigmoid{}, Sigmoid{}, Sigmoid{}},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
		RV:    mathFuncTestRVec,
	}
	f.FullCheck(t)
}

func TestLogSigmoidGradient(t *testing.T) {
	f := &functest.FuncChecker{
		F:     ComposedFunc{LogSigmoid{}, LogSigmoid{}, LogSigmoid{}},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
	}
	f.FullCheck(t)
}

func TestLogSigmoidRGradient(t *testing.T) {
	f := &functest.RFuncChecker{
		F:     ComposedRFunc{LogSigmoid{}, LogSigmoid{}, LogSigmoid{}},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
		RV:    mathFuncTestRVec,
	}
	f.FullCheck(t)
}

func TestSoftmaxGradient(t *testing.T) {
	f := &functest.FuncChecker{
		F:     &Softmax{},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
	}
	f.FullCheck(t)
}

func TestSoftmaxGradient3(t *testing.T) {
	f := &functest.FuncChecker{
		F:     &Softmax{3},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
	}
	f.FullCheck(t)
}

func TestSoftmaxRGradient(t *testing.T) {
	f := &functest.RFuncChecker{
		F:     &Softmax{},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
		RV:    mathFuncTestRVec,
	}
	f.FullCheck(t)
}

func TestSoftmaxRGradient3(t *testing.T) {
	f := &functest.RFuncChecker{
		F:     &Softmax{3},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
		RV:    mathFuncTestRVec,
	}
	f.FullCheck(t)
}

func TestSinGradient(t *testing.T) {
	f := &functest.FuncChecker{
		F:     Sin{},
		Vars:  mathFuncTestVars,
		Input: mathFuncTestVec,
	}
	f.FullCheck(t)
}

func TestSinRGradient(t *testing.T) {
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
