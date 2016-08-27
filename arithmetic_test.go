package autofunc

import (
	"math"
	"testing"

	"github.com/unixpickle/num-analysis/linalg"
)

var (
	arithmeticTestVec1 = &Variable{
		Vector: linalg.Vector([]float64{1, 2, -4, 3, -2}),
	}
	arithmeticTestVec2 = &Variable{
		Vector: linalg.Vector([]float64{4, -2, 3, -0.5, 0.3}),
	}
	arithmeticTestVec3 = &Variable{
		Vector: linalg.Vector([]float64{0.99196, 0.48826, 0.51066, 0.66715, 0.44423}),
	}
	arithmeticTestVec4 = &Variable{
		Vector: linalg.Vector([]float64{0.509893, 0.874112, -0.080468, 0.372740, -0.172097}),
	}
	arithmeticTestVars = []*Variable{
		arithmeticTestVec1, arithmeticTestVec2, arithmeticTestVec3, arithmeticTestVec4,
	}
	arithmeticTestRVec = RVector{
		arithmeticTestVec1: linalg.Vector([]float64{
			0.340162, -0.325063, 0.179612, 0.056463, -0.812274,
		}),

		// Vector 2 intentionally left out to see what happens
		// when a vector isn't in the RVector.

		arithmeticTestVec3: linalg.Vector([]float64{
			0.59824, -0.63322, 0.13379, 0.99559, 0.53748,
		}),
		arithmeticTestVec4: linalg.Vector([]float64{
			4.2222, 5.2762, -7.5762, 2.3420, -1.8927,
		}),
	}
)

type arithmeticTestFunc struct{}

func (_ arithmeticTestFunc) Apply(r Result) Result {
	sq1 := Square(Mul(Scale(Add(r, arithmeticTestVec1), -2), arithmeticTestVec2))
	sum1 := AddScaler(Add(Mul(sq1, arithmeticTestVec3), Scale(arithmeticTestVec4, -0.5)), 2)
	powed := Pow(Pow(Inverse(sum1), 2), 1/3.0)
	allSum := SumAll(AddFirst(powed, arithmeticTestVec1))
	return ScaleFirst(AddLogDomain(arithmeticTestVec1, arithmeticTestVec2), allSum)
}

func (_ arithmeticTestFunc) ApplyR(v RVector, r RResult) RResult {
	rVec1 := NewRVariable(arithmeticTestVec1, v)
	rVec2 := NewRVariable(arithmeticTestVec2, v)
	rVec3 := NewRVariable(arithmeticTestVec3, v)
	rVec4 := NewRVariable(arithmeticTestVec4, v)
	sq1 := SquareR(MulR(ScaleR(AddR(r, rVec1), -2), rVec2))
	sum1 := AddScalerR(AddR(MulR(sq1, rVec3), ScaleR(rVec4, -0.5)), 2)
	powed := PowR(PowR(InverseR(sum1), 2), 1/3.0)
	allSum := SumAllR(AddFirstR(powed, rVec1))
	return ScaleFirstR(AddLogDomainR(rVec1, rVec2), allSum)
}

func TestArithmeticGradients(t *testing.T) {
	f := &FuncTest{
		F:     ComposedFunc{arithmeticTestFunc{}, AddTwice{}},
		Vars:  arithmeticTestVars,
		Input: arithmeticTestVec4,
	}
	f.Run(t)
}

func TestArithmeticRGradients(t *testing.T) {
	f := &RFuncTest{
		F:     ComposedRFunc{arithmeticTestFunc{}, AddTwice{}},
		Vars:  arithmeticTestVars,
		Input: arithmeticTestVec4,
		RV:    arithmeticTestRVec,
	}
	f.Run(t)
}

func TestAddLogDomainOutput(t *testing.T) {
	expected := make(linalg.Vector, len(arithmeticTestVec1.Output()))
	for i, a := range arithmeticTestVec1.Output() {
		b := arithmeticTestVec2.Output()[i]
		expected[i] = math.Log(math.Exp(a) + math.Exp(b))
	}

	actual := AddLogDomain(arithmeticTestVec1, arithmeticTestVec2).Output()

	for i, x := range expected {
		a := actual[i]
		if math.Abs(x-a) > 1e-5 {
			t.Errorf("entry %d: should be %f but got %f", i, x, a)
		}
	}
}
