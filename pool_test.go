package autofunc

import "testing"

var poolTestVar = &Variable{[]float64{1, -2, 3, -4, 5}}
var poolTestRVec = RVector{
	poolTestVar: []float64{1, -1, 2, -2, 3},
}

type poolTestFunc struct{}

func (_ poolTestFunc) Apply(r Result) Result {
	return Pool(r, func(pooled Result) Result {
		return AddTwice{}.Apply(Pow(Exp{}.Apply(AddScaler(pooled, 2)), 0.5))
	})
}

func (_ poolTestFunc) ApplyR(v RVector, r RResult) RResult {
	return PoolR(r, func(pooled RResult) RResult {
		return AddTwice{}.ApplyR(v, PowR(Exp{}.ApplyR(v, AddScalerR(pooled, 2)), 0.5))
	})
}

func TestPool(t *testing.T) {
	testFunc := ComposedFunc{Sigmoid{}, AddTwice{}, Exp{}, poolTestFunc{}}
	f := FuncTest{
		F:     testFunc,
		Vars:  []*Variable{poolTestVar},
		Input: poolTestVar,
	}
	f.Run(t)
}

func TestPoolR(t *testing.T) {
	testFunc := ComposedRFunc{Sigmoid{}, AddTwice{}, Exp{}, poolTestFunc{}}
	f := RFuncTest{
		F:     testFunc,
		Vars:  []*Variable{poolTestVar},
		Input: NewRVariable(poolTestVar, poolTestRVec),
		RV:    poolTestRVec,
	}
	f.Run(t)
}
