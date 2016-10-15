package autofunc

import (
	"testing"

	. "github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
)

var poolTestVar = &Variable{[]float64{1, -2, 3, -4, 5}}
var poolTestRVec = RVector{
	poolTestVar: []float64{1, -1, 2, -2, 3},
}

type poolTestFunc struct{}

func (_ poolTestFunc) Apply(r Result) Result {
	return Pool(r, func(pooled Result) Result {
		return Pow(Exp{}.Apply(AddScaler(pooled, 2)), 0.5)
	})
}

func (_ poolTestFunc) ApplyR(v RVector, r RResult) RResult {
	return PoolR(r, func(pooled RResult) RResult {
		return PowR(Exp{}.ApplyR(v, AddScalerR(pooled, 2)), 0.5)
	})
}

func TestPool(t *testing.T) {
	testFunc := ComposedFunc{Sigmoid{}, Exp{}, poolTestFunc{}}
	f := functest.FuncChecker{
		F:     testFunc,
		Vars:  []*Variable{poolTestVar},
		Input: poolTestVar,
	}
	f.FullCheck(t)
}

func TestPoolR(t *testing.T) {
	testFunc := ComposedRFunc{Sigmoid{}, Exp{}, poolTestFunc{}}
	f := functest.RFuncChecker{
		F:     testFunc,
		Vars:  []*Variable{poolTestVar},
		Input: poolTestVar,
		RV:    poolTestRVec,
	}
	f.FullCheck(t)
}
