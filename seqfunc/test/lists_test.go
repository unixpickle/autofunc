package seqfunctest

import (
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/autofunc/seqfunc"
)

type SliceListTestFunc struct{}

func (_ SliceListTestFunc) ApplySeqs(in seqfunc.Result) seqfunc.Result {
	scaled := seqfunc.Map(in, func(x autofunc.Result) autofunc.Result {
		return autofunc.Scale(x, -2.5)
	})
	regularSlice := seqfunc.SliceList(scaled, 1, 3)
	optimizedSlice := seqfunc.Pool(in, func(pooled seqfunc.Result) seqfunc.Result {
		return seqfunc.SliceList(in, 1, 3)
	})
	return seqfunc.MapN(func(in ...autofunc.Result) autofunc.Result {
		return autofunc.Add(in[0], in[1])
	}, regularSlice, optimizedSlice)
}

func (_ SliceListTestFunc) ApplySeqsR(rv autofunc.RVector, in seqfunc.RResult) seqfunc.RResult {
	scaled := seqfunc.MapR(in, func(x autofunc.RResult) autofunc.RResult {
		return autofunc.ScaleR(x, -2.5)
	})
	regularSlice := seqfunc.SliceListR(scaled, 1, 3)
	optimizedSlice := seqfunc.PoolR(in, func(pooled seqfunc.RResult) seqfunc.RResult {
		return seqfunc.SliceListR(in, 1, 3)
	})
	return seqfunc.MapNR(func(in ...autofunc.RResult) autofunc.RResult {
		return autofunc.AddR(in[0], in[1])
	}, regularSlice, optimizedSlice)
}

func TestSliceLists(t *testing.T) {
	sc := &functest.SeqRFuncChecker{
		F:     &SliceListTestFunc{},
		Vars:  TestVars,
		Input: TestSeqs,
		RV:    TestRV,
	}
	sc.FullCheck(t)
}
