package bench

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

const linTranSeed = 123

var DefaultLinTranBenchmark = &LinTranBenchmark{
	Rows:      1024,
	Cols:      2048,
	BatchSize: 10,
}

type LinTranBenchmark struct {
	Rows      int
	Cols      int
	BatchSize int
}

func (l *LinTranBenchmark) Run(b *testing.B, backProp bool) {
	rand.Seed(linTranSeed)

	matrix := &autofunc.LinTran{
		Data: &autofunc.Variable{Vector: make(linalg.Vector, l.Rows*l.Cols)},
		Rows: l.Rows,
		Cols: l.Cols,
	}
	inputVector := make(linalg.Vector, l.Cols*l.BatchSize)
	for i := range inputVector {
		inputVector[i] = rand.Float64()*2 - 1
	}
	inputVar := &autofunc.Variable{Vector: inputVector}
	upstream := make(linalg.Vector, l.Rows*l.BatchSize)
	for i := range upstream {
		upstream[i] = rand.Float64()*2 - 1
	}
	grad := autofunc.NewGradient([]*autofunc.Variable{matrix.Data, inputVar})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := matrix.Batch(inputVar, l.BatchSize)
		if backProp {
			res.PropagateGradient(upstream, grad)
		}
		res.Release()
	}
}

func (l *LinTranBenchmark) RunR(b *testing.B, backProp bool) {
	rand.Seed(linTranSeed)

	matrix := &autofunc.LinTran{
		Data: &autofunc.Variable{Vector: make(linalg.Vector, l.Rows*l.Cols)},
		Rows: l.Rows,
		Cols: l.Cols,
	}
	inputVector := make(linalg.Vector, l.Cols*l.BatchSize)
	for i := range inputVector {
		inputVector[i] = rand.Float64()*2 - 1
	}
	inputVar := &autofunc.Variable{Vector: inputVector}
	upstream := make(linalg.Vector, l.Rows*l.BatchSize)
	upstreamR := make(linalg.Vector, len(upstream))
	for i := range upstream {
		upstream[i] = rand.Float64()*2 - 1
		upstreamR[i] = rand.Float64()*2 - 1
	}
	grad := autofunc.NewGradient([]*autofunc.Variable{matrix.Data, inputVar})
	rgrad := autofunc.NewRGradient([]*autofunc.Variable{matrix.Data, inputVar})

	rVec := autofunc.RVector{}
	for variable := range grad {
		rVec[variable] = make(linalg.Vector, len(variable.Vector))
		for i := range rVec[variable] {
			rVec[variable][i] = rand.Float64()*2 - 1
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rVar := autofunc.NewRVariable(inputVar, rVec)
		res := matrix.BatchR(rVec, rVar, l.BatchSize)
		if backProp {
			res.PropagateRGradient(upstream, upstreamR, rgrad, grad)
		}
		res.Release()
	}
}
