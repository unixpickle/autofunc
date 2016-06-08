package bench

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

const (
	mlpFuncSeed = 123
	mlpDataSeed = 124
	mlpRVecSeed = 125
)

var DefaultMLPBenchmark = &MLPBenchmark{
	LayerSizes: []int{1000, 2000, 512, 10},
	Activation: autofunc.Sigmoid{},
}

// MLPBenchmark tests how quickly autofunc can perform
// operations on a multilayer perception network.
type MLPBenchmark struct {
	LayerSizes []int
	Activation autofunc.RFunc
}

func (m *MLPBenchmark) Run(b *testing.B, backProp bool) {
	netFunc, vars := m.makeFunc()
	input := m.makeInput()
	outGrad, _ := m.outputGrads()
	grad := autofunc.NewGradient(vars)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := netFunc.Apply(input)
		if backProp {
			res.PropagateGradient(outGrad, grad)
		}
	}
}

func (m *MLPBenchmark) RunR(b *testing.B, backProp bool) {
	netFunc, vars := m.makeFunc()
	input := m.makeInput()
	rVec := m.makeRVector(vars)
	outGrad, outRGrad := m.outputGrads()
	rgrad := autofunc.NewRGradient(vars)
	grad := autofunc.NewGradient(vars)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := netFunc.ApplyR(rVec, autofunc.NewRVariable(input, rVec))
		if backProp {
			res.PropagateRGradient(outGrad, outRGrad, rgrad, grad)
		}
	}
}

func (m *MLPBenchmark) makeFunc() (autofunc.RFunc, []*autofunc.Variable) {
	rand.Seed(mlpFuncSeed)

	var layers autofunc.ComposedRFunc
	var variables []*autofunc.Variable
	for i, outSize := range m.LayerSizes[1:] {
		inSize := m.LayerSizes[i]
		mat := make(linalg.Vector, inSize*outSize)
		biasVec := make(linalg.Vector, outSize)
		for i := range mat {
			mat[i] = rand.Float64()*2 - 1
		}
		for i := range biasVec {
			biasVec[i] = rand.Float64()*2 - 1
		}
		matVar := &autofunc.Variable{Vector: mat}
		biasVar := &autofunc.Variable{Vector: biasVec}
		weightLayer := &autofunc.LinTran{
			Data:  matVar,
			Rows:  outSize,
			Cols:  inSize,
			Cache: autofunc.NewVectorCache(0),
		}
		biasLayer := &autofunc.LinAdd{
			Var: biasVar,
		}
		layers = append(layers, weightLayer)
		layers = append(layers, biasLayer)
		layers = append(layers, m.Activation)
		variables = append(variables, matVar)
		variables = append(variables, biasVar)
	}

	return layers, variables
}

func (m *MLPBenchmark) makeInput() *autofunc.Variable {
	rand.Seed(mlpDataSeed)

	vec := make(linalg.Vector, m.LayerSizes[0])
	for i := range vec {
		vec[i] = rand.Float64()*2 - 1
	}

	return &autofunc.Variable{Vector: vec}
}

func (m *MLPBenchmark) makeRVector(vars []*autofunc.Variable) autofunc.RVector {
	rand.Seed(mlpRVecSeed)
	res := autofunc.RVector{}
	for _, v := range vars {
		vec := make(linalg.Vector, len(v.Vector))
		for i := range vec {
			vec[i] = rand.Float64()*2 - 1
		}
		res[v] = vec
	}
	return res
}

func (m *MLPBenchmark) outputGrads() (grad, rGrad linalg.Vector) {
	outSize := m.LayerSizes[len(m.LayerSizes)-1]
	grad = make(linalg.Vector, outSize)
	for i := range grad {
		grad[i] = 1
	}
	rGrad = make(linalg.Vector, outSize)
	return
}
