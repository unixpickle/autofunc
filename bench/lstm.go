package bench

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

const lstmSeed = 123

var DefaultLSTMBenchmark = &LSTMBenchmark{
	InputSize:  30,
	HiddenSize: 100,
	OutputSize: 10,
	TimeSteps:  50,
}

// LSTMBenchmark tests how quickly autofunc can perform
// operations on a single-layer LSTM RNN.
//
// See https://en.wikipedia.org/wiki/Long_short-term_memory
// for more info on LSTM.
type LSTMBenchmark struct {
	InputSize  int
	HiddenSize int
	OutputSize int
	TimeSteps  int
}

func (l *LSTMBenchmark) Run(b *testing.B, backProp bool) {
	rand.Seed(lstmSeed)
	net := l.generateNet()
	inputs := l.generateInputs()
	outputGrads := l.generateOutputs()

	b.ResetTimer()
	gradVal := net.AllocGradient()
	for i := 0; i < b.N; i++ {
		net.Reset()
		for _, in := range inputs {
			net.StepTime(in)
		}
		if backProp {
			net.PropagateGradient(outputGrads, gradVal)
		}
	}
}

func (l *LSTMBenchmark) generateInputs() []linalg.Vector {
	ins := make([]linalg.Vector, l.TimeSteps)
	for i := range ins {
		ins[i] = make(linalg.Vector, l.InputSize)
		for j := range ins[i] {
			ins[i][j] = rand.Float64()
		}
	}
	return ins
}

func (l *LSTMBenchmark) generateOutputs() []linalg.Vector {
	outs := make([]linalg.Vector, l.TimeSteps)
	for i := range outs {
		outs[i] = make(linalg.Vector, l.OutputSize)
		for j := range outs[i] {
			outs[i][j] = rand.Float64()
		}
	}
	return outs
}

func (l *LSTMBenchmark) generateNet() *lstmNet {
	inputWeights, inputBiases := l.generateGate()
	inGate, inGateBiases := l.generateGate()
	forgetGate, forgetGateBiases := l.generateGate()
	outputGate, outputGateBiases := l.generateGate()

	outputWeights, outputBiases := l.generateLayer(l.OutputSize)

	return &lstmNet{
		LSTM: &lstmBlock{
			InWeights:         inputWeights,
			InGateWeights:     inGate,
			ForgetGateWeights: forgetGate,
			InBiases:          inputBiases,
			InGateBiases:      inGateBiases,
			ForgetGateBiases:  forgetGateBiases,
		},

		OutputGate:    outputGate,
		OutputWeights: outputWeights,

		OutputGateBiases: outputGateBiases,
		OutputBiases:     outputBiases,

		StateSize: l.HiddenSize,
	}
}

func (l *LSTMBenchmark) generateGate() (*autofunc.LinTran, *autofunc.LinAdd) {
	return l.generateLayer(l.HiddenSize)
}

func (l *LSTMBenchmark) generateLayer(outSize int) (*autofunc.LinTran, *autofunc.LinAdd) {
	weightMat := make(linalg.Vector, (l.InputSize+l.HiddenSize)*outSize)
	for i := range weightMat {
		weightMat[i] = rand.Float64()*2 - 1
	}

	biasMat := make(linalg.Vector, outSize)
	for i := range biasMat {
		biasMat[i] = rand.Float64()*2 - 1
	}

	weightVar := &autofunc.Variable{Vector: weightMat}
	biasVar := &autofunc.Variable{Vector: biasMat}

	linTran := &autofunc.LinTran{
		Data: weightVar,
		Rows: outSize,
		Cols: l.InputSize + l.HiddenSize,
	}
	linAdd := &autofunc.LinAdd{Var: biasVar}

	return linTran, linAdd
}

type lstmBlock struct {
	InWeights         *autofunc.LinTran
	InGateWeights     *autofunc.LinTran
	ForgetGateWeights *autofunc.LinTran

	InBiases         *autofunc.LinAdd
	InGateBiases     *autofunc.LinAdd
	ForgetGateBiases *autofunc.LinAdd
}

func (l *lstmBlock) Apply(state, input autofunc.Result) autofunc.Result {
	in := autofunc.Concat(state, input)

	s := autofunc.Sigmoid{}
	inState := s.Apply(l.InBiases.Apply(l.InWeights.Apply(in)))
	inMask := s.Apply(l.InGateBiases.Apply(l.InGateWeights.Apply(in)))
	forgetMask := s.Apply(l.ForgetGateBiases.Apply(l.ForgetGateWeights.Apply(in)))

	maskedNew := autofunc.Mul(inMask, inState)
	maskedOld := autofunc.Mul(forgetMask, state)

	return autofunc.Add(maskedOld, maskedNew)
}

type lstmNet struct {
	LSTM      *lstmBlock
	StateSize int

	OutputGate    *autofunc.LinTran
	OutputWeights *autofunc.LinTran

	OutputGateBiases *autofunc.LinAdd
	OutputBiases     *autofunc.LinAdd

	inputStates     []*autofunc.Variable
	outputStates    []autofunc.Result
	outputStateVars []*autofunc.Variable
	outputs         []autofunc.Result
}

func (l *lstmNet) StepTime(sample linalg.Vector) linalg.Vector {
	var inState *autofunc.Variable
	if len(l.outputStates) > 0 {
		outVec := l.outputStates[len(l.outputStates)-1].Output()
		inState = &autofunc.Variable{Vector: outVec}
	} else {
		inState = &autofunc.Variable{Vector: make(linalg.Vector, l.StateSize)}
	}

	sampleVar := &autofunc.Variable{Vector: sample}
	outState := l.LSTM.Apply(inState, sampleVar)

	l.inputStates = append(l.inputStates, inState)
	l.outputStates = append(l.outputStates, outState)

	outStateVar := &autofunc.Variable{Vector: outState.Output()}
	l.outputStateVars = append(l.outputStateVars, outStateVar)

	s := autofunc.Sigmoid{}
	joinedInput := autofunc.Concat(inState, sampleVar)
	outputGate := s.Apply(l.OutputGateBiases.Apply(l.OutputGate.Apply(joinedInput)))

	maskedState := autofunc.Mul(outStateVar, outputGate)

	augOut := autofunc.Concat(maskedState, sampleVar)
	result := s.Apply(l.OutputBiases.Apply(l.OutputWeights.Apply(augOut)))
	l.outputs = append(l.outputs, result)

	return result.Output()
}

func (l *lstmNet) AllocGradient() autofunc.Gradient {
	return autofunc.NewGradient([]*autofunc.Variable{
		l.OutputBiases.Var,
		l.OutputGateBiases.Var,
		l.OutputWeights.Data,
		l.OutputGate.Data,
		l.LSTM.ForgetGateBiases.Var,
		l.LSTM.ForgetGateWeights.Data,
		l.LSTM.InGateBiases.Var,
		l.LSTM.InGateWeights.Data,
		l.LSTM.InBiases.Var,
		l.LSTM.InWeights.Data,
	})
}

func (l *lstmNet) PropagateGradient(upstreams []linalg.Vector, grad autofunc.Gradient) {
	stateGrad := make(linalg.Vector, l.StateSize)
	for i := len(upstreams) - 1; i >= 0; i-- {
		upstream := upstreams[i]
		output := l.outputs[i]
		outStateVar := l.outputStateVars[i]
		lastStateVar := l.inputStates[i]

		grad[outStateVar] = make(linalg.Vector, len(outStateVar.Vector))
		grad[lastStateVar] = make(linalg.Vector, len(lastStateVar.Vector))

		output.PropagateGradient(upstream, grad)
		stateGrad.Add(grad[outStateVar])
		delete(grad, outStateVar)

		l.outputStates[i].PropagateGradient(stateGrad, grad)
		stateGrad = grad[lastStateVar]
		delete(grad, lastStateVar)
	}
}

func (l *lstmNet) Reset() {
	for _, l := range l.outputStates {
		l.Release()
	}
	for _, l := range l.outputs {
		l.Release()
	}
	l.inputStates = nil
	l.outputStates = nil
	l.outputStateVars = nil
	l.outputs = nil
}
