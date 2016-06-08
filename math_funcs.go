package autofunc

import (
	"math"

	"github.com/unixpickle/num-analysis/linalg"
)

// Exp is a Func and RFunc which computes e^x
// for each component x in the input vector.
type Exp struct{}

func (_ Exp) Apply(in Result) Result {
	input := in.Output()
	output := make(linalg.Vector, len(input))
	for i, x := range input {
		output[i] = math.Exp(x)
	}
	return &ExpResult{
		OutputVec: output,
		Input:     in,
	}
}

func (_ Exp) ApplyR(v RVector, in RResult) RResult {
	input := in.Output()
	inputR := in.ROutput()
	output := make(linalg.Vector, len(input))
	outputR := make(linalg.Vector, len(input))
	for i, x := range input {
		exp := math.Exp(x)
		output[i] = exp
		outputR[i] = exp * inputR[i]
	}
	return &ExpRResult{
		OutputVec:  output,
		ROutputVec: outputR,
		Input:      in,
	}
}

type ExpResult struct {
	OutputVec linalg.Vector
	Input     Result
}

func (e *ExpResult) Output() linalg.Vector {
	return e.OutputVec
}

func (e *ExpResult) Constant(g Gradient) bool {
	return e.Input.Constant(g)
}

func (e *ExpResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if !e.Input.Constant(grad) {
		for i, x := range e.OutputVec {
			upstream[i] *= x
		}
		e.Input.PropagateGradient(upstream, grad)
	}
}

type ExpRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      RResult
}

func (e *ExpRResult) Output() linalg.Vector {
	return e.OutputVec
}

func (e *ExpRResult) ROutput() linalg.Vector {
	return e.ROutputVec
}

func (e *ExpRResult) Constant(rg RGradient, g Gradient) bool {
	return e.Input.Constant(rg, g)
}

func (e *ExpRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	if !e.Input.Constant(rgrad, grad) {
		rOut := e.ROutputVec
		out := e.OutputVec
		for i, u := range upstream {
			uR := upstreamR[i]
			x := out[i]
			xR := rOut[i]
			upstream[i] = u * x
			upstreamR[i] = uR*x + u*xR
		}
		e.Input.PropagateRGradient(upstream, upstreamR, rgrad, grad)
	}
}

// Log is a Func and RFunc which applies the
// natural logarithm component-wise.
type Log struct{}

func (_ Log) Apply(in Result) Result {
	inVec := in.Output()
	outVec := make(linalg.Vector, len(inVec))
	for i, in := range inVec {
		outVec[i] = math.Log(in)
	}
	return &LogResult{
		OutputVec: outVec,
		Input:     in,
	}
}

func (_ Log) ApplyR(v RVector, in RResult) RResult {
	inVec := in.Output()
	inVecR := in.ROutput()
	outVec := make(linalg.Vector, len(inVec))
	outVecR := make(linalg.Vector, len(inVec))
	for i, in := range inVec {
		outVec[i] = math.Log(in)
		outVecR[i] = inVecR[i] / in
	}
	return &LogRResult{
		OutputVec:  outVec,
		ROutputVec: outVecR,
		Input:      in,
	}
}

type LogResult struct {
	OutputVec linalg.Vector
	Input     Result
}

func (l *LogResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *LogResult) Constant(g Gradient) bool {
	return l.Input.Constant(g)
}

func (l *LogResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if l.Input.Constant(grad) {
		return
	}
	inVec := l.Input.Output()
	for i := range upstream {
		upstream[i] *= 1 / inVec[i]
	}
	l.Input.PropagateGradient(upstream, grad)
}

type LogRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      RResult
}

func (l *LogRResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *LogRResult) ROutput() linalg.Vector {
	return l.ROutputVec
}

func (l *LogRResult) Constant(rg RGradient, g Gradient) bool {
	return l.Input.Constant(rg, g)
}

func (l *LogRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	if l.Input.Constant(rgrad, grad) {
		return
	}
	inVec := l.Input.Output()
	inVecR := l.Input.ROutput()
	for i, u := range upstream {
		uR := upstreamR[i]
		input := inVec[i]
		upstream[i] = u / input
		upstreamR[i] = uR/input - u*inVecR[i]/(input*input)
	}
	l.Input.PropagateRGradient(upstream, upstreamR, rgrad, grad)
}

// SquaredNorm is a Func and RFunc which computes
// the squared Euclidean norm of its input.
type SquaredNorm struct{}

func (_ SquaredNorm) Apply(r Result) Result {
	return SumAll(Square(r))
}

func (_ SquaredNorm) ApplyR(v RVector, r RResult) RResult {
	return SumAllR(SquareR(r))
}

// Sigmoid is a Func and RFunc which applies
// the logistic sigmoid function 1/(1+exp(-x))
// to every component in its input vector.
type Sigmoid struct{}

func (s Sigmoid) Apply(in Result) Result {
	inVec := in.Output()
	res := make(linalg.Vector, len(inVec))
	for i, x := range inVec {
		res[i] = 1 / (1 + math.Exp(-x))
	}
	return &SigmoidResult{
		OutputVec: res,
		Input:     in,
	}
}

func (s Sigmoid) ApplyR(v RVector, in RResult) RResult {
	inVec := in.Output()
	inVecR := in.ROutput()
	res := make(linalg.Vector, len(inVec))
	resR := make(linalg.Vector, len(inVec))
	for i, x := range inVec {
		sigVal := 1 / (1 + math.Exp(-x))
		res[i] = sigVal
		resR[i] = sigVal * (1 - sigVal) * inVecR[i]
	}
	return &SigmoidRResult{
		OutputVec:  res,
		ROutputVec: resR,
		Input:      in,
	}
}

type SigmoidResult struct {
	OutputVec linalg.Vector
	Input     Result
}

func (s *SigmoidResult) Output() linalg.Vector {
	return s.OutputVec
}

func (s *SigmoidResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if !s.Input.Constant(grad) {
		inGrad := make(linalg.Vector, len(s.OutputVec))
		for i, x := range s.OutputVec {
			inGrad[i] = x * (1 - x) * upstream[i]
		}
		s.Input.PropagateGradient(inGrad, grad)
	}
}

func (s *SigmoidResult) Constant(g Gradient) bool {
	return s.Input.Constant(g)
}

type SigmoidRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      RResult
}

func (s *SigmoidRResult) Output() linalg.Vector {
	return s.OutputVec
}

func (s *SigmoidRResult) ROutput() linalg.Vector {
	return s.ROutputVec
}

func (s *SigmoidRResult) Constant(rg RGradient, g Gradient) bool {
	return s.Input.Constant(rg, g)
}

func (s *SigmoidRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	if s.Input.Constant(rgrad, grad) {
		return
	}

	inGrad := make(linalg.Vector, len(s.OutputVec))
	inGradR := make(linalg.Vector, len(s.OutputVec))
	outR := s.ROutputVec

	for i, x := range s.OutputVec {
		partial := upstream[i]
		partialR := upstreamR[i]
		xR := outR[i]
		inGrad[i] = x * (1 - x) * partial
		inGradR[i] = xR*(1-x)*partial - x*xR*partial + x*(1-x)*partialR
	}

	s.Input.PropagateRGradient(inGrad, inGradR, rgrad, grad)
}

// Softmax is a Func and RFunc which evaluates the
// softmax function with a given temperature.
type Softmax struct {
	// Temperature is used to divide the input values
	// before they are exponentiated.
	// If the temperature is 0, then a temperature of 1
	// is used like in the standard softmax function.
	Temperature float64
}

func (s *Softmax) Apply(in Result) Result {
	scaledInputs := in
	if s.Temperature != 0 && s.Temperature != 1 {
		scaledInputs = Scale(in, 1/s.Temperature)
	}
	exps := Exp{}.Apply(scaledInputs)
	sum := SumAll(exps)
	return ScaleFirst(exps, Inverse(sum))
}

func (s *Softmax) ApplyR(v RVector, in RResult) RResult {
	scaledInputs := in
	if s.Temperature != 0 && s.Temperature != 1 {
		scaledInputs = ScaleR(in, 1/s.Temperature)
	}
	exps := Exp{}.ApplyR(v, scaledInputs)
	sum := SumAllR(exps)
	return ScaleFirstR(exps, InverseR(sum))
}
