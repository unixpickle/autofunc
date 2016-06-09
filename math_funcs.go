package autofunc

import (
	"math"

	"github.com/unixpickle/num-analysis/linalg"
)

// Exp is a Func and RFunc which computes e^x
// for each component x in the input vector.
type Exp struct {
	Cache *VectorCache
}

func (e Exp) Apply(in Result) Result {
	input := in.Output()
	output := e.Cache.Alloc(len(input))
	for i, x := range input {
		output[i] = math.Exp(x)
	}
	return &expResult{
		Cache:     e.Cache,
		OutputVec: output,
		Input:     in,
	}
}

func (e Exp) ApplyR(v RVector, in RResult) RResult {
	input := in.Output()
	inputR := in.ROutput()
	output := e.Cache.Alloc(len(input))
	outputR := e.Cache.Alloc(len(input))
	for i, x := range input {
		exp := math.Exp(x)
		output[i] = exp
		outputR[i] = exp * inputR[i]
	}
	return &expRResult{
		Cache:      e.Cache,
		OutputVec:  output,
		ROutputVec: outputR,
		Input:      in,
	}
}

type expResult struct {
	Cache     *VectorCache
	OutputVec linalg.Vector
	Input     Result
}

func (e *expResult) Output() linalg.Vector {
	return e.OutputVec
}

func (e *expResult) Constant(g Gradient) bool {
	return e.Input.Constant(g)
}

func (e *expResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if !e.Input.Constant(grad) {
		for i, x := range e.OutputVec {
			upstream[i] *= x
		}
		e.Input.PropagateGradient(upstream, grad)
	}
}

func (e *expResult) Release() {
	e.Cache.Free(e.OutputVec)
	e.OutputVec = nil
	e.Input.Release()
}

type expRResult struct {
	Cache      *VectorCache
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      RResult
}

func (e *expRResult) Output() linalg.Vector {
	return e.OutputVec
}

func (e *expRResult) ROutput() linalg.Vector {
	return e.ROutputVec
}

func (e *expRResult) Constant(rg RGradient, g Gradient) bool {
	return e.Input.Constant(rg, g)
}

func (e *expRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
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

func (e *expRResult) Release() {
	e.Cache.Free(e.OutputVec)
	e.Cache.Free(e.ROutputVec)
	e.OutputVec = nil
	e.ROutputVec = nil
	e.Input.Release()
}

// Log is a Func and RFunc which applies the
// natural logarithm component-wise.
type Log struct {
	Cache *VectorCache
}

func (l Log) Apply(in Result) Result {
	inVec := in.Output()
	outVec := l.Cache.Alloc(len(inVec))
	for i, in := range inVec {
		outVec[i] = math.Log(in)
	}
	return &logResult{
		Cache:     l.Cache,
		OutputVec: outVec,
		Input:     in,
	}
}

func (l Log) ApplyR(v RVector, in RResult) RResult {
	inVec := in.Output()
	inVecR := in.ROutput()
	outVec := l.Cache.Alloc(len(inVec))
	outVecR := l.Cache.Alloc(len(inVec))
	for i, in := range inVec {
		outVec[i] = math.Log(in)
		outVecR[i] = inVecR[i] / in
	}
	return &logRResult{
		Cache:      l.Cache,
		OutputVec:  outVec,
		ROutputVec: outVecR,
		Input:      in,
	}
}

type logResult struct {
	Cache     *VectorCache
	OutputVec linalg.Vector
	Input     Result
}

func (l *logResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *logResult) Constant(g Gradient) bool {
	return l.Input.Constant(g)
}

func (l *logResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if l.Input.Constant(grad) {
		return
	}
	inVec := l.Input.Output()
	for i := range upstream {
		upstream[i] *= 1 / inVec[i]
	}
	l.Input.PropagateGradient(upstream, grad)
}

func (l *logResult) Release() {
	l.Cache.Free(l.OutputVec)
	l.OutputVec = nil
	l.Input.Release()
}

type logRResult struct {
	Cache      *VectorCache
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      RResult
}

func (l *logRResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *logRResult) ROutput() linalg.Vector {
	return l.ROutputVec
}

func (l *logRResult) Constant(rg RGradient, g Gradient) bool {
	return l.Input.Constant(rg, g)
}

func (l *logRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
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

func (l *logRResult) Release() {
	l.Cache.Free(l.OutputVec)
	l.Cache.Free(l.ROutputVec)
	l.OutputVec = nil
	l.ROutputVec = nil
	l.Input.Release()
}

// SquaredNorm is a Func and RFunc which computes
// the squared Euclidean norm of its input.
type SquaredNorm struct {
	Cache *VectorCache
}

func (s SquaredNorm) Apply(r Result) Result {
	arith := &Arithmetic{s.Cache}
	return arith.SumAll(arith.Square(r))
}

func (s SquaredNorm) ApplyR(v RVector, r RResult) RResult {
	arith := &Arithmetic{s.Cache}
	return arith.SumAllR(arith.SquareR(r))
}

// Sigmoid is a Func and RFunc which applies
// the logistic sigmoid function 1/(1+exp(-x))
// to every component in its input vector.
type Sigmoid struct {
	Cache *VectorCache
}

func (s Sigmoid) Apply(in Result) Result {
	inVec := in.Output()
	res := s.Cache.Alloc(len(inVec))
	for i, x := range inVec {
		res[i] = 1 / (1 + math.Exp(-x))
	}
	return &sigmoidResult{
		Cache:     s.Cache,
		OutputVec: res,
		Input:     in,
	}
}

func (s Sigmoid) ApplyR(v RVector, in RResult) RResult {
	inVec := in.Output()
	inVecR := in.ROutput()
	res := s.Cache.Alloc(len(inVec))
	resR := s.Cache.Alloc(len(inVec))
	for i, x := range inVec {
		sigVal := 1 / (1 + math.Exp(-x))
		res[i] = sigVal
		resR[i] = sigVal * (1 - sigVal) * inVecR[i]
	}
	return &sigmoidRResult{
		Cache:      s.Cache,
		OutputVec:  res,
		ROutputVec: resR,
		Input:      in,
	}
}

type sigmoidResult struct {
	Cache     *VectorCache
	OutputVec linalg.Vector
	Input     Result
}

func (s *sigmoidResult) Output() linalg.Vector {
	return s.OutputVec
}

func (s *sigmoidResult) Constant(g Gradient) bool {
	return s.Input.Constant(g)
}

func (s *sigmoidResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if !s.Input.Constant(grad) {
		for i, x := range s.OutputVec {
			upstream[i] *= x * (1 - x)
		}
		s.Input.PropagateGradient(upstream, grad)
	}
}

func (s *sigmoidResult) Release() {
	s.Cache.Free(s.OutputVec)
	s.OutputVec = nil
	s.Input.Release()
}

type sigmoidRResult struct {
	Cache      *VectorCache
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      RResult
}

func (s *sigmoidRResult) Output() linalg.Vector {
	return s.OutputVec
}

func (s *sigmoidRResult) ROutput() linalg.Vector {
	return s.ROutputVec
}

func (s *sigmoidRResult) Constant(rg RGradient, g Gradient) bool {
	return s.Input.Constant(rg, g)
}

func (s *sigmoidRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	if s.Input.Constant(rgrad, grad) {
		return
	}

	outR := s.ROutputVec

	for i, x := range s.OutputVec {
		partial := upstream[i]
		partialR := upstreamR[i]
		xR := outR[i]
		upstream[i] = x * (1 - x) * partial
		upstreamR[i] = xR*(1-x)*partial - x*xR*partial + x*(1-x)*partialR
	}

	s.Input.PropagateRGradient(upstream, upstreamR, rgrad, grad)
}

func (s *sigmoidRResult) Release() {
	s.Cache.Free(s.OutputVec)
	s.Cache.Free(s.ROutputVec)
	s.OutputVec = nil
	s.ROutputVec = nil
	s.Input.Release()
}

// Softmax is a Func and RFunc which evaluates the
// softmax function with a given temperature.
type Softmax struct {
	// Temperature is used to divide the input values
	// before they are exponentiated.
	// If the temperature is 0, then a temperature of 1
	// is used like in the standard softmax function.
	Temperature float64

	Cache *VectorCache
}

func (s *Softmax) Apply(in Result) Result {
	arith := Arithmetic{s.Cache}
	scaledInputs := in
	if s.Temperature != 0 && s.Temperature != 1 {
		scaledInputs = arith.Scale(in, 1/s.Temperature)
	}
	exps := Exp{s.Cache}.Apply(scaledInputs)
	sum := arith.SumAll(exps)
	return arith.ScaleFirst(exps, arith.Inverse(sum))
}

func (s *Softmax) ApplyR(v RVector, in RResult) RResult {
	arith := Arithmetic{s.Cache}
	scaledInputs := in
	if s.Temperature != 0 && s.Temperature != 1 {
		scaledInputs = arith.ScaleR(in, 1/s.Temperature)
	}
	exps := Exp{s.Cache}.ApplyR(v, scaledInputs)
	sum := arith.SumAllR(exps)
	return arith.ScaleFirstR(exps, arith.InverseR(sum))
}
