package autofunc

import (
	"math"

	"github.com/gonum/blas/blas64"
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
	return &expResult{
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
	return &expRResult{
		OutputVec:  output,
		ROutputVec: outputR,
		Input:      in,
	}
}

type expResult struct {
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

type expRResult struct {
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

// Log is a Func and RFunc which applies the
// natural logarithm component-wise.
type Log struct{}

func (_ Log) Apply(in Result) Result {
	inVec := in.Output()
	outVec := make(linalg.Vector, len(inVec))
	for i, in := range inVec {
		outVec[i] = math.Log(in)
	}
	return &logResult{
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
	return &logRResult{
		OutputVec:  outVec,
		ROutputVec: outVecR,
		Input:      in,
	}
}

type logResult struct {
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

type logRResult struct {
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

// SquaredNorm is a Func and RFunc which computes
// the squared Euclidean norm of its input.
type SquaredNorm struct{}

func (_ SquaredNorm) Apply(r Result) Result {
	return SumAll(Square(r))
}

func (_ SquaredNorm) ApplyR(v RVector, r RResult) RResult {
	return SumAllR(SquareR(r))
}

// Norm is a Func and RFunc which computes the Euclidean
// norm of its input.
type Norm struct{}

func (_ Norm) Apply(r Result) Result {
	v := blas64.Vector{Data: r.Output(), Inc: 1}
	output := blas64.Nrm2(len(v.Data), v)
	return &normResult{
		Input:     r,
		OutputVec: []float64{output},
	}
}

func (_ Norm) ApplyR(rv RVector, r RResult) RResult {
	v := blas64.Vector{Data: r.Output(), Inc: 1}
	output := blas64.Nrm2(len(v.Data), v)
	rout := (1 / output) * r.Output().DotFast(r.ROutput())
	return &normRResult{
		Input:      r,
		OutputVec:  []float64{output},
		ROutputVec: []float64{rout},
	}
}

type normResult struct {
	Input     Result
	OutputVec linalg.Vector
}

func (n *normResult) Output() linalg.Vector {
	return n.OutputVec
}

func (n *normResult) Constant(g Gradient) bool {
	return n.Input.Constant(g)
}

func (n *normResult) PropagateGradient(u linalg.Vector, g Gradient) {
	if n.Constant(g) {
		return
	}
	scale := u[0] / n.Output()[0]
	downstream := make(linalg.Vector, len(n.Input.Output()))
	copy(downstream, n.Input.Output())
	blas64.Scal(len(downstream), scale, blas64.Vector{Data: downstream, Inc: 1})
	n.Input.PropagateGradient(downstream, g)
}

type normRResult struct {
	Input      RResult
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
}

func (n *normRResult) Output() linalg.Vector {
	return n.OutputVec
}

func (n *normRResult) ROutput() linalg.Vector {
	return n.ROutputVec
}

func (n *normRResult) Constant(rg RGradient, g Gradient) bool {
	return n.Input.Constant(rg, g)
}

func (n *normRResult) PropagateRGradient(u, uR linalg.Vector, rg RGradient, g Gradient) {
	if n.Constant(rg, g) {
		return
	}
	scale := u[0] / n.Output()[0]
	scaleR := -u[0]*n.ROutput()[0]/(n.Output()[0]*n.Output()[0]) + uR[0]/n.Output()[0]
	downstream := make(linalg.Vector, len(n.Input.Output()))
	downstreamR := make(linalg.Vector, len(n.Input.Output()))
	copy(downstream, n.Input.Output())
	copy(downstreamR, downstream)
	blas64.Scal(len(downstream), scale, blas64.Vector{Data: downstream, Inc: 1})
	blas64.Scal(len(downstream), scaleR, blas64.Vector{Data: downstreamR, Inc: 1})
	downstreamR.Add(n.Input.ROutput().Copy().Scale(scale))
	n.Input.PropagateRGradient(downstream, downstreamR, rg, g)
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
	return &sigmoidResult{
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
	return &sigmoidRResult{
		OutputVec:  res,
		ROutputVec: resR,
		Input:      in,
	}
}

type sigmoidResult struct {
	OutputVec linalg.Vector
	Input     Result
}

func (s *sigmoidResult) Output() linalg.Vector {
	return s.OutputVec
}

func (s *sigmoidResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if !s.Input.Constant(grad) {
		for i, x := range s.OutputVec {
			upstream[i] *= x * (1 - x)
		}
		s.Input.PropagateGradient(upstream, grad)
	}
}

func (s *sigmoidResult) Constant(g Gradient) bool {
	return s.Input.Constant(g)
}

type sigmoidRResult struct {
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

// LogSigmoid is a Func and RFunc which computes the
// logarithm of the logistic sigmoid function,
// i.e. ln(1/(1+exp(-x))).
// This is more numerically reliable than taking the
// log of the sigmoid in two steps.
type LogSigmoid struct{}

func (l LogSigmoid) Apply(in Result) Result {
	return &logSigmoidResult{
		OutputVec: l.logSigmoid(in.Output()),
		Input:     in,
	}
}

func (l LogSigmoid) ApplyR(v RVector, in RResult) RResult {
	inVec := in.Output()
	return &logSigmoidRResult{
		OutputVec:  l.logSigmoid(inVec),
		ROutputVec: l.logSigmoidR(inVec, in.ROutput()),
		Input:      in,
	}
}

func (l LogSigmoid) logSigmoid(inVec linalg.Vector) linalg.Vector {
	res := make(linalg.Vector, len(inVec))
	for i, x := range inVec {
		// Avoid taking the log of a big number.
		if x > 0 {
			res[i] = -math.Log(1 + math.Exp(-x))
		} else {
			res[i] = x - math.Log(1+math.Exp(x))
		}
	}
	return res
}

func (l LogSigmoid) logSigmoidR(inVec, inVecR linalg.Vector) linalg.Vector {
	res := make(linalg.Vector, len(inVec))
	for i, x := range inVec {
		res[i] = inVecR[i] / (1 + math.Exp(x))
	}
	return res
}

type logSigmoidResult struct {
	OutputVec linalg.Vector
	Input     Result
}

func (l *logSigmoidResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *logSigmoidResult) Constant(g Gradient) bool {
	return l.Input.Constant(g)
}

func (l *logSigmoidResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if !l.Input.Constant(grad) {
		for i, x := range l.Input.Output() {
			upstream[i] /= (1 + math.Exp(x))
		}
		l.Input.PropagateGradient(upstream, grad)
	}
}

type logSigmoidRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      RResult
}

func (l *logSigmoidRResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *logSigmoidRResult) ROutput() linalg.Vector {
	return l.ROutputVec
}

func (l *logSigmoidRResult) Constant(rg RGradient, g Gradient) bool {
	return l.Input.Constant(rg, g)
}

func (l *logSigmoidRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	if !l.Input.Constant(rgrad, grad) {
		rInput := l.Input.ROutput()
		for i, x := range l.Input.Output() {
			xR := rInput[i]
			uR := upstreamR[i]
			u := upstream[i]
			expX := math.Exp(x)
			sigmoid := 1 / (1 + expX)
			upstream[i] = u * sigmoid
			upstreamR[i] = uR*sigmoid - u*sigmoid*sigmoid*expX*xR
		}
		l.Input.PropagateRGradient(upstream, upstreamR, rgrad, grad)
	}
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
	return Pool(exps, func(exps Result) Result {
		sum := SumAll(exps)
		return ScaleFirst(exps, Inverse(sum))
	})
}

func (s *Softmax) ApplyR(v RVector, in RResult) RResult {
	scaledInputs := in
	if s.Temperature != 0 && s.Temperature != 1 {
		scaledInputs = ScaleR(in, 1/s.Temperature)
	}
	exps := Exp{}.ApplyR(v, scaledInputs)
	return PoolR(exps, func(in RResult) RResult {
		sum := SumAllR(exps)
		return ScaleFirstR(exps, InverseR(sum))
	})
}

// Sin is a Func and RFunc which evaluates the sine
// (in radians) of each of its input components.
type Sin struct{}

func (_ Sin) Apply(in Result) Result {
	input := in.Output()
	res := make(linalg.Vector, len(input))
	for i, x := range input {
		res[i] = math.Sin(x)
	}
	return &sinResult{
		OutputVec: res,
		Input:     in,
	}
}

func (_ Sin) ApplyR(v RVector, in RResult) RResult {
	input := in.Output()
	inputR := in.ROutput()
	res := make(linalg.Vector, len(input))
	resR := make(linalg.Vector, len(inputR))
	for i, x := range input {
		res[i] = math.Sin(x)
		resR[i] = math.Cos(x) * inputR[i]
	}
	return &sinRResult{
		OutputVec:  res,
		ROutputVec: resR,
		Input:      in,
	}
}

type sinResult struct {
	OutputVec linalg.Vector
	Input     Result
}

func (s *sinResult) Output() linalg.Vector {
	return s.OutputVec
}

func (s *sinResult) Constant(g Gradient) bool {
	return s.Input.Constant(g)
}

func (s *sinResult) PropagateGradient(upstream linalg.Vector, g Gradient) {
	if !s.Input.Constant(g) {
		for i, input := range s.Input.Output() {
			upstream[i] *= math.Cos(input)
		}
		s.Input.PropagateGradient(upstream, g)
	}
}

type sinRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      RResult
}

func (s *sinRResult) Output() linalg.Vector {
	return s.OutputVec
}

func (s *sinRResult) ROutput() linalg.Vector {
	return s.ROutputVec
}

func (s *sinRResult) Constant(rg RGradient, g Gradient) bool {
	return s.Input.Constant(rg, g)
}

func (s *sinRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rg RGradient, g Gradient) {
	if !s.Input.Constant(rg, g) {
		rIn := s.Input.ROutput()
		for i, input := range s.Input.Output() {
			cosIn := math.Cos(input)
			cosDeriv := -s.OutputVec[i] * rIn[i]
			upstreamR[i] = upstreamR[i]*cosIn + upstream[i]*cosDeriv
			upstream[i] *= cosIn
		}
		s.Input.PropagateRGradient(upstream, upstreamR, rg, g)
	}
}

// Cos is a Func and RFunc which evaluates the cosine
// (in radians) of all its input's components.
type Cos struct{}

func (_ Cos) Apply(in Result) Result {
	return Sin{}.Apply(AddScaler(in, math.Pi/2))
}

func (_ Cos) ApplyR(v RVector, in RResult) RResult {
	return Sin{}.ApplyR(v, AddScalerR(in, math.Pi/2))
}
