package autofunc

import (
	"math"

	"github.com/unixpickle/num-analysis/linalg"
)

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
