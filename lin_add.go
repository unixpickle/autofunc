package autofunc

import "github.com/unixpickle/num-analysis/linalg"

// LinAdd is a Func and RFunc which adds
// a vector to its input.
type LinAdd struct {
	Var *Variable
}

// Apply applies the addition operation to
// the input, returning a *LinAddResult.
func (l LinAdd) Apply(in Result) Result {
	return &LinAddResult{
		OutputVec: in.Output().Copy().Add(l.Var.Vector),
		SumVar:    l.Var,
		Input:     in,
	}
}

// ApplyR is like Apply, but acts on RResults
// and returns *LinAddRResult.
func (l LinAdd) ApplyR(v RVector, in RResult) RResult {
	rVar := NewRVariable(l.Var, v)
	return &LinAddRResult{
		OutputVec:  in.Output().Copy().Add(l.Var.Vector),
		ROutputVec: in.ROutput().Copy().Add(rVar.ROutput()),
		SumVar:     rVar,
		Input:      in,
	}
}

type LinAddResult struct {
	OutputVec linalg.Vector
	SumVar    *Variable
	Input     Result
}

func (l *LinAddResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *LinAddResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if sumGrad, ok := grad[l.SumVar]; ok {
		for i, x := range upstream {
			sumGrad[i] += x
		}
	}
	if !l.Input.Constant(grad) {
		l.Input.PropagateGradient(upstream, grad)
	}
}

func (l *LinAddResult) Constant(g Gradient) bool {
	return l.Input.Constant(g) && l.SumVar.Constant(g)
}

type LinAddRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	SumVar     *RVariable
	Input      RResult
}

func (l *LinAddRResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *LinAddRResult) ROutput() linalg.Vector {
	return l.ROutputVec
}

func (l *LinAddRResult) Constant(rg RGradient, g Gradient) bool {
	return l.SumVar.Constant(rg, g) && l.Input.Constant(rg, g)
}

func (l *LinAddRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	if grad != nil {
		if sumGrad, ok := grad[l.SumVar.Variable]; ok {
			for i, x := range upstream {
				sumGrad[i] += x
			}
		}
	}

	if sumGrad, ok := rgrad[l.SumVar.Variable]; ok {
		for i, x := range upstreamR {
			sumGrad[i] += x
		}
	}

	if !l.Input.Constant(rgrad, grad) {
		l.Input.PropagateRGradient(upstream, upstreamR, rgrad, grad)
	}
}
