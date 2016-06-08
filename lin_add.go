package autofunc

import "github.com/unixpickle/num-analysis/linalg"

// LinAdd is a Func and RFunc which adds
// a vector to its input.
type LinAdd struct {
	Var *Variable
}

// Apply applies the addition operation to
// the input.
func (l LinAdd) Apply(in Result) Result {
	return &linAddResult{
		OutputVec: in.Output().Copy().Add(l.Var.Vector),
		SumVar:    l.Var,
		Input:     in,
	}
}

// ApplyR is like Apply but for RResults.
func (l LinAdd) ApplyR(v RVector, in RResult) RResult {
	rVar := NewRVariable(l.Var, v)
	return &linAddRResult{
		OutputVec:  in.Output().Copy().Add(l.Var.Vector),
		ROutputVec: in.ROutput().Copy().Add(rVar.ROutput()),
		SumVar:     rVar,
		Input:      in,
	}
}

type linAddResult struct {
	OutputVec linalg.Vector
	SumVar    *Variable
	Input     Result
}

func (l *linAddResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *linAddResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if sumGrad, ok := grad[l.SumVar]; ok {
		sumGrad.Add(upstream)
	}
	if !l.Input.Constant(grad) {
		l.Input.PropagateGradient(upstream, grad)
	}
}

func (l *linAddResult) Constant(g Gradient) bool {
	return l.Input.Constant(g) && l.SumVar.Constant(g)
}

type linAddRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	SumVar     *RVariable
	Input      RResult
}

func (l *linAddRResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *linAddRResult) ROutput() linalg.Vector {
	return l.ROutputVec
}

func (l *linAddRResult) Constant(rg RGradient, g Gradient) bool {
	return l.SumVar.Constant(rg, g) && l.Input.Constant(rg, g)
}

func (l *linAddRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	if grad != nil {
		if sumGrad, ok := grad[l.SumVar.Variable]; ok {
			sumGrad.Add(upstream)
		}
	}

	if sumGrad, ok := rgrad[l.SumVar.Variable]; ok {
		sumGrad.Add(upstreamR)
	}

	if !l.Input.Constant(rgrad, grad) {
		l.Input.PropagateRGradient(upstream, upstreamR, rgrad, grad)
	}
}
