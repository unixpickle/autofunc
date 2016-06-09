package autofunc

import "github.com/unixpickle/num-analysis/linalg"

// LinAdd is a Func and RFunc which adds
// a vector to its input.
type LinAdd struct {
	Var *Variable

	Cache *VectorCache
}

// Apply applies the addition operation to
// the input.
func (l LinAdd) Apply(in Result) Result {
	outVec := l.Cache.Alloc(len(l.Var.Vector))
	for i, x := range in.Output() {
		outVec[i] = x + l.Var.Vector[i]
	}
	return &linAddResult{
		Cache:     l.Cache,
		OutputVec: outVec,
		SumVar:    l.Var,
		Input:     in,
	}
}

// ApplyR is like Apply but for RResults.
func (l LinAdd) ApplyR(v RVector, in RResult) RResult {
	rVar := NewRVariableCache(l.Var, v, l.Cache)

	value1 := rVar.Output()
	value2 := in.Output()
	value1R := rVar.ROutput()
	value2R := in.ROutput()

	sum := l.Cache.Alloc(len(value1))
	sumR := l.Cache.Alloc(len(value1))

	for i, x := range value1 {
		sum[i] = x + value2[i]
	}
	for i, x := range value1R {
		sumR[i] = x + value2R[i]
	}

	return &linAddRResult{
		Cache:      l.Cache,
		OutputVec:  sum,
		ROutputVec: sumR,
		SumVar:     rVar,
		Input:      in,
	}
}

type linAddResult struct {
	Cache     *VectorCache
	OutputVec linalg.Vector
	SumVar    *Variable
	Input     Result
}

func (l *linAddResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *linAddResult) Constant(g Gradient) bool {
	return l.Input.Constant(g) && l.SumVar.Constant(g)
}

func (l *linAddResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if sumGrad, ok := grad[l.SumVar]; ok {
		sumGrad.Add(upstream)
	}
	if !l.Input.Constant(grad) {
		l.Input.PropagateGradient(upstream, grad)
	}
}

func (l *linAddResult) Release() {
	l.Cache.Free(l.OutputVec)
	l.OutputVec = nil
	l.Input.Release()
}

type linAddRResult struct {
	Cache      *VectorCache
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

func (l *linAddRResult) Release() {
	l.Cache.Free(l.OutputVec)
	l.Cache.Free(l.ROutputVec)
	l.OutputVec = nil
	l.ROutputVec = nil
	l.SumVar.Release()
	l.Input.Release()
}
