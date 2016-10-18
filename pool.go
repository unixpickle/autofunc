package autofunc

import "github.com/unixpickle/num-analysis/linalg"

// Pool evaluates the function f in such a way that
// its input r will only be propagated through once
// during back propagation.
// It is useful when a Result is expensive to
// propagate through, and said result needs to be
// used more than once.
//
// For example, suppose r is the result of a neural
// network and f evaluates the cost function for the
// neural network.
// If f uses r multiple times, then back propagating
// through the result of f will call
// r.PropagateGradient() more than once.
// To address this, Pool allows f to back-propagate
// completely and accumulate the total gradient with
// respect to r before r.PropagateGradient().
func Pool(r Result, f func(Result) Result) Result {
	return PoolAll([]Result{r}, func(in []Result) Result {
		return f(in[0])
	})
}

// PoolR is like Pool, but for RResults.
func PoolR(r RResult, f func(RResult) RResult) RResult {
	return PoolAllR([]RResult{r}, func(in []RResult) RResult {
		return f(in[0])
	})
}

// PoolAll is like Pool, but it can pool multiple Results.
// Each entry in ins will only be back-propagated through
// one time for each back-propagation through the result
// of the pool.
func PoolAll(ins []Result, f func([]Result) Result) Result {
	poolVars := make([]*Variable, len(ins))
	poolRes := make([]Result, len(ins))
	for i, in := range ins {
		poolVars[i] = &Variable{Vector: in.Output()}
		poolRes[i] = poolVars[i]
	}
	return &pooledResult{
		Inputs:   ins,
		PoolVars: poolVars,
		FOutput:  f(poolRes),
	}
}

// PoolAllR is like PoolAll, but for RResults.
func PoolAllR(ins []RResult, f func([]RResult) RResult) RResult {
	poolVars := make([]*Variable, len(ins))
	poolRes := make([]RResult, len(ins))
	for i, in := range ins {
		poolVars[i] = &Variable{Vector: in.Output()}
		poolRes[i] = &RVariable{
			Variable:   poolVars[i],
			ROutputVec: in.ROutput(),
		}
	}
	return &pooledRResult{
		Inputs:   ins,
		PoolVars: poolVars,
		FOutput:  f(poolRes),
	}
}

// PoolSplit slices a Result into n even parts and passes
// the split parts into a function f.
// It pools the sliced parts so that the result of f only
// back-propagates once through the original input r.
//
// The input's length must be divisible by n.
func PoolSplit(n int, r Result, f func([]Result) Result) Result {
	return Pool(r, func(in Result) Result {
		// This is efficient because Slice's result checks if
		// its input was a *Variable.
		return f(Split(n, in))
	})
}

// PoolSplitR is like PoolSplit, but for RResults.
func PoolSplitR(n int, r RResult, f func([]RResult) RResult) RResult {
	return PoolR(r, func(in RResult) RResult {
		return f(SplitR(n, in))
	})
}

type pooledResult struct {
	Inputs   []Result
	PoolVars []*Variable
	FOutput  Result
}

func (p *pooledResult) Output() linalg.Vector {
	return p.FOutput.Output()
}

func (p *pooledResult) Constant(g Gradient) bool {
	if !p.FOutput.Constant(g) {
		return false
	}
	for i, v := range p.PoolVars {
		if !p.FOutput.Constant(Gradient{v: linalg.Vector{}}) &&
			!p.Inputs[i].Constant(g) {
			return false
		}
	}
	return true
}

func (p *pooledResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	constants := make([]bool, len(p.PoolVars))
	for i, v := range p.PoolVars {
		constants[i] = p.FOutput.Constant(Gradient{v: linalg.Vector{}}) ||
			p.Inputs[i].Constant(grad)
		if !constants[i] {
			grad[v] = make(linalg.Vector, len(v.Vector))
		}
	}
	p.FOutput.PropagateGradient(upstream, grad)
	upstreams := make([]linalg.Vector, len(p.PoolVars))
	for i, v := range p.PoolVars {
		if !constants[i] {
			upstreams[i] = grad[v]
			delete(grad, v)
		}
	}
	for i, c := range constants {
		if !c {
			p.Inputs[i].PropagateGradient(upstreams[i], grad)
		}
	}
}

type pooledRResult struct {
	Inputs   []RResult
	PoolVars []*Variable
	FOutput  RResult
}

func (p *pooledRResult) Output() linalg.Vector {
	return p.FOutput.Output()
}

func (p *pooledRResult) ROutput() linalg.Vector {
	return p.FOutput.ROutput()
}

func (p *pooledRResult) Constant(rg RGradient, g Gradient) bool {
	if !p.FOutput.Constant(rg, g) {
		return false
	}
	for i, v := range p.PoolVars {
		if !p.FOutput.Constant(RGradient{v: linalg.Vector{}}, nil) &&
			!p.Inputs[i].Constant(rg, g) {
			return false
		}
	}
	return true
}

func (p *pooledRResult) PropagateRGradient(upstream, upstreamR linalg.Vector, rgrad RGradient,
	grad Gradient) {
	if grad == nil {
		grad = Gradient{}
	}
	constants := make([]bool, len(p.PoolVars))
	for i, v := range p.PoolVars {
		constants[i] = p.FOutput.Constant(RGradient{v: linalg.Vector{}}, nil) ||
			p.Inputs[i].Constant(rgrad, grad)
		if !constants[i] {
			grad[v] = make(linalg.Vector, len(v.Vector))
			rgrad[v] = make(linalg.Vector, len(v.Vector))
		}
	}
	p.FOutput.PropagateRGradient(upstream, upstreamR, rgrad, grad)
	upstreams := make([]linalg.Vector, len(p.PoolVars))
	upstreamsR := make([]linalg.Vector, len(p.PoolVars))
	for i, v := range p.PoolVars {
		if !constants[i] {
			upstreams[i] = grad[v]
			upstreamsR[i] = rgrad[v]
			delete(grad, v)
			delete(rgrad, v)
		}
	}
	for i, c := range constants {
		if !c {
			p.Inputs[i].PropagateRGradient(upstreams[i], upstreamsR[i], rgrad, grad)
		}
	}
}
