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
	poolVar := &Variable{r.Output()}
	return &pooledResult{
		Input:   r,
		PoolVar: poolVar,
		FOutput: f(poolVar),
	}
}

// PoolR is like Pool, but for RResults.
func PoolR(r RResult, f func(RResult) RResult) RResult {
	rvar := &RVariable{
		Variable:   &Variable{r.Output()},
		ROutputVec: r.ROutput(),
	}
	return &pooledRResult{
		Input:   r,
		PoolVar: rvar,
		FOutput: f(rvar),
	}
}

type pooledResult struct {
	Input   Result
	PoolVar *Variable
	FOutput Result
}

func (p *pooledResult) Output() linalg.Vector {
	return p.FOutput.Output()
}

func (p *pooledResult) Constant(g Gradient) bool {
	if !p.FOutput.Constant(g) {
		return false
	} else if p.Input.Constant(g) {
		return true
	} else {
		return p.FOutput.Constant(Gradient{p.PoolVar: linalg.Vector{}})
	}
}

func (p *pooledResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	inConst := p.Input.Constant(grad)
	if !inConst {
		grad[p.PoolVar] = make(linalg.Vector, len(p.PoolVar.Vector))
	}
	p.FOutput.PropagateGradient(upstream, grad)
	if !inConst {
		upsGrad := grad[p.PoolVar]
		delete(grad, p.PoolVar)
		p.Input.PropagateGradient(upsGrad, grad)
	}
}

func (p *pooledResult) Release() {
	p.FOutput.Release()
	p.Input.Release()
}

type pooledRResult struct {
	Input   RResult
	PoolVar *RVariable
	FOutput RResult
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
	} else if p.Input.Constant(rg, g) {
		return true
	} else {
		return p.FOutput.Constant(RGradient{p.PoolVar.Variable: linalg.Vector{}}, nil)
	}
}

func (p *pooledRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	inConst := p.Input.Constant(rgrad, grad)
	if !inConst {
		rgrad[p.PoolVar.Variable] = make(linalg.Vector, len(p.PoolVar.Variable.Vector))
		if grad == nil {
			grad = Gradient{}
		}
		grad[p.PoolVar.Variable] = make(linalg.Vector, len(p.PoolVar.Variable.Vector))
	}
	p.FOutput.PropagateRGradient(upstream, upstreamR, rgrad, grad)
	if !inConst {
		upsGradR := rgrad[p.PoolVar.Variable]
		upsGrad := grad[p.PoolVar.Variable]
		delete(grad, p.PoolVar.Variable)
		delete(rgrad, p.PoolVar.Variable)
		p.Input.PropagateRGradient(upsGrad, upsGradR, rgrad, grad)
	}
}

func (p *pooledRResult) Release() {
	p.FOutput.Release()
	p.Input.Release()
}
