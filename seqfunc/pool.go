package seqfunc

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

type poolResult struct {
	Input  Result
	Pool   [][]*autofunc.Variable
	Output Result
}

// Pool is like autofunc.Pool, but for a sequence list.
func Pool(in Result, f func(Result) Result) Result {
	pool := make([][]*autofunc.Variable, len(in.OutputSeqs()))
	for i, seq := range in.OutputSeqs() {
		pool[i] = make([]*autofunc.Variable, len(seq))
		for j, x := range seq {
			pool[i][j] = &autofunc.Variable{Vector: x}
		}
	}
	return &poolResult{
		Input:  in,
		Pool:   pool,
		Output: f(VarResult(pool)),
	}
}

func (p *poolResult) OutputSeqs() [][]linalg.Vector {
	return p.Output.OutputSeqs()
}

func (p *poolResult) PropagateGradient(u [][]linalg.Vector, g autofunc.Gradient) {
	for _, seq := range p.Pool {
		for _, v := range seq {
			g[v] = make(linalg.Vector, len(v.Vector))
		}
	}
	p.Output.PropagateGradient(u, g)
	downstream := make([][]linalg.Vector, len(p.Pool))
	for i, seq := range p.Pool {
		downstream[i] = make([]linalg.Vector, len(seq))
		for j, v := range seq {
			downstream[i][j] = g[v]
			delete(g, v)
		}
	}
	p.Input.PropagateGradient(downstream, g)
}

type poolRResult struct {
	Input  RResult
	Pool   [][]*autofunc.Variable
	Output RResult
}

// PoolR is like Pool but for RResults.
func PoolR(in RResult, f func(RResult) RResult) RResult {
	pool := make([][]*autofunc.Variable, len(in.OutputSeqs()))
	rv := autofunc.RVector{}
	rOut := in.ROutputSeqs()
	for i, seq := range in.OutputSeqs() {
		pool[i] = make([]*autofunc.Variable, len(seq))
		for j, x := range seq {
			pool[i][j] = &autofunc.Variable{Vector: x}
			rv[pool[i][j]] = rOut[i][j]
		}
	}
	return &poolRResult{
		Input:  in,
		Pool:   pool,
		Output: f(VarRResult(rv, pool)),
	}
}

func (p *poolRResult) OutputSeqs() [][]linalg.Vector {
	return p.Output.OutputSeqs()
}

func (p *poolRResult) ROutputSeqs() [][]linalg.Vector {
	return p.Output.ROutputSeqs()
}

func (p *poolRResult) PropagateRGradient(u, uR [][]linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	if g == nil {
		g = autofunc.Gradient{}
	}
	for _, seq := range p.Pool {
		for _, v := range seq {
			g[v] = make(linalg.Vector, len(v.Vector))
			rg[v] = make(linalg.Vector, len(v.Vector))
		}
	}
	p.Output.PropagateRGradient(u, uR, rg, g)
	downstream := make([][]linalg.Vector, len(p.Pool))
	downstreamR := make([][]linalg.Vector, len(p.Pool))
	for i, seq := range p.Pool {
		downstream[i] = make([]linalg.Vector, len(seq))
		downstreamR[i] = make([]linalg.Vector, len(seq))
		for j, v := range seq {
			downstream[i][j] = g[v]
			downstreamR[i][j] = rg[v]
			delete(g, v)
			delete(rg, v)
		}
	}
	p.Input.PropagateRGradient(downstream, downstreamR, rg, g)
}
