package seqfunc

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// A MapFunc applies a differentiable function on each of
// the vectors in lists of vector sequences.
type MapFunc struct {
	F autofunc.Func
}

// ApplySeqs runs m.F on every vector in r and returns the
// resulting sequence list.
func (m *MapFunc) ApplySeqs(r Result) Result {
	return Map(r, m.F.Apply)
}

// A MapRFunc is like a MapFunc, but for RResults.
type MapRFunc struct {
	F autofunc.RFunc
}

// ApplySeqs is like MapFunc.ApplySeqs.
func (m *MapRFunc) ApplySeqs(r Result) Result {
	return Map(r, m.F.Apply)
}

// ApplySeqsR is like ApplySeqs but for RResults.
func (m *MapRFunc) ApplySeqsR(rv autofunc.RVector, r RResult) RResult {
	return MapR(r, func(in autofunc.RResult) autofunc.RResult {
		return m.F.ApplyR(rv, in)
	})
}

// Map calls the passed function for each vector in r,
// generating a new sequence of outputs.
func Map(r Result, f func(in autofunc.Result) autofunc.Result) Result {
	pool := make([][]*autofunc.Variable, len(r.OutputSeqs()))
	res := make([][]autofunc.Result, len(pool))
	out := make([][]linalg.Vector, len(pool))
	for i, seq := range r.OutputSeqs() {
		for _, vec := range seq {
			p := &autofunc.Variable{Vector: vec}
			subRes := f(p)
			pool[i] = append(pool[i], p)
			res[i] = append(res[i], subRes)
			out[i] = append(out[i], subRes.Output())
		}
	}
	return &mapResult{
		Input:  r,
		Pool:   pool,
		Res:    res,
		Output: out,
	}
}

// MapR is like Map but for RResults.
func MapR(r RResult, f func(in autofunc.RResult) autofunc.RResult) RResult {
	pool := make([][]*autofunc.Variable, len(r.OutputSeqs()))
	res := make([][]autofunc.RResult, len(pool))
	out := make([][]linalg.Vector, len(pool))
	outR := make([][]linalg.Vector, len(pool))
	rSeqs := r.ROutputSeqs()
	for i, seq := range r.OutputSeqs() {
		seqR := rSeqs[i]
		for j, vec := range seq {
			vecR := seqR[j]
			p := &autofunc.Variable{Vector: vec}
			subRes := f(&autofunc.RVariable{
				Variable:   p,
				ROutputVec: vecR,
			})
			pool[i] = append(pool[i], p)
			res[i] = append(res[i], subRes)
			out[i] = append(out[i], subRes.Output())
			outR[i] = append(outR[i], subRes.ROutput())
		}
	}
	return &mapRResult{
		Input:   r,
		Pool:    pool,
		Res:     res,
		Output:  out,
		ROutput: outR,
	}
}

type mapResult struct {
	Input  Result
	Pool   [][]*autofunc.Variable
	Res    [][]autofunc.Result
	Output [][]linalg.Vector
}

func (m *mapResult) OutputSeqs() [][]linalg.Vector {
	return m.Output
}

func (m *mapResult) PropagateGradient(u [][]linalg.Vector, g autofunc.Gradient) {
	downstream := make([][]linalg.Vector, len(u))
	for i, uSeq := range u {
		for j, uVec := range uSeq {
			pool := m.Pool[i][j]
			g[pool] = make(linalg.Vector, len(pool.Vector))
			uCopy := make(linalg.Vector, len(uVec))
			copy(uCopy, uVec)
			m.Res[i][j].PropagateGradient(uCopy, g)
			downstream[i] = append(downstream[i], g[pool])
			delete(g, pool)
		}
	}
	m.Input.PropagateGradient(downstream, g)
}

type mapRResult struct {
	Input   RResult
	Pool    [][]*autofunc.Variable
	Res     [][]autofunc.RResult
	Output  [][]linalg.Vector
	ROutput [][]linalg.Vector
}

func (m *mapRResult) OutputSeqs() [][]linalg.Vector {
	return m.Output
}

func (m *mapRResult) ROutputSeqs() [][]linalg.Vector {
	return m.ROutput
}

func (m *mapRResult) PropagateRGradient(u, uR [][]linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	downstream := make([][]linalg.Vector, len(u))
	downstreamR := make([][]linalg.Vector, len(u))
	for i, uSeq := range u {
		for j, uVec := range uSeq {
			uVecR := uR[i][j]
			pool := m.Pool[i][j]
			g[pool] = make(linalg.Vector, len(pool.Vector))
			rg[pool] = make(linalg.Vector, len(pool.Vector))
			uCopy := make(linalg.Vector, len(uVec))
			copy(uCopy, uVec)
			uCopyR := make(linalg.Vector, len(uVecR))
			copy(uCopyR, uVecR)
			m.Res[i][j].PropagateRGradient(uCopy, uCopyR, rg, g)
			downstream[i] = append(downstream[i], g[pool])
			downstreamR[i] = append(downstreamR[i], rg[pool])
			delete(g, pool)
			delete(rg, pool)
		}
	}
	m.Input.PropagateRGradient(downstream, downstreamR, rg, g)
}
