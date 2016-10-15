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

// A MapBatcher maps a Batcher over input sequences by
// feeding the batcher all of the vectors in a given
// timestep at once.
//
// This can only operate on input sequence lists where all
// the vectors at a given timestep are the same size.
type MapBatcher struct {
	B autofunc.Batcher
}

// ApplySeqs generates a new sequence list by mapping the
// batcher over the sequence list.
func (m *MapBatcher) ApplySeqs(r Result) Result {
	res := &mapBatcherResult{
		Input:  r,
		Output: make([][]linalg.Vector, len(r.OutputSeqs())),
	}

	maxLen := maxSequenceLen(r.OutputSeqs())
	for t := 0; t < maxLen; t++ {
		joined, n := joinTime(r.OutputSeqs(), t)
		inVar := &autofunc.Variable{Vector: joined}
		output := m.B.Batch(inVar, n)
		res.Pool = append(res.Pool, inVar)
		res.Results = append(res.Results, output)
		appendSplit(r.OutputSeqs(), t, res.Output, output.Output())
	}

	return res
}

type mapBatcherResult struct {
	Input   Result
	Pool    []*autofunc.Variable
	Results []autofunc.Result
	Output  [][]linalg.Vector
}

func (m *mapBatcherResult) OutputSeqs() [][]linalg.Vector {
	return m.Output
}

func (m *mapBatcherResult) PropagateGradient(u [][]linalg.Vector, g autofunc.Gradient) {
	maxLen := maxSequenceLen(u)
	downstream := make([][]linalg.Vector, len(u))
	for t := 0; t < maxLen; t++ {
		upstream, _ := joinTime(u, t)
		poolVar := m.Pool[t]
		g[poolVar] = make(linalg.Vector, len(poolVar.Vector))
		m.Results[t].PropagateGradient(upstream, g)
		appendSplit(u, t, downstream, g[poolVar])
		delete(g, poolVar)
	}
	m.Input.PropagateGradient(downstream, g)
}

// A MapRBatcher is like a MapBatcher but for an RBatcher.
type MapRBatcher struct {
	B autofunc.RBatcher
}

// ApplySeqs is like MapBatcher.ApplySeqs.
func (m *MapRBatcher) ApplySeqs(r Result) Result {
	mb := MapBatcher{B: m.B}
	return mb.ApplySeqs(r)
}

// ApplySeqs is like ApplySeqs, but for RResults.
func (m *MapRBatcher) ApplySeqsR(rv autofunc.RVector, r RResult) RResult {
	res := &mapBatcherRResult{
		Input:   r,
		Output:  make([][]linalg.Vector, len(r.OutputSeqs())),
		ROutput: make([][]linalg.Vector, len(r.OutputSeqs())),
	}

	maxLen := maxSequenceLen(r.OutputSeqs())
	for t := 0; t < maxLen; t++ {
		joined, n := joinTime(r.OutputSeqs(), t)
		joinedR, _ := joinTime(r.ROutputSeqs(), t)
		inVar := &autofunc.RVariable{
			Variable:   &autofunc.Variable{Vector: joined},
			ROutputVec: joinedR,
		}
		output := m.B.BatchR(rv, inVar, n)
		res.Pool = append(res.Pool, inVar.Variable)
		res.Results = append(res.Results, output)
		appendSplit(r.OutputSeqs(), t, res.Output, output.Output())
		appendSplit(r.ROutputSeqs(), t, res.ROutput, output.ROutput())
	}

	return res
}

type mapBatcherRResult struct {
	Input   RResult
	Pool    []*autofunc.Variable
	Results []autofunc.RResult
	Output  [][]linalg.Vector
	ROutput [][]linalg.Vector
}

func (m *mapBatcherRResult) OutputSeqs() [][]linalg.Vector {
	return m.Output
}

func (m *mapBatcherRResult) ROutputSeqs() [][]linalg.Vector {
	return m.ROutput
}

func (m *mapBatcherRResult) PropagateRGradient(u, uR [][]linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	if g == nil {
		// We use g for temporary gradients.
		g = autofunc.Gradient{}
	}

	maxLen := maxSequenceLen(u)
	downstream := make([][]linalg.Vector, len(u))
	downstreamR := make([][]linalg.Vector, len(u))
	for t := 0; t < maxLen; t++ {
		upstream, _ := joinTime(u, t)
		upstreamR, _ := joinTime(uR, t)
		poolVar := m.Pool[t]
		rg[poolVar] = make(linalg.Vector, len(poolVar.Vector))
		g[poolVar] = make(linalg.Vector, len(poolVar.Vector))
		m.Results[t].PropagateRGradient(upstream, upstreamR, rg, g)
		appendSplit(u, t, downstream, g[poolVar])
		appendSplit(uR, t, downstreamR, rg[poolVar])
		delete(g, poolVar)
		delete(rg, poolVar)
	}
	m.Input.PropagateRGradient(downstream, downstreamR, rg, g)
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

func maxSequenceLen(seqs [][]linalg.Vector) int {
	var max int
	for _, x := range seqs {
		if len(x) > max {
			max = len(x)
		}
	}
	return max
}

func joinTime(seqs [][]linalg.Vector, t int) (joined linalg.Vector, n int) {
	for _, s := range seqs {
		if len(s) > t {
			joined = append(joined, s[t]...)
			n++
		}
	}
	return
}

func appendSplit(seqs [][]linalg.Vector, t int, dest [][]linalg.Vector, joined linalg.Vector) {
	_, n := joinTime(seqs, t)
	splitSize := len(joined) / n
	if len(joined)%n != 0 {
		panic("cannot evenly split vector")
	}
	for lane, seq := range seqs {
		if len(seq) > t {
			dest[lane] = append(dest[lane], joined[:splitSize])
			joined = joined[splitSize:]
		}
	}
}
