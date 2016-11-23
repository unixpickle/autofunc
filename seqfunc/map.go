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

// Map maps a function over a sequence, producing a new
// sequence whose entries are the result mapping.
func Map(in Result, f func(in autofunc.Result) autofunc.Result) Result {
	return MapN(func(ins ...autofunc.Result) autofunc.Result {
		return f(ins[0])
	}, in)
}

// MapR is like Map but for RResults.
func MapR(in RResult, f func(in autofunc.RResult) autofunc.RResult) RResult {
	return MapNR(func(ins ...autofunc.RResult) autofunc.RResult {
		return f(ins[0])
	}, in)
}

// MapN takes one or more input sequences and maps a
// function over them.
// The function will receive one argument for each input
// sequence list.
//
// The input sequence lists must have the same lengths,
// and corresponding sequences must also match in length,
// but the vectors in a given position may differ in size
// across different input sequence lists.
func MapN(f func(in ...autofunc.Result) autofunc.Result, in1 Result, ins ...Result) Result {
	allIns := append([]Result{in1}, ins...)
	allLists := make([][][]linalg.Vector, len(allIns))
	for i, x := range allIns {
		allLists[i] = x.OutputSeqs()
		if i > 0 {
			if len(allLists[i]) != len(allLists[0]) {
				panic("input dimensions mismatch")
			}
			for j, seq := range allLists[i] {
				if len(seq) != len(allLists[0][j]) {
					panic("input dimensions mismatch")
				}
			}
		}
	}

	pool := make([][][]*autofunc.Variable, len(allIns))
	for i := range pool {
		pool[i] = make([][]*autofunc.Variable, len(allLists[0]))
	}
	res := make([][]autofunc.Result, len(allLists[0]))
	out := make([][]linalg.Vector, len(allLists[0]))

	for i, seq := range allLists[0] {
		for j := range seq {
			var mapIns []autofunc.Result
			for k, list := range allLists {
				p := &autofunc.Variable{Vector: list[i][j]}
				pool[k][i] = append(pool[k][i], p)
				mapIns = append(mapIns, p)
			}
			subRes := f(mapIns...)
			res[i] = append(res[i], subRes)
			out[i] = append(out[i], subRes.Output())
		}
	}

	return &mapResult{
		Inputs: allIns,
		Pool:   pool,
		Res:    res,
		Output: out,
	}
}

// MapNR is like MapN, but for RResults.
func MapNR(f func(in ...autofunc.RResult) autofunc.RResult, in1 RResult, ins ...RResult) RResult {
	allIns := append([]RResult{in1}, ins...)
	allLists := make([][][]linalg.Vector, len(allIns))
	allListsR := make([][][]linalg.Vector, len(allIns))
	for i, x := range allIns {
		allLists[i] = x.OutputSeqs()
		allListsR[i] = x.ROutputSeqs()
		if i > 0 {
			if len(allLists[i]) != len(allLists[0]) {
				panic("input dimensions mismatch")
			}
			for j, seq := range allLists[i] {
				if len(seq) != len(allLists[0][j]) {
					panic("input dimensions mismatch")
				}
			}
		}
	}

	pool := make([][][]*autofunc.Variable, len(allIns))
	for i := range pool {
		pool[i] = make([][]*autofunc.Variable, len(allLists[0]))
	}
	res := make([][]autofunc.RResult, len(allLists[0]))
	out := make([][]linalg.Vector, len(allLists[0]))
	outR := make([][]linalg.Vector, len(allLists[0]))

	for i, seq := range allLists[0] {
		for j := range seq {
			var mapIns []autofunc.RResult
			for k, list := range allLists {
				p := &autofunc.Variable{Vector: list[i][j]}
				pool[k][i] = append(pool[k][i], p)
				mapIns = append(mapIns, &autofunc.RVariable{
					Variable:   p,
					ROutputVec: allListsR[k][i][j],
				})
			}
			subRes := f(mapIns...)
			res[i] = append(res[i], subRes)
			out[i] = append(out[i], subRes.Output())
			outR[i] = append(outR[i], subRes.ROutput())
		}
	}

	return &mapRResult{
		Inputs:  allIns,
		Pool:    pool,
		Res:     res,
		Output:  out,
		ROutput: outR,
	}
}

type mapResult struct {
	Inputs []Result
	Pool   [][][]*autofunc.Variable
	Res    [][]autofunc.Result
	Output [][]linalg.Vector
}

func (m *mapResult) OutputSeqs() [][]linalg.Vector {
	return m.Output
}

func (m *mapResult) PropagateGradient(u [][]linalg.Vector, g autofunc.Gradient) {
	downstream := make([][][]linalg.Vector, len(m.Pool))
	for i := range downstream {
		downstream[i] = make([][]linalg.Vector, len(u))
	}
	for i, uSeq := range u {
		for j, uVec := range uSeq {
			for k := range m.Pool {
				pool := m.Pool[k][i][j]
				g[pool] = make(linalg.Vector, len(pool.Vector))
			}
			uCopy := make(linalg.Vector, len(uVec))
			copy(uCopy, uVec)
			m.Res[i][j].PropagateGradient(uCopy, g)
			for k := range m.Pool {
				pool := m.Pool[k][i][j]
				downstream[k][i] = append(downstream[k][i], g[pool])
				delete(g, pool)
			}
		}
	}
	for i, down := range downstream {
		m.Inputs[i].PropagateGradient(down, g)
	}
}

type mapRResult struct {
	Inputs  []RResult
	Pool    [][][]*autofunc.Variable
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
	if g == nil {
		g = autofunc.Gradient{}
	}
	downstream := make([][][]linalg.Vector, len(m.Pool))
	downstreamR := make([][][]linalg.Vector, len(m.Pool))
	for i := range downstream {
		downstream[i] = make([][]linalg.Vector, len(u))
		downstreamR[i] = make([][]linalg.Vector, len(u))
	}
	for i, uSeq := range u {
		for j, uVec := range uSeq {
			for k := range m.Pool {
				pool := m.Pool[k][i][j]
				g[pool] = make(linalg.Vector, len(pool.Vector))
				rg[pool] = make(linalg.Vector, len(pool.Vector))
			}
			uCopy := make(linalg.Vector, len(uVec))
			copy(uCopy, uVec)
			uCopyR := make(linalg.Vector, len(uVec))
			copy(uCopyR, uR[i][j])
			m.Res[i][j].PropagateRGradient(uCopy, uCopyR, rg, g)
			for k := range m.Pool {
				pool := m.Pool[k][i][j]
				downstream[k][i] = append(downstream[k][i], g[pool])
				downstreamR[k][i] = append(downstreamR[k][i], rg[pool])
				delete(g, pool)
				delete(rg, pool)
			}
		}
	}
	for i, down := range downstream {
		m.Inputs[i].PropagateRGradient(down, downstreamR[i], rg, g)
	}
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
