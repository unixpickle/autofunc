package seqfunc

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

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

// A FixedMapBatcher provides the same API as a MapBatcher
// without restricting itself to calling the Batcher with
// one input from each sequence.
// Instead, inputs to the Batcher may be selected from the
// same sequence or from different sequences, allowing for
// a fixed BatchSize regardless of the number of input
// sequences.
//
// One use of FixedMapBatcher is to get the performance
// benefits of a MapBatcher without running multiple
// sequences in batch.
//
// This only works if all vector across all sequences are
// the same length.
type FixedMapBatcher struct {
	B         autofunc.Batcher
	BatchSize int
}

// ApplySeqs maps the batcher over an input.
func (f *FixedMapBatcher) ApplySeqs(r Result) Result {
	var poolVars []*autofunc.Variable
	var poolRes []autofunc.Result
	for _, seq := range r.OutputSeqs() {
		for _, vec := range seq {
			v := &autofunc.Variable{Vector: vec}
			poolVars = append(poolVars, v)
			poolRes = append(poolRes, v)
		}
	}
	var batchResults []autofunc.Result
	var batchOutputs []linalg.Vector
	for i := 0; i < len(poolRes); i += f.BatchSize {
		bs := f.BatchSize
		if i+bs > len(poolRes) {
			bs = len(poolRes) - i
		}
		input := autofunc.Concat(poolRes[i : i+bs]...)
		br := f.B.Batch(input, bs)
		batchResults = append(batchResults, br)
		batchOutputs = append(batchOutputs, br.Output())
	}
	split := splitBatchVecs(batchOutputs, r.OutputSeqs(), f.BatchSize)
	return &fixedMapBatcherResult{
		Batch:   f.BatchSize,
		Input:   r,
		Pool:    poolVars,
		Results: batchResults,
		Output:  split,
	}
}

type fixedMapBatcherResult struct {
	Batch   int
	Input   Result
	Pool    []*autofunc.Variable
	Results []autofunc.Result
	Output  [][]linalg.Vector
}

func (f *fixedMapBatcherResult) OutputSeqs() [][]linalg.Vector {
	return f.Output
}

func (f *fixedMapBatcherResult) PropagateGradient(u [][]linalg.Vector, g autofunc.Gradient) {
	for _, p := range f.Pool {
		g[p] = make(linalg.Vector, len(p.Vector))
	}

	joinedUp := joinBatchVecs(u, f.Batch)
	for i, r := range f.Results {
		r.PropagateGradient(joinedUp[i], g)
	}

	var down []linalg.Vector
	for _, p := range f.Pool {
		down = append(down, g[p])
		delete(g, p)
	}

	downVecs := splitBatchVecs(down, u, 1)
	f.Input.PropagateGradient(downVecs, g)
}

// FixedMapRBatcher is like FixedMapBatcher but with
// support for RResults.
type FixedMapRBatcher struct {
	B         autofunc.RBatcher
	BatchSize int
}

// ApplySeqs maps the batcher over an input.
func (f *FixedMapRBatcher) ApplySeqs(r Result) Result {
	x := &FixedMapBatcher{B: f.B, BatchSize: f.BatchSize}
	return x.ApplySeqs(r)
}

// ApplySeqsR maps the batcher over an input.
func (f *FixedMapRBatcher) ApplySeqsR(rv autofunc.RVector, r RResult) RResult {
	var poolVars []*autofunc.Variable
	var poolRes []autofunc.RResult
	rOut := r.ROutputSeqs()
	for i, seq := range r.OutputSeqs() {
		for j, vec := range seq {
			v := &autofunc.Variable{Vector: vec}
			poolVars = append(poolVars, v)
			poolRes = append(poolRes, &autofunc.RVariable{
				Variable:   v,
				ROutputVec: rOut[i][j],
			})
		}
	}
	var batchResults []autofunc.RResult
	var batchOutputs []linalg.Vector
	var batchOutputsR []linalg.Vector
	for i := 0; i < len(poolRes); i += f.BatchSize {
		bs := f.BatchSize
		if i+bs > len(poolRes) {
			bs = len(poolRes) - i
		}
		input := autofunc.ConcatR(poolRes[i : i+bs]...)
		br := f.B.BatchR(rv, input, bs)
		batchResults = append(batchResults, br)
		batchOutputs = append(batchOutputs, br.Output())
		batchOutputsR = append(batchOutputsR, br.ROutput())
	}
	split := splitBatchVecs(batchOutputs, r.OutputSeqs(), f.BatchSize)
	splitR := splitBatchVecs(batchOutputsR, r.OutputSeqs(), f.BatchSize)
	return &fixedMapRBatcherResult{
		Batch:   f.BatchSize,
		Input:   r,
		Pool:    poolVars,
		Results: batchResults,
		Output:  split,
		ROutput: splitR,
	}
}

type fixedMapRBatcherResult struct {
	Batch   int
	Input   RResult
	Pool    []*autofunc.Variable
	Results []autofunc.RResult
	Output  [][]linalg.Vector
	ROutput [][]linalg.Vector
}

func (f *fixedMapRBatcherResult) OutputSeqs() [][]linalg.Vector {
	return f.Output
}

func (f *fixedMapRBatcherResult) ROutputSeqs() [][]linalg.Vector {
	return f.ROutput
}

func (f *fixedMapRBatcherResult) PropagateRGradient(u, uR [][]linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	if g == nil {
		g = autofunc.Gradient{}
	}
	for _, p := range f.Pool {
		g[p] = make(linalg.Vector, len(p.Vector))
		rg[p] = make(linalg.Vector, len(p.Vector))
	}

	joinedUp := joinBatchVecs(u, f.Batch)
	joinedUpR := joinBatchVecs(uR, f.Batch)
	for i, r := range f.Results {
		r.PropagateRGradient(joinedUp[i], joinedUpR[i], rg, g)
	}

	var down, downR []linalg.Vector
	for _, p := range f.Pool {
		down = append(down, g[p])
		downR = append(downR, rg[p])
		delete(g, p)
		delete(rg, p)
	}

	downVecs := splitBatchVecs(down, u, 1)
	downVecsR := splitBatchVecs(downR, u, 1)
	f.Input.PropagateRGradient(downVecs, downVecsR, rg, g)
}

func splitBatchVecs(toSplit []linalg.Vector, shaper [][]linalg.Vector,
	batch int) [][]linalg.Vector {
	var totalCount int
	for _, x := range shaper {
		totalCount += len(x)
	}

	var res [][]linalg.Vector
	var batchIdx, subIdx int
	for _, seq := range shaper {
		var resSeq []linalg.Vector
		for _ = range seq {
			fullVec := toSplit[batchIdx]
			n := batch
			if batchIdx == len(toSplit)-1 && totalCount%batch != 0 {
				n = totalCount % batch
			}
			subSize := len(fullVec) / n
			subVec := fullVec[subIdx*subSize : (subIdx+1)*subSize]
			subIdx++
			if subIdx == n {
				batchIdx++
				subIdx = 0
			}
			resSeq = append(resSeq, subVec)
		}
		res = append(res, resSeq)
	}

	return res
}

func joinBatchVecs(join [][]linalg.Vector, batch int) []linalg.Vector {
	var allVecs []linalg.Vector
	for _, seq := range join {
		for _, v := range seq {
			allVecs = append(allVecs, v)
		}
	}

	var res []linalg.Vector
	for i := 0; i < len(allVecs); i += batch {
		var joined linalg.Vector
		for j := i; j < i+batch && j < len(allVecs); j++ {
			joined = append(joined, allVecs[j]...)
		}
		res = append(res, joined)
	}

	return res
}
