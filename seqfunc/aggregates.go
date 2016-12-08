package seqfunc

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// AddAll adds all time steps of all the sequences and
// returns the total sum.
// This requires that every vector in every sequence be
// of the same length.
// The result's Constant() method will always give false.
func AddAll(in Result) autofunc.Result {
	var sum linalg.Vector
	for _, seq := range in.OutputSeqs() {
		for _, x := range seq {
			if sum == nil {
				sum = x.Copy()
			} else {
				sum.Add(x)
			}
		}
	}
	return &addAllResult{In: in, Sum: sum}
}

// AddAllR is like AddAll for RResults.
func AddAllR(in RResult) autofunc.RResult {
	var sum, sumR linalg.Vector
	seqsR := in.ROutputSeqs()
	for i, seq := range in.OutputSeqs() {
		for j, x := range seq {
			if sum == nil {
				sum = x.Copy()
				sumR = seqsR[i][j].Copy()
			} else {
				sum.Add(x)
				sumR.Add(seqsR[i][j])
			}
		}
	}
	return &addAllRResult{In: in, Sum: sum, SumR: sumR}
}

// Mean computes the overall mean of all the vectors in
// the sequence list.
// Like AddAll, Mean requires that all vectors in all
// sequences have the same length.
// It is invalid to compute the mean over a list of empty
// sequences or an empty list of sequences.
func Mean(in Result) autofunc.Result {
	sum := AddAll(in)
	return autofunc.Scale(sum, 1/float64(count(in.OutputSeqs())))
}

// MeanR is like Mean for RResults.
func MeanR(in RResult) autofunc.RResult {
	sum := AddAllR(in)
	return autofunc.ScaleR(sum, 1/float64(count(in.OutputSeqs())))
}

// ConcatAll joins all of the timesteps in all of the
// sequences in to one packed autofunc.Result.
// The packing is done as follows: first timesteps from
// the same sequence are packed left to right, then the
// packed vectors from each sequence are joined together
// from the first sequence to the last.
func ConcatAll(in Result) autofunc.Result {
	var joined linalg.Vector
	for _, seq := range in.OutputSeqs() {
		for _, vec := range seq {
			joined = append(joined, vec...)
		}
	}
	return &concatAllResult{Input: in, OutVec: joined}
}

// ConcatAllR is like ConcatAll for RResults.
func ConcatAllR(in RResult) autofunc.RResult {
	var joined, joinedR linalg.Vector
	rOut := in.ROutputSeqs()
	for i, seq := range in.OutputSeqs() {
		for j, vec := range seq {
			joined = append(joined, vec...)
			joinedR = append(joinedR, rOut[i][j]...)
		}
	}
	return &concatAllRResult{Input: in, OutVec: joined, ROutVec: joinedR}
}

func count(s [][]linalg.Vector) int {
	var res int
	for _, seq := range s {
		res += len(seq)
	}
	return res
}

type addAllResult struct {
	In  Result
	Sum linalg.Vector
}

func (a *addAllResult) Output() linalg.Vector {
	return a.Sum
}

func (a *addAllResult) Constant(g autofunc.Gradient) bool {
	return false
}

func (a *addAllResult) PropagateGradient(u linalg.Vector, g autofunc.Gradient) {
	upSeqs := make([][]linalg.Vector, len(a.In.OutputSeqs()))
	for i, outSeq := range a.In.OutputSeqs() {
		upSeqs[i] = make([]linalg.Vector, len(outSeq))
		for j := range upSeqs[i] {
			upSeqs[i][j] = u
		}
	}
	a.In.PropagateGradient(upSeqs, g)
}

type addAllRResult struct {
	In   RResult
	Sum  linalg.Vector
	SumR linalg.Vector
}

func (a *addAllRResult) Output() linalg.Vector {
	return a.Sum
}

func (a *addAllRResult) ROutput() linalg.Vector {
	return a.SumR
}

func (a *addAllRResult) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	return false
}

func (a *addAllRResult) PropagateRGradient(u, uR linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	upSeqs := make([][]linalg.Vector, len(a.In.OutputSeqs()))
	upSeqsR := make([][]linalg.Vector, len(a.In.OutputSeqs()))
	for i, outSeq := range a.In.OutputSeqs() {
		upSeqs[i] = make([]linalg.Vector, len(outSeq))
		upSeqsR[i] = make([]linalg.Vector, len(outSeq))
		for j := range upSeqs[i] {
			upSeqs[i][j] = u
			upSeqsR[i][j] = uR
		}
	}
	a.In.PropagateRGradient(upSeqs, upSeqsR, rg, g)
}

type concatAllResult struct {
	Input  Result
	OutVec linalg.Vector
}

func (a *concatAllResult) Output() linalg.Vector {
	return a.OutVec
}

func (a *concatAllResult) Constant(g autofunc.Gradient) bool {
	return false
}

func (a *concatAllResult) PropagateGradient(u linalg.Vector, g autofunc.Gradient) {
	var idx int
	var splitUpstream [][]linalg.Vector
	for _, outSeq := range a.Input.OutputSeqs() {
		var splitSeq []linalg.Vector
		for _, step := range outSeq {
			splitSeq = append(splitSeq, u[idx:idx+len(step)])
			idx += len(step)
		}
		splitUpstream = append(splitUpstream, splitSeq)
	}
	a.Input.PropagateGradient(splitUpstream, g)
}

type concatAllRResult struct {
	Input   RResult
	OutVec  linalg.Vector
	ROutVec linalg.Vector
}

func (a *concatAllRResult) Output() linalg.Vector {
	return a.OutVec
}

func (a *concatAllRResult) ROutput() linalg.Vector {
	return a.ROutVec
}

func (a *concatAllRResult) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	return false
}

func (a *concatAllRResult) PropagateRGradient(u, uR linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	var idx int
	var splitUpstream, splitUpstreamR [][]linalg.Vector
	for _, outSeq := range a.Input.OutputSeqs() {
		var splitSeq, splitSeqR []linalg.Vector
		for _, step := range outSeq {
			splitSeq = append(splitSeq, u[idx:idx+len(step)])
			splitSeqR = append(splitSeqR, uR[idx:idx+len(step)])
			idx += len(step)
		}
		splitUpstream = append(splitUpstream, splitSeq)
		splitUpstreamR = append(splitUpstreamR, splitSeqR)
	}
	a.Input.PropagateRGradient(splitUpstream, splitUpstreamR, rg, g)
}
