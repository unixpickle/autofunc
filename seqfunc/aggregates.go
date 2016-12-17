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
