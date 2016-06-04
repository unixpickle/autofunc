package autofunc

import "github.com/unixpickle/num-analysis/linalg"

type JoinedResults struct {
	OutputVec linalg.Vector
	Results   []Result
}

// Concat joins the outputs of several Results.
// The results are concatenated first to last,
// so Concat({1,2,3}, {4,5,6}) = {1,2,3,4,5,6}.
func Concat(rs ...Result) *JoinedResults {
	outputs := make([]linalg.Vector, len(rs))
	var totalLen int
	for i, x := range rs {
		outputs[i] = x.Output()
		totalLen += len(outputs[i])
	}

	outVec := make(linalg.Vector, totalLen)
	vecIdx := 0
	for _, x := range outputs {
		copy(outVec[vecIdx:], x)
		vecIdx += len(x)
	}

	return &JoinedResults{
		OutputVec: outVec,
		Results:   rs,
	}
}

func (j *JoinedResults) Output() linalg.Vector {
	return j.OutputVec
}

func (j *JoinedResults) Constant(g Gradient) bool {
	for _, x := range j.Results {
		if !x.Constant(g) {
			return false
		}
	}
	return true
}

func (j *JoinedResults) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	vecIdx := 0
	for _, x := range j.Results {
		if !x.Constant(grad) {
			l := len(x.Output())
			x.PropagateGradient(upstream[vecIdx:vecIdx+l], grad)
			vecIdx += l
		}
	}
}

type JoinedRResults struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Results    []RResult
}

// ConcatR is like Concat, but for RResults.
func ConcatR(rs ...RResult) *JoinedRResults {
	outputs := make([]linalg.Vector, len(rs))
	routputs := make([]linalg.Vector, len(rs))
	var totalLen int
	for i, x := range rs {
		outputs[i] = x.Output()
		routputs[i] = x.ROutput()
		totalLen += len(outputs[i])
	}

	outVec := make(linalg.Vector, totalLen)
	outVecR := make(linalg.Vector, totalLen)
	vecIdx := 0
	for i, x := range outputs {
		copy(outVec[vecIdx:], x)
		copy(outVecR[vecIdx:], routputs[i])
		vecIdx += len(x)
	}

	return &JoinedRResults{
		OutputVec:  outVec,
		ROutputVec: outVecR,
		Results:    rs,
	}
}

func (j *JoinedRResults) Output() linalg.Vector {
	return j.OutputVec
}

func (j *JoinedRResults) ROutput() linalg.Vector {
	return j.ROutputVec
}

func (j *JoinedRResults) Constant(rg RGradient, g Gradient) bool {
	for _, x := range j.Results {
		if !x.Constant(rg, g) {
			return false
		}
	}
	return true
}

func (j *JoinedRResults) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	vecIdx := 0
	for _, x := range j.Results {
		if !x.Constant(rgrad, grad) {
			l := len(x.Output())
			x.PropagateRGradient(upstream[vecIdx:vecIdx+l], upstreamR[vecIdx:vecIdx+l],
				rgrad, grad)
			vecIdx += l
		}
	}
}
