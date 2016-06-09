package autofunc

import "github.com/unixpickle/num-analysis/linalg"

type joinedResults struct {
	Cache     *VectorCache
	OutputVec linalg.Vector
	Results   []Result
}

// Concat joins the outputs of several Results.
// The results are concatenated first to last,
// so Concat({1,2,3}, {4,5,6}) = {1,2,3,4,5,6}.
func Concat(rs ...Result) Result {
	return ConcatCache(nil, rs...)
}

// ConcatCache is like Concat, but it lets you
// specify which VectorCache to use.
func ConcatCache(c *VectorCache, rs ...Result) Result {
	outputs := make([]linalg.Vector, len(rs))
	var totalLen int
	for i, x := range rs {
		outputs[i] = x.Output()
		totalLen += len(outputs[i])
	}

	outVec := c.Alloc(totalLen)
	vecIdx := 0
	for _, x := range outputs {
		copy(outVec[vecIdx:], x)
		vecIdx += len(x)
	}

	return &joinedResults{
		Cache:     c,
		OutputVec: outVec,
		Results:   rs,
	}
}

func (j *joinedResults) Output() linalg.Vector {
	return j.OutputVec
}

func (j *joinedResults) Constant(g Gradient) bool {
	for _, x := range j.Results {
		if !x.Constant(g) {
			return false
		}
	}
	return true
}

func (j *joinedResults) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	vecIdx := 0
	for _, x := range j.Results {
		if !x.Constant(grad) {
			l := len(x.Output())
			x.PropagateGradient(upstream[vecIdx:vecIdx+l], grad)
			vecIdx += l
		}
	}
}

func (j *joinedResults) Release() {
	j.Cache.Free(j.OutputVec)
	j.OutputVec = nil
	for _, r := range j.Results {
		r.Release()
	}
}

type joinedRResults struct {
	Cache      *VectorCache
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Results    []RResult
}

// ConcatR is like Concat, but for RResults.
func ConcatR(rs ...RResult) RResult {
	return ConcatCacheR(nil, rs...)
}

// ConcatCacheR is like ConcatR, but it lets you
// specify which VectorCache to use.
func ConcatCacheR(c *VectorCache, rs ...RResult) RResult {
	outputs := make([]linalg.Vector, len(rs))
	routputs := make([]linalg.Vector, len(rs))
	var totalLen int
	for i, x := range rs {
		outputs[i] = x.Output()
		routputs[i] = x.ROutput()
		totalLen += len(outputs[i])
	}

	outVec := c.Alloc(totalLen)
	outVecR := c.Alloc(totalLen)
	vecIdx := 0
	for i, x := range outputs {
		copy(outVec[vecIdx:], x)
		copy(outVecR[vecIdx:], routputs[i])
		vecIdx += len(x)
	}

	return &joinedRResults{
		Cache:      c,
		OutputVec:  outVec,
		ROutputVec: outVecR,
		Results:    rs,
	}
}

func (j *joinedRResults) Output() linalg.Vector {
	return j.OutputVec
}

func (j *joinedRResults) ROutput() linalg.Vector {
	return j.ROutputVec
}

func (j *joinedRResults) Constant(rg RGradient, g Gradient) bool {
	for _, x := range j.Results {
		if !x.Constant(rg, g) {
			return false
		}
	}
	return true
}

func (j *joinedRResults) PropagateRGradient(upstream, upstreamR linalg.Vector,
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

func (j *joinedRResults) Release() {
	j.Cache.Free(j.OutputVec)
	j.Cache.Free(j.ROutputVec)
	j.OutputVec = nil
	j.ROutputVec = nil
	for _, r := range j.Results {
		r.Release()
	}
}
