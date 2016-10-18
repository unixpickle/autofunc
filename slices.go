package autofunc

import "github.com/unixpickle/num-analysis/linalg"

type joinedResults struct {
	OutputVec linalg.Vector
	Results   []Result
}

// Concat joins the outputs of several Results.
// The results are concatenated first to last,
// so Concat({1,2,3}, {4,5,6}) = {1,2,3,4,5,6}.
func Concat(rs ...Result) Result {
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

	return &joinedResults{
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
		l := len(x.Output())
		if !x.Constant(grad) {
			x.PropagateGradient(upstream[vecIdx:vecIdx+l], grad)
		}
		vecIdx += l
	}
}

type joinedRResults struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Results    []RResult
}

// ConcatR is like Concat, but for RResults.
func ConcatR(rs ...RResult) RResult {
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

	return &joinedRResults{
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
		l := len(x.Output())
		if !x.Constant(rgrad, grad) {
			x.PropagateRGradient(upstream[vecIdx:vecIdx+l], upstreamR[vecIdx:vecIdx+l],
				rgrad, grad)
		}
		vecIdx += l
	}
}

type slicedResult struct {
	Input    Result
	StartIdx int
	EndIdx   int
}

// Slice generates a Result which contains the
// sub-range of input.Output() between the start
// index (inclusive) and end index (exclusive).
func Slice(in Result, start, end int) Result {
	return &slicedResult{
		Input:    in,
		StartIdx: start,
		EndIdx:   end,
	}
}

func (s *slicedResult) Output() linalg.Vector {
	return s.Input.Output()[s.StartIdx:s.EndIdx]
}

func (s *slicedResult) Constant(g Gradient) bool {
	return s.Input.Constant(g)
}

func (s *slicedResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if !s.Input.Constant(grad) {
		if variable, ok := s.Input.(*Variable); ok {
			sumGradVec := grad[variable]
			sumGradVec[s.StartIdx:s.EndIdx].Add(upstream)
		} else {
			downstream := make(linalg.Vector, len(s.Input.Output()))
			copy(downstream[s.StartIdx:], upstream)
			s.Input.PropagateGradient(downstream, grad)
		}
	}
}

type slicedRResult struct {
	Input    RResult
	StartIdx int
	EndIdx   int
}

// SliceR is like Slice, but for RResults.
func SliceR(in RResult, start, end int) RResult {
	return &slicedRResult{
		Input:    in,
		StartIdx: start,
		EndIdx:   end,
	}
}

func (s *slicedRResult) Output() linalg.Vector {
	return s.Input.Output()[s.StartIdx:s.EndIdx]
}

func (s *slicedRResult) ROutput() linalg.Vector {
	return s.Input.ROutput()[s.StartIdx:s.EndIdx]
}

func (s *slicedRResult) Constant(rg RGradient, g Gradient) bool {
	return s.Input.Constant(rg, g)
}

func (s *slicedRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	if !s.Input.Constant(rgrad, grad) {
		if rVariable, ok := s.Input.(*RVariable); ok {
			gradVec := grad[rVariable.Variable]
			if gradVec != nil {
				gradVec[s.StartIdx:s.EndIdx].Add(upstream)
			}
			rgradVec := rgrad[rVariable.Variable]
			if rgradVec != nil {
				rgradVec[s.StartIdx:s.EndIdx].Add(upstreamR)
			}
		} else {
			downstream := make(linalg.Vector, len(s.Input.Output()))
			downstreamR := make(linalg.Vector, len(s.Input.Output()))
			copy(downstream[s.StartIdx:], upstream)
			copy(downstreamR[s.StartIdx:], upstreamR)
			s.Input.PropagateRGradient(downstream, downstreamR, rgrad, grad)
		}
	}
}

type repeatResult struct {
	OutputVec linalg.Vector
	Repeated  Result
	N         int
}

// Repeat concatenates a Result with itself n times.
// This may be more efficient than using Concat.
func Repeat(in Result, n int) Result {
	inVec := in.Output()
	outVec := make(linalg.Vector, len(inVec)*n)
	for i := 0; i < n; i++ {
		copy(outVec[i*len(inVec):], inVec)
	}
	return &repeatResult{
		OutputVec: outVec,
		Repeated:  in,
		N:         n,
	}
}

func (r *repeatResult) Output() linalg.Vector {
	return r.OutputVec
}

func (r *repeatResult) Constant(g Gradient) bool {
	return r.Repeated.Constant(g)
}

func (r *repeatResult) PropagateGradient(upstream linalg.Vector, g Gradient) {
	if r.Repeated.Constant(g) {
		return
	}
	if len(upstream) != len(r.OutputVec) {
		panic("invalid upstream size")
	}
	partLen := len(r.Repeated.Output())
	firstPart := upstream[:partLen]
	for i := 1; i < r.N; i++ {
		partVec := upstream[i*partLen : (i+1)*partLen]
		firstPart.Add(partVec)
	}
	r.Repeated.PropagateGradient(firstPart, g)
}

type repeatRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Repeated   RResult
	N          int
}

// RepeatR is like Repeat, but for RResults.
func RepeatR(in RResult, n int) RResult {
	inVec := in.Output()
	inVecR := in.ROutput()
	outVec := make(linalg.Vector, len(inVec)*n)
	outVecR := make(linalg.Vector, len(inVec)*n)
	for i := 0; i < n; i++ {
		copy(outVec[i*len(inVec):], inVec)
		copy(outVecR[i*len(inVec):], inVecR)
	}
	return &repeatRResult{
		OutputVec:  outVec,
		ROutputVec: outVecR,
		Repeated:   in,
		N:          n,
	}
}

func (r *repeatRResult) Output() linalg.Vector {
	return r.OutputVec
}

func (r *repeatRResult) ROutput() linalg.Vector {
	return r.ROutputVec
}

func (r *repeatRResult) Constant(rg RGradient, g Gradient) bool {
	return r.Repeated.Constant(rg, g)
}

func (r *repeatRResult) PropagateRGradient(upstream, upstreamR linalg.Vector, rg RGradient,
	g Gradient) {
	if r.Repeated.Constant(rg, g) {
		return
	}
	if len(upstream) != len(r.OutputVec) {
		panic("invalid upstream size")
	}
	partLen := len(r.Repeated.Output())
	firstPart := upstream[:partLen]
	firstPartR := upstreamR[:partLen]
	for i := 1; i < r.N; i++ {
		partVec := upstream[i*partLen : (i+1)*partLen]
		partVecR := upstreamR[i*partLen : (i+1)*partLen]
		firstPart.Add(partVec)
		firstPartR.Add(partVecR)
	}
	r.Repeated.PropagateRGradient(firstPart, firstPartR, rg, g)
}

// Split slices a Result into n even sub-slices.
// The length of the input must be divisible by n.
func Split(n int, in Result) []Result {
	if len(in.Output())%n != 0 {
		panic("count does not divide input length")
	}
	parts := make([]Result, n)
	partSize := len(in.Output()) / n
	for i := 0; i < n; i++ {
		parts[i] = Slice(in, i*partSize, (i+1)*partSize)
	}
	return parts
}

// SplitR is like Split for RResults.
func SplitR(n int, in RResult) []RResult {
	if len(in.Output())%n != 0 {
		panic("count does not divide input length")
	}
	parts := make([]RResult, n)
	partSize := len(in.Output()) / n
	for i := 0; i < n; i++ {
		parts[i] = SliceR(in, i*partSize, (i+1)*partSize)
	}
	return parts
}
