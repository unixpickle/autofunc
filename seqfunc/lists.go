package seqfunc

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

type sliceListResult struct {
	Res   Result
	Start int
	End   int
}

// SliceList extracts a sub-range of the sequences in a
// sequence list.
func SliceList(r Result, start, end int) Result {
	if start < 0 || start > end || end > len(r.OutputSeqs()) {
		panic("bad slice bounds")
	}
	return &sliceListResult{
		Res:   r,
		Start: start,
		End:   end,
	}
}

func (s *sliceListResult) OutputSeqs() [][]linalg.Vector {
	return s.Res.OutputSeqs()[s.Start:s.End]
}

func (s *sliceListResult) PropagateGradient(u [][]linalg.Vector, g autofunc.Gradient) {
	if vr, ok := s.Res.(*varResult); ok {
		for i, seq := range u {
			for j, vec := range seq {
				if gradVec := g[vr.Vars[i+s.Start][j]]; gradVec != nil {
					gradVec.Add(vec)
				}
			}
		}
		return
	}
	fullUpstream := make([][]linalg.Vector, len(s.Res.OutputSeqs()))
	for i, seq := range s.Res.OutputSeqs() {
		if i < s.Start || i >= s.End {
			fullUpstream[i] = make([]linalg.Vector, len(seq))
			for j, v := range seq {
				fullUpstream[i][j] = make(linalg.Vector, len(v))
			}
		} else {
			fullUpstream[i] = u[i-s.Start]
		}
	}
	s.Res.PropagateGradient(fullUpstream, g)
}

type sliceListRResult struct {
	Res   RResult
	Start int
	End   int
}

// SliceListR extracts a sub-range of the sequences in a
// sequence list.
func SliceListR(r RResult, start, end int) RResult {
	if start < 0 || start > end || end > len(r.OutputSeqs()) {
		panic("bad slice bounds")
	}
	return &sliceListRResult{
		Res:   r,
		Start: start,
		End:   end,
	}
}

func (s *sliceListRResult) OutputSeqs() [][]linalg.Vector {
	return s.Res.OutputSeqs()[s.Start:s.End]
}

func (s *sliceListRResult) ROutputSeqs() [][]linalg.Vector {
	return s.Res.ROutputSeqs()[s.Start:s.End]
}

func (s *sliceListRResult) PropagateRGradient(u, uR [][]linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	if vr, ok := s.Res.(*varRResult); ok {
		for i, seq := range u {
			if g != nil {
				for j, vec := range seq {
					if gradVec := g[vr.RVars[i+s.Start][j].Variable]; gradVec != nil {
						gradVec.Add(vec)
					}
				}
			}
		}
		for i, seqR := range uR {
			for j, vecR := range seqR {
				if rgradVec := rg[vr.RVars[i+s.Start][j].Variable]; rgradVec != nil {
					rgradVec.Add(vecR)
				}
			}
		}
		return
	}
	fullUpstream := make([][]linalg.Vector, len(s.Res.OutputSeqs()))
	fullUpstreamR := make([][]linalg.Vector, len(s.Res.OutputSeqs()))
	for i, seq := range s.Res.OutputSeqs() {
		if i < s.Start || i >= s.End {
			fullUpstream[i] = make([]linalg.Vector, len(seq))
			fullUpstreamR[i] = make([]linalg.Vector, len(seq))
			for j, v := range seq {
				fullUpstream[i][j] = make(linalg.Vector, len(v))
				fullUpstreamR[i][j] = make(linalg.Vector, len(v))
			}
		} else {
			fullUpstream[i] = u[i-s.Start]
			fullUpstreamR[i] = uR[i-s.Start]
		}
	}
	s.Res.PropagateRGradient(fullUpstream, fullUpstreamR, rg, g)
}
