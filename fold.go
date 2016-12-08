package autofunc

import "github.com/unixpickle/num-analysis/linalg"

type foldResult struct {
	Pool         []*Variable
	Intermediate []Result
	Final        Result
}

// Fold applies a fold operation to the list of inputs.
// For each timestep i, the previous state and input i are
// fed into f, producing a new state for time i+1.
// The final state is returned.
//
// This may employ pooling between timesteps, so that the
// input to f() might be a different underlying object
// than the previous output of f().
func Fold(state Result, ins []Result, f func(state, in Result) Result) Result {
	res := &foldResult{}
	for _, in := range ins {
		res.Intermediate = append(res.Intermediate, state)
		pool := &Variable{Vector: state.Output()}
		res.Pool = append(res.Pool, pool)
		state = f(pool, in)
	}
	res.Final = state
	return res
}

func (f *foldResult) Output() linalg.Vector {
	return f.Final.Output()
}

func (f *foldResult) Constant(g Gradient) bool {
	if !f.Final.Constant(g) {
		return false
	}
	for i := len(f.Pool) - 1; i >= 0; i-- {
		if !f.Intermediate[i].Constant(g) {
			return false
		}
		if i > 0 {
			p := f.Pool[i-1]
			g[p] = make(linalg.Vector, len(p.Vector))
			c := f.Intermediate[i].Constant(g)
			delete(g, p)
			if c {
				return true
			}
		}
	}
	return true
}

func (f *foldResult) PropagateGradient(u linalg.Vector, g Gradient) {
	if len(f.Pool) == 0 {
		f.Final.PropagateGradient(u, g)
		return
	}

	stateUp := u
	for i := len(f.Pool); i > 0; i-- {
		var res Result
		if i == len(f.Pool) {
			res = f.Final
		} else {
			res = f.Intermediate[i]
		}
		p := f.Pool[i-1]
		g[p] = make(linalg.Vector, len(p.Vector))
		if res.Constant(g) {
			delete(g, p)
			break
		}
		res.PropagateGradient(stateUp, g)
		stateUp = g[p]
		delete(g, p)
	}
	if len(f.Intermediate) > 0 {
		f.Intermediate[0].PropagateGradient(stateUp, g)
	}
}

type foldRResult struct {
	Pool         []*Variable
	Intermediate []RResult
	Final        RResult
}

// FoldR is like Fold, but for RResults.
func FoldR(state RResult, ins []RResult, f func(state, in RResult) RResult) RResult {
	res := &foldRResult{}
	for _, in := range ins {
		res.Intermediate = append(res.Intermediate, state)
		pool := &Variable{Vector: state.Output()}
		res.Pool = append(res.Pool, pool)
		state = f(&RVariable{
			Variable:   pool,
			ROutputVec: state.ROutput(),
		}, in)
	}
	res.Final = state
	return res
}

func (f *foldRResult) Output() linalg.Vector {
	return f.Final.Output()
}

func (f *foldRResult) ROutput() linalg.Vector {
	return f.Final.ROutput()
}

func (f *foldRResult) Constant(rg RGradient, g Gradient) bool {
	if !f.Final.Constant(rg, g) {
		return false
	}
	for i := len(f.Pool) - 1; i >= 0; i-- {
		if !f.Intermediate[i].Constant(rg, g) {
			return false
		}
		if i > 0 {
			p := f.Pool[i-1]
			if g != nil {
				g[p] = make(linalg.Vector, len(p.Vector))
			}
			rg[p] = make(linalg.Vector, len(p.Vector))
			c := f.Intermediate[i].Constant(rg, g)
			if g != nil {
				delete(g, p)
			}
			delete(rg, p)
			if c {
				return true
			}
		}
	}
	return true
}

func (f *foldRResult) PropagateRGradient(u, uR linalg.Vector, rg RGradient, g Gradient) {
	if len(f.Pool) == 0 {
		f.Final.PropagateRGradient(u, uR, rg, g)
		return
	}

	if g == nil {
		g = Gradient{}
	}

	stateUp := u
	stateUpR := uR
	for i := len(f.Pool); i > 0; i-- {
		var res RResult
		if i == len(f.Pool) {
			res = f.Final
		} else {
			res = f.Intermediate[i]
		}
		p := f.Pool[i-1]
		g[p] = make(linalg.Vector, len(p.Vector))
		rg[p] = make(linalg.Vector, len(p.Vector))
		if res.Constant(rg, g) {
			delete(g, p)
			delete(rg, p)
			break
		}
		res.PropagateRGradient(stateUp, stateUpR, rg, g)
		stateUp = g[p]
		stateUpR = rg[p]
		delete(g, p)
		delete(rg, p)
	}
	if len(f.Intermediate) > 0 {
		f.Intermediate[0].PropagateRGradient(stateUp, stateUpR, rg, g)
	}
}
