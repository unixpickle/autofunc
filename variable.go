package autofunc

import "github.com/unixpickle/num-analysis/linalg"

// A Variable is a numerical vector, wrapped in
// a struct so pointers to it can be used as a
// map key in things like Gradient.
type Variable struct {
	Vector linalg.Vector
}

func (v *Variable) Output() linalg.Vector {
	return v.Vector
}

func (v *Variable) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if gradVec, ok := grad[v]; ok {
		gradVec.Add(upstream)
	}
}

func (v *Variable) Constant(g Gradient) bool {
	_, variable := g[v]
	return !variable
}

// Release does absolutely nothing.
func (v *Variable) Release() {
}

// An RVariable is a variable that knows about
// a particular RVector and can thus behave
// like an RResult.
type RVariable struct {
	Variable *Variable

	ROutputVec linalg.Vector

	VecWasAllocated bool
	VecCache        *VectorCache
}

func NewRVariable(v *Variable, rv RVector) *RVariable {
	return NewRVariableCache(v, rv, nil)
}

func NewRVariableCache(v *Variable, rv RVector, c *VectorCache) *RVariable {
	if vec, ok := rv[v]; ok {
		return &RVariable{
			Variable:   v,
			ROutputVec: vec,
		}
	} else {
		outputDeriv := c.Alloc(len(v.Vector))
		return &RVariable{
			Variable:        v,
			ROutputVec:      outputDeriv,
			VecWasAllocated: true,
			VecCache:        c,
		}
	}
}

func (r *RVariable) Output() linalg.Vector {
	return r.Variable.Output()
}

func (r *RVariable) ROutput() linalg.Vector {
	return r.ROutputVec
}

func (r *RVariable) Constant(rg RGradient, g Gradient) bool {
	if _, ok := rg[r.Variable]; ok {
		return false
	}
	if g == nil {
		return true
	}
	_, variable := g[r.Variable]
	return !variable
}

func (r *RVariable) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	if grad != nil {
		if gradVec, ok := grad[r.Variable]; ok {
			gradVec.Add(upstream)
		}
	}
	if gradVec, ok := rgrad[r.Variable]; ok {
		gradVec.Add(upstreamR)
	}
}

// Release releases the ROutput if it was allocated to
// be all zeroes.
func (r *RVariable) Release() {
	if r.VecWasAllocated {
		r.VecCache.Free(r.ROutputVec)
		r.ROutputVec = nil
	}
}
