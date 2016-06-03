package autofunc

import "github.com/unixpickle/num-analysis/linalg"

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

// An RVariable is a variable that knows about
// a particular RVector and can thus behave
// like an RResult.
type RVariable struct {
	Variable   *Variable
	ROutputVec linalg.Vector
}

func NewRVariable(v *Variable, rv RVector) *RVariable {
	if vec, ok := rv[v]; ok {
		return &RVariable{
			Variable:   v,
			ROutputVec: vec,
		}
	} else {
		outputDeriv := make(linalg.Vector, len(v.Vector))
		return &RVariable{
			Variable:   v,
			ROutputVec: outputDeriv,
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
