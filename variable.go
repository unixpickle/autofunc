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
