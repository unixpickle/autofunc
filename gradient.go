package autofunc

import "github.com/unixpickle/num-analysis/linalg"

// A Gradient stores the rates of change of a
// numerical quantity with respect to components
// of various variables.
type Gradient map[*Variable]linalg.Vector

func NewGradient(vars []*Variable) Gradient {
	res := Gradient{}
	for _, v := range vars {
		res[v] = make(linalg.Vector, len(v.Vector))
	}
	return res
}

// An RGradient is like a Gradient, but its entries
// correspond to the derivatives of the components
// of the gradient with respect to a variable r.
type RGradient map[*Variable]linalg.Vector

func NewRGradient(vars []*Variable) RGradient {
	return RGradient(NewGradient(vars))
}

// An RVector specifies how much each variable
// changes with respect to r for a given r-operator
// propagation.
type RVector map[*Variable]linalg.Vector
