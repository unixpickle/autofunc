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

// Zero resets all the values of the gradient to 0.
func (g Gradient) Zero() {
	zeroVariableMap(g)
}

// An RGradient is like a Gradient, but its entries
// correspond to the derivatives of the components
// of the gradient with respect to a variable r.
type RGradient map[*Variable]linalg.Vector

func NewRGradient(vars []*Variable) RGradient {
	return RGradient(NewGradient(vars))
}

// Zero resets all the values of the RGradient to 0.
func (g RGradient) Zero() {
	zeroVariableMap(g)
}

// An RVector specifies how much each variable
// changes with respect to r for a given r-operator
// propagation.
type RVector map[*Variable]linalg.Vector

// Zero resets all the values of the RVector to 0.
func (r RVector) Zero() {
	zeroVariableMap(r)
}

func zeroVariableMap(m map[*Variable]linalg.Vector) {
	for _, v := range m {
		for i := range v {
			v[i] = 0
		}
	}
}
