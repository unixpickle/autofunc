package autofunc

import (
	"github.com/gonum/blas/blas64"
	"github.com/unixpickle/num-analysis/linalg"
)

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

// Add adds all the values from g1 to g.
// The gradients should have the exact same keys.
func (g Gradient) Add(g1 Gradient) {
	addVariableMaps(g, g1)
}

// Scale scales all the partials in g by f.
func (g Gradient) Scale(f float64) {
	scaleVariableMap(g, f)
}

// AddToVars performs gradient ascent, adding the
// values from the gradient to their corresponding
// variables.
func (g Gradient) AddToVars(scale float64) {
	for variable, grad := range g {
		v1 := blas64.Vector{
			Data: variable.Vector,
			Inc:  1,
		}
		v2 := blas64.Vector{
			Data: grad,
			Inc:  1,
		}
		blas64.Axpy(len(grad), scale, v2, v1)
	}
}

// Copy creates a copy of a Gradient.
func (g Gradient) Copy() Gradient {
	return copyVariableMap(g)
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

// Add adds all the values from g1 to g.
// The RGradients should have the exact same keys.
func (g RGradient) Add(g1 RGradient) {
	addVariableMaps(g, g1)
}

// Scale scales all the partials in g by f.
func (g RGradient) Scale(f float64) {
	scaleVariableMap(g, f)
}

// Copy creates a copy of an RGradient.
func (g RGradient) Copy() RGradient {
	return copyVariableMap(g)
}

// An RVector specifies how much each variable
// changes with respect to a variable r.
// This is used for operating on RResults and
// creating RVariables.
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

func addVariableMaps(m, m1 map[*Variable]linalg.Vector) {
	for k, v := range m {
		vVector := blas64.Vector{
			Data: v,
			Inc:  1,
		}
		v1Vector := blas64.Vector{
			Data: m1[k],
			Inc:  1,
		}
		blas64.Axpy(len(v), 1, v1Vector, vVector)
	}
}

func scaleVariableMap(m map[*Variable]linalg.Vector, f float64) {
	for _, v := range m {
		v.Scale(f)
	}
}

func copyVariableMap(m map[*Variable]linalg.Vector) map[*Variable]linalg.Vector {
	res := map[*Variable]linalg.Vector{}
	for k, v := range m {
		newV := make(linalg.Vector, len(v))
		copy(newV, v)
		res[k] = newV
	}
	return res
}
