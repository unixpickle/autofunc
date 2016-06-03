package autofunc

import "github.com/unixpickle/num-analysis/linalg"

// A Gradient stores the rates of change of a
// numerical quantity with respect to components
// of various variables.
type Gradient map[*Variable]linalg.Vector

// An RGradient is like a Gradient, but its entries
// correspond to the derivatives of the components
// of the gradient with respect to a variable r.
type RGradient map[*Variable]linalg.Vector
