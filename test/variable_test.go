package autofunc

import (
	"math/rand"
	"testing"

	. "github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
)

func TestVariableSerialize(t *testing.T) {
	v := &Variable{Vector: make(linalg.Vector, 17)}
	for i := range v.Vector {
		v.Vector[i] = rand.NormFloat64()
	}
	data, err := serializer.SerializeWithType(v)
	if err != nil {
		t.Fatal(err)
	}
	obj, err := serializer.DeserializeWithType(data)
	if err != nil {
		t.Fatal(err)
	}
	newV := obj.(*Variable)
	if len(newV.Vector) != len(v.Vector) {
		t.Fatal("lengths don't match")
	}
	if newV.Vector.Copy().Scale(-1).Add(v.Vector).MaxAbs() > 1e-8 {
		t.Fatalf("expected %v got %v", v, newV)
	}
}
