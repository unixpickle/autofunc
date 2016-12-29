package autofunc

import (
	"bytes"
	"encoding/binary"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
)

func init() {
	var v Variable
	serializer.RegisterTypedDeserializer(v.SerializerType(), DeserializeVariable)
}

// A Variable is a numerical vector, wrapped in
// a struct so pointers to it can be used as a
// map key in things like Gradient.
type Variable struct {
	Vector linalg.Vector
}

// DeserializeVariable deserializes a Variable.
func DeserializeVariable(d []byte) (*Variable, error) {
	reader := bytes.NewBuffer(d)
	var size uint64
	if err := binary.Read(reader, binary.LittleEndian, &size); err != nil {
		return nil, err
	}
	vec := make(linalg.Vector, int(size))
	for i := range vec {
		if err := binary.Read(reader, binary.LittleEndian, &vec[i]); err != nil {
			return nil, err
		}
	}
	return &Variable{Vector: vec}, nil
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

// SerializerType returns the unique ID used to serialize
// a Variable using the serializer package.
func (v *Variable) SerializerType() string {
	return "github.com/unixpickle/autofunc.Variable"
}

// Serialize serializes the variable.
func (v *Variable) Serialize() ([]byte, error) {
	var w bytes.Buffer
	binary.Write(&w, binary.LittleEndian, uint64(len(v.Vector)))
	for _, x := range v.Vector {
		binary.Write(&w, binary.LittleEndian, x)
	}
	return w.Bytes(), nil
}

// An RVariable is a variable that knows about
// a particular RVector and can thus behave
// like an RResult.
type RVariable struct {
	Variable *Variable

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
