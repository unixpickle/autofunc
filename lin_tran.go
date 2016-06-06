package autofunc

import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"github.com/unixpickle/num-analysis/linalg"
)

// A LinTran is a Func and RFunc that represents
// a linear transformation.
// A linear transformation is represented by a
// set of dimensions (rows by columns) and a
// vector of entries in the matrix (going left
// to right, then top to bottom).
type LinTran struct {
	Data *Variable
	Rows int
	Cols int
}

// Apply performs matrix multiplication (i.e. m*in)
// and returns the result as a *LinTranResult.
func (l *LinTran) Apply(in Result) Result {
	return &LinTranResult{
		Matrix:    l,
		Input:     in,
		OutputVec: l.multiply(in.Output()),
	}
}

// ApplyR is like Apply, but generates a
// *LinTranRResult to reflect r-operator information.
func (l *LinTran) ApplyR(v RVector, in RResult) RResult {
	rData := NewRVariable(l.Data, v)
	return &LinTranRResult{
		Matrix:     l,
		OutputVec:  l.multiply(in.Output()),
		ROutputVec: l.multiplyR(rData, in),
		Input:      in,
		RData:      rData,
	}
}

func (l *LinTran) multiply(vec linalg.Vector) linalg.Vector {
	res := make(linalg.Vector, l.Rows)

	mat := blas64.General{
		Rows:   l.Rows,
		Cols:   l.Cols,
		Stride: l.Cols,
		Data:   l.Data.Vector,
	}
	inVec := blas64.Vector{
		Inc:  1,
		Data: vec,
	}
	outVec := blas64.Vector{
		Inc:  1,
		Data: res,
	}
	blas64.Gemv(blas.NoTrans, 1, mat, inVec, 0, outVec)

	return res
}

func (l *LinTran) multiplyR(rData *RVariable, rVec RResult) linalg.Vector {
	res := make(linalg.Vector, l.Rows)
	matData := rData.Output()
	matDerivs := rData.ROutput()
	inVec := rVec.Output()
	inDerivs := rVec.ROutput()

	matIdx := 0
	for i := range res {
		for j, vecVal := range inVec {
			vecDeriv := inDerivs[j]
			res[i] += matData[matIdx]*vecDeriv + matDerivs[matIdx]*vecVal
			matIdx++
		}
	}
	return res
}

func (l *LinTran) dataGradient(upstream linalg.Vector, grad Gradient, input linalg.Vector) {
	gradMat := blas64.General{
		Data:   grad[l.Data],
		Rows:   l.Rows,
		Cols:   l.Cols,
		Stride: l.Cols,
	}
	partialVec := blas64.Vector{
		Data: upstream,
		Inc:  1,
	}
	inputVec := blas64.Vector{
		Data: input,
		Inc:  1,
	}
	blas64.Ger(1, partialVec, inputVec, gradMat)
}

func (l *LinTran) inputGradient(upstream linalg.Vector) linalg.Vector {
	matData := l.Data.Output()
	gradVal := make(linalg.Vector, l.Cols)

	matrix := blas64.General{
		Rows:   l.Rows,
		Cols:   l.Cols,
		Stride: l.Cols,
		Data:   matData,
	}
	vectorIn := blas64.Vector{Data: upstream, Inc: 1}
	vectorOut := blas64.Vector{Data: gradVal, Inc: 1}
	blas64.Gemv(blas.Trans, 1, matrix, vectorIn, 0, vectorOut)

	return gradVal
}

// LinTranResult represents the result of applying
// a LinTran to a Result.
type LinTranResult struct {
	OutputVec linalg.Vector
	Input     Result
	Matrix    *LinTran
}

func (l *LinTranResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *LinTranResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if len(upstream) != l.Matrix.Rows {
		panic("output dimension mismatch")
	}

	if !l.Matrix.Data.Constant(grad) {
		l.Matrix.dataGradient(upstream, grad, l.Input.Output())
	}

	if !l.Input.Constant(grad) {
		gradVal := l.Matrix.inputGradient(upstream)
		l.Input.PropagateGradient(gradVal, grad)
	}
}

func (l *LinTranResult) Constant(g Gradient) bool {
	return l.Matrix.Data.Constant(g) && l.Input.Constant(g)
}

type LinTranRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      RResult
	RData      *RVariable
	Matrix     *LinTran
}

func (l *LinTranRResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *LinTranRResult) ROutput() linalg.Vector {
	return l.ROutputVec
}

func (l *LinTranRResult) Constant(rg RGradient, g Gradient) bool {
	return l.Input.Constant(rg, g) && l.RData.Constant(rg, g)
}

func (l *LinTranRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	if grad != nil && !l.Matrix.Data.Constant(grad) {
		l.Matrix.dataGradient(upstream, grad, l.Input.Output())
	}

	if outGrad, ok := rgrad[l.Matrix.Data]; ok {
		inputDerivs := l.Input.ROutput()
		input := l.Input.Output()

		outGradIdx := 0
		for row, partial := range upstream {
			partialR := upstreamR[row]
			for col := 0; col < l.Matrix.Cols; col++ {
				outGrad[outGradIdx] += input[col]*partialR + inputDerivs[col]*partial
				outGradIdx++
			}
		}
	}

	if !l.Input.Constant(rgrad, grad) {
		dataDerivs := l.RData.ROutput()
		data := l.RData.Output()
		dataIdx := 0

		downstreamRVec := make(linalg.Vector, l.Matrix.Cols)
		for row, partial := range upstream {
			partialR := upstreamR[row]
			for col := 0; col < l.Matrix.Cols; col++ {
				downstreamRVec[col] += partialR*data[dataIdx] + partial*dataDerivs[dataIdx]
				dataIdx++
			}
		}

		l.Input.PropagateRGradient(l.Matrix.inputGradient(upstream),
			downstreamRVec, rgrad, grad)
	}
}
