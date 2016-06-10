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

	Cache *VectorCache
}

// Apply performs matrix multiplication (i.e. m*in).
func (l *LinTran) Apply(in Result) Result {
	if len(in.Output()) != l.Cols {
		panic("input length is invalid")
	}
	return &linTranResult{
		Matrix:    l,
		Input:     in,
		OutputVec: l.multiply(in.Output()),
	}
}

// ApplyR is like Apply but for RResults.
func (l *LinTran) ApplyR(v RVector, in RResult) RResult {
	if len(in.Output()) != l.Cols {
		panic("input length is invalid")
	}
	rData := NewRVariableCache(l.Data, v, l.Cache)
	return &linTranRResult{
		Matrix:     l,
		OutputVec:  l.multiply(in.Output()),
		ROutputVec: l.multiplyR(rData, in),
		Input:      in,
		RData:      rData,
	}
}

// Batch performs matrix multiplication on all
// of the input vectors.
func (l *LinTran) Batch(in Result, n int) Result {
	b := FuncBatcher{
		F:     l,
		Cache: l.Cache,
	}
	return b.Batch(in, n)
}

// BatchR performs matrix multiplication on all
// of the input vectors.
func (l *LinTran) BatchR(v RVector, in RResult, n int) RResult {
	b := RFuncBatcher{
		F:     l,
		Cache: l.Cache,
	}
	return b.BatchR(v, in, n)
}

func (l *LinTran) multiply(vec linalg.Vector) linalg.Vector {
	res := l.Cache.Alloc(l.Rows)

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
	res := l.Cache.Alloc(l.Rows)
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
	gradVal := l.Cache.Alloc(l.Cols)

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

// linTranResult represents the result of applying
// a LinTran to a Result.
type linTranResult struct {
	OutputVec linalg.Vector
	Input     Result
	Matrix    *LinTran
}

func (l *linTranResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *linTranResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if len(upstream) != l.Matrix.Rows {
		panic("output dimension mismatch")
	}

	if !l.Matrix.Data.Constant(grad) {
		l.Matrix.dataGradient(upstream, grad, l.Input.Output())
	}

	if !l.Input.Constant(grad) {
		gradVal := l.Matrix.inputGradient(upstream)
		l.Input.PropagateGradient(gradVal, grad)
		l.Matrix.Cache.Free(gradVal)
	}
}

func (l *linTranResult) Constant(g Gradient) bool {
	return l.Matrix.Data.Constant(g) && l.Input.Constant(g)
}

func (l *linTranResult) Release() {
	l.Matrix.Cache.Free(l.OutputVec)
	l.OutputVec = nil
	l.Input.Release()
}

type linTranRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      RResult
	RData      *RVariable
	Matrix     *LinTran
}

func (l *linTranRResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *linTranRResult) ROutput() linalg.Vector {
	return l.ROutputVec
}

func (l *linTranRResult) Constant(rg RGradient, g Gradient) bool {
	return l.Input.Constant(rg, g) && l.RData.Constant(rg, g)
}

func (l *linTranRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
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

		downstreamRVec := l.Matrix.Cache.Alloc(l.Matrix.Cols)
		for row, partial := range upstream {
			partialR := upstreamR[row]
			for col := 0; col < l.Matrix.Cols; col++ {
				downstreamRVec[col] += partialR*data[dataIdx] + partial*dataDerivs[dataIdx]
				dataIdx++
			}
		}
		downstreamVec := l.Matrix.inputGradient(upstream)

		l.Input.PropagateRGradient(downstreamVec, downstreamRVec, rgrad, grad)

		l.Matrix.Cache.Free(downstreamRVec)
		l.Matrix.Cache.Free(downstreamVec)
	}
}

func (l *linTranRResult) Release() {
	l.Matrix.Cache.Free(l.OutputVec)
	l.Matrix.Cache.Free(l.ROutputVec)
	l.OutputVec = nil
	l.ROutputVec = nil
	l.Input.Release()
	l.RData.Release()
}
