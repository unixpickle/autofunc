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
	return l.Batch(in, 1)
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
	return &linTranResult{
		Matrix:    l,
		Input:     in,
		OutputVec: l.multiply(in.Output()),
	}
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
	n := len(vec) / l.Cols
	res := l.Cache.Alloc(l.Rows * n)

	mat := blas64.General{
		Rows:   l.Rows,
		Cols:   l.Cols,
		Stride: l.Cols,
		Data:   l.Data.Vector,
	}
	inMat := blas64.General{
		Rows:   n,
		Cols:   l.Cols,
		Stride: l.Cols,
		Data:   vec,
	}
	outMat := blas64.General{
		Rows:   n,
		Cols:   l.Rows,
		Stride: l.Rows,
		Data:   res,
	}
	blas64.Gemm(blas.NoTrans, blas.Trans, 1, inMat, mat, 0, outMat)

	return res
}

func (l *LinTran) multiplyR(rData *RVariable, rVec RResult) linalg.Vector {
	vec := rVec.Output()
	vecR := rVec.ROutput()

	n := len(vec) / l.Cols
	res := l.Cache.Alloc(l.Rows * n)

	mat := blas64.General{
		Rows:   l.Rows,
		Cols:   l.Cols,
		Stride: l.Cols,
		Data:   rData.Output(),
	}
	matR := blas64.General{
		Rows:   l.Rows,
		Cols:   l.Cols,
		Stride: l.Cols,
		Data:   rData.ROutput(),
	}
	inMat := blas64.General{
		Rows:   n,
		Cols:   l.Cols,
		Stride: l.Cols,
		Data:   vec,
	}
	inMatR := blas64.General{
		Rows:   n,
		Cols:   l.Cols,
		Stride: l.Cols,
		Data:   vecR,
	}
	outMat := blas64.General{
		Rows:   n,
		Cols:   l.Rows,
		Stride: l.Rows,
		Data:   res,
	}
	blas64.Gemm(blas.NoTrans, blas.Trans, 1, inMat, matR, 0, outMat)
	blas64.Gemm(blas.NoTrans, blas.Trans, 1, inMatR, mat, 1, outMat)

	return res
}

func (l *LinTran) dataGradient(upstream linalg.Vector, input linalg.Vector, grad Gradient) {
	n := len(input) / l.Cols
	gradMat := blas64.General{
		Data:   grad[l.Data],
		Rows:   l.Rows,
		Cols:   l.Cols,
		Stride: l.Cols,
	}
	if n == 1 {
		partialVec := blas64.Vector{
			Data: upstream,
			Inc:  1,
		}
		inputVec := blas64.Vector{
			Data: input,
			Inc:  1,
		}
		blas64.Ger(1, partialVec, inputVec, gradMat)
	} else {
		partialMat := blas64.General{
			Rows:   n,
			Cols:   l.Rows,
			Stride: l.Rows,
			Data:   upstream,
		}
		inputMat := blas64.General{
			Rows:   n,
			Cols:   l.Cols,
			Stride: l.Cols,
			Data:   input,
		}
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, partialMat, inputMat, 1, gradMat)
	}
}

func (l *LinTran) dataGradientR(upstream, upstreamR linalg.Vector, input, inputR linalg.Vector,
	rgrad linalg.Vector) {
	n := len(input) / l.Cols
	gradMat := blas64.General{
		Data:   rgrad,
		Rows:   l.Rows,
		Cols:   l.Cols,
		Stride: l.Cols,
	}
	if n == 1 {
		partialVec := blas64.Vector{
			Data: upstream,
			Inc:  1,
		}
		inputVec := blas64.Vector{
			Data: input,
			Inc:  1,
		}
		partialVecR := blas64.Vector{
			Data: upstreamR,
			Inc:  1,
		}
		inputVecR := blas64.Vector{
			Data: inputR,
			Inc:  1,
		}
		blas64.Ger(1, partialVec, inputVecR, gradMat)
		blas64.Ger(1, partialVecR, inputVec, gradMat)
	} else {
		partialMat := blas64.General{
			Rows:   n,
			Cols:   l.Rows,
			Stride: l.Rows,
			Data:   upstream,
		}
		inputMat := blas64.General{
			Rows:   n,
			Cols:   l.Cols,
			Stride: l.Cols,
			Data:   input,
		}
		partialMatR := blas64.General{
			Rows:   n,
			Cols:   l.Rows,
			Stride: l.Rows,
			Data:   upstreamR,
		}
		inputMatR := blas64.General{
			Rows:   n,
			Cols:   l.Cols,
			Stride: l.Cols,
			Data:   inputR,
		}
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, partialMat, inputMatR, 1, gradMat)
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, partialMatR, inputMat, 1, gradMat)
	}
}

func (l *LinTran) inputGradient(upstream linalg.Vector) linalg.Vector {
	n := len(upstream) / l.Rows

	matData := l.Data.Output()
	gradVal := l.Cache.Alloc(l.Cols * n)

	matrix := blas64.General{
		Rows:   l.Rows,
		Cols:   l.Cols,
		Stride: l.Cols,
		Data:   matData,
	}

	if n == 1 {
		vectorIn := blas64.Vector{Data: upstream, Inc: 1}
		vectorOut := blas64.Vector{Data: gradVal, Inc: 1}
		blas64.Gemv(blas.Trans, 1, matrix, vectorIn, 0, vectorOut)
	} else {
		upstreamMat := blas64.General{
			Rows:   n,
			Cols:   l.Rows,
			Stride: l.Rows,
			Data:   upstream,
		}
		outputMat := blas64.General{
			Rows:   n,
			Cols:   l.Cols,
			Stride: l.Cols,
			Data:   gradVal,
		}
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, upstreamMat, matrix, 0, outputMat)
	}

	return gradVal
}

func (l *LinTran) inputGradientR(rData *RVariable, upstream,
	upstreamR linalg.Vector) linalg.Vector {
	n := len(upstream) / l.Rows

	matData := rData.Output()
	matDataR := rData.ROutput()
	gradVal := l.Cache.Alloc(l.Cols * n)

	matrix := blas64.General{
		Rows:   l.Rows,
		Cols:   l.Cols,
		Stride: l.Cols,
		Data:   matData,
	}
	matrixR := blas64.General{
		Rows:   l.Rows,
		Cols:   l.Cols,
		Stride: l.Cols,
		Data:   matDataR,
	}

	if n == 1 {
		vectorIn := blas64.Vector{Data: upstream, Inc: 1}
		vectorInR := blas64.Vector{Data: upstreamR, Inc: 1}
		vectorOut := blas64.Vector{Data: gradVal, Inc: 1}
		blas64.Gemv(blas.Trans, 1, matrixR, vectorIn, 0, vectorOut)
		blas64.Gemv(blas.Trans, 1, matrix, vectorInR, 1, vectorOut)
	} else {
		upstreamMat := blas64.General{
			Rows:   n,
			Cols:   l.Rows,
			Stride: l.Rows,
			Data:   upstream,
		}
		upstreamMatR := blas64.General{
			Rows:   n,
			Cols:   l.Rows,
			Stride: l.Rows,
			Data:   upstreamR,
		}
		outputMat := blas64.General{
			Rows:   n,
			Cols:   l.Cols,
			Stride: l.Cols,
			Data:   gradVal,
		}
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, upstreamMatR, matrix, 0, outputMat)
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, upstreamMat, matrixR, 1, outputMat)
	}

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
	if !l.Matrix.Data.Constant(grad) {
		l.Matrix.dataGradient(upstream, l.Input.Output(), grad)
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
		l.Matrix.dataGradient(upstream, l.Input.Output(), grad)
	}

	if outGrad, ok := rgrad[l.Matrix.Data]; ok {
		input := l.Input.Output()
		inputR := l.Input.ROutput()
		l.Matrix.dataGradientR(upstream, upstreamR, input, inputR, outGrad)
	}

	if !l.Input.Constant(rgrad, grad) {
		downstreamRVec := l.Matrix.inputGradientR(l.RData, upstream, upstreamR)
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
