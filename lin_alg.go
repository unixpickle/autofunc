package autofunc

import (
	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"github.com/unixpickle/num-analysis/linalg"
)

type matMulResult struct {
	MatIn  Result
	MatVar *Variable
	Res    Result
}

// MatMulVec multiplies a row-major matrix by a column
// vector.
func MatMulVec(mat Result, rows, cols int, vec Result) Result {
	return MatMulVecs(mat, rows, cols, vec)
}

// MatMulVecs multiplies a row-major matrix by a
// column-major matrix, producing another column-major
// matrix.
//
// The number of columns in the right matrix is inferred
// by dividing its length by the number of columns in the
// left matrix.
func MatMulVecs(mat Result, rows, cols int, vecs Result) Result {
	if len(mat.Output()) != rows*cols {
		panic("invalid matrix data size")
	}
	if len(vecs.Output())%cols != 0 {
		panic("invalid vecs size")
	}
	n := len(vecs.Output()) / cols
	v := &Variable{Vector: mat.Output()}
	lt := &LinTran{
		Data: v,
		Rows: rows,
		Cols: cols,
	}
	return &matMulResult{
		MatIn:  mat,
		MatVar: v,
		Res:    lt.Batch(vecs, n),
	}
}

func (m *matMulResult) Output() linalg.Vector {
	return m.Res.Output()
}

func (m *matMulResult) Constant(g Gradient) bool {
	return m.Res.Constant(g) && m.MatIn.Constant(g)
}

func (m *matMulResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	matConst := m.MatIn.Constant(grad)
	if !matConst {
		grad[m.MatVar] = make(linalg.Vector, len(m.MatVar.Vector))
	}
	m.Res.PropagateGradient(upstream, grad)
	if !matConst {
		downstream := grad[m.MatVar]
		delete(grad, m.MatVar)
		m.MatIn.PropagateGradient(downstream, grad)
	}
}

type matMulRResult struct {
	MatIn  RResult
	MatVar *Variable
	Res    RResult
}

// MatMulVecR is like MatMulVec but for RResults.
func MatMulVecR(mat RResult, rows, cols int, vec RResult) RResult {
	return MatMulVecsR(mat, rows, cols, vec)
}

// MatMulVecsR is like MatMulVecs, but for RResults.
func MatMulVecsR(mat RResult, rows, cols int, vecs RResult) RResult {
	if len(mat.Output()) != rows*cols {
		panic("invalid matrix data size")
	}
	if len(vecs.Output())%cols != 0 {
		panic("invalid vecs size")
	}
	n := len(vecs.Output()) / cols
	v := &Variable{Vector: mat.Output()}
	lt := &LinTran{
		Data: v,
		Rows: rows,
		Cols: cols,
	}
	return &matMulRResult{
		MatIn:  mat,
		MatVar: v,
		Res:    lt.BatchR(RVector{v: mat.ROutput()}, vecs, n),
	}
}

func (m *matMulRResult) Output() linalg.Vector {
	return m.Res.Output()
}

func (m *matMulRResult) ROutput() linalg.Vector {
	return m.Res.ROutput()
}

func (m *matMulRResult) Constant(rg RGradient, g Gradient) bool {
	return m.Res.Constant(rg, g) && m.MatIn.Constant(rg, g)
}

func (m *matMulRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	matConst := m.MatIn.Constant(rgrad, grad)
	if !matConst {
		if grad == nil {
			grad = Gradient{}
		}
		grad[m.MatVar] = make(linalg.Vector, len(m.MatVar.Vector))
		rgrad[m.MatVar] = make(linalg.Vector, len(m.MatVar.Vector))
	}
	m.Res.PropagateRGradient(upstream, upstreamR, rgrad, grad)
	if !matConst {
		downstream := grad[m.MatVar]
		downstreamR := rgrad[m.MatVar]
		delete(grad, m.MatVar)
		delete(rgrad, m.MatVar)
		m.MatIn.PropagateRGradient(downstream, downstreamR, rgrad, grad)
	}
}

type outerProductResult struct {
	OutputVec linalg.Vector
	LeftIn    Result
	RightIn   Result
}

// OuterProduct computes the outer product between two
// vectors, expressed as left*transpose(right) where both
// vectors are column vectors.
func OuterProduct(left, right Result) Result {
	outMat := blas64.General{
		Data:   make([]float64, len(left.Output())*len(right.Output())),
		Rows:   len(left.Output()),
		Cols:   len(right.Output()),
		Stride: len(right.Output()),
	}
	leftVec := blas64.Vector{
		Data: left.Output(),
		Inc:  1,
	}
	rightVec := blas64.Vector{
		Data: right.Output(),
		Inc:  1,
	}
	blas64.Ger(1, leftVec, rightVec, outMat)
	return &outerProductResult{
		OutputVec: outMat.Data,
		LeftIn:    left,
		RightIn:   right,
	}
}

func (o *outerProductResult) Output() linalg.Vector {
	return o.OutputVec
}

func (o *outerProductResult) Constant(g Gradient) bool {
	return o.LeftIn.Constant(g) && o.RightIn.Constant(g)
}

func (o *outerProductResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	upstreamMatrix := blas64.General{
		Data:   upstream,
		Rows:   len(o.LeftIn.Output()),
		Cols:   len(o.RightIn.Output()),
		Stride: len(o.RightIn.Output()),
	}
	if !o.LeftIn.Constant(grad) {
		rightVec := blas64.Vector{
			Data: o.RightIn.Output(),
			Inc:  1,
		}
		leftDownstream := blas64.Vector{
			Data: make(linalg.Vector, len(o.LeftIn.Output())),
			Inc:  1,
		}
		blas64.Gemv(blas.NoTrans, 1, upstreamMatrix, rightVec, 0, leftDownstream)
		o.LeftIn.PropagateGradient(leftDownstream.Data, grad)
	}
	if !o.RightIn.Constant(grad) {
		leftVec := blas64.Vector{
			Data: o.LeftIn.Output(),
			Inc:  1,
		}
		rightDownstream := blas64.Vector{
			Data: make(linalg.Vector, len(o.RightIn.Output())),
			Inc:  1,
		}
		blas64.Gemv(blas.Trans, 1, upstreamMatrix, leftVec, 0, rightDownstream)
		o.RightIn.PropagateGradient(rightDownstream.Data, grad)
	}
}

type outerProductRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	LeftIn     RResult
	RightIn    RResult
}

// OuterProductR is like OuterProduct but for RResults.
func OuterProductR(left, right RResult) RResult {
	outMat := blas64.General{
		Data:   make([]float64, len(left.Output())*len(right.Output())),
		Rows:   len(left.Output()),
		Cols:   len(right.Output()),
		Stride: len(right.Output()),
	}
	outMatR := outMat
	outMatR.Data = make([]float64, len(outMat.Data))

	leftVec := blas64.Vector{
		Data: left.Output(),
		Inc:  1,
	}
	leftVecR := blas64.Vector{
		Data: left.ROutput(),
		Inc:  1,
	}
	rightVec := blas64.Vector{
		Data: right.Output(),
		Inc:  1,
	}
	rightVecR := blas64.Vector{
		Data: right.ROutput(),
		Inc:  1,
	}

	blas64.Ger(1, leftVec, rightVec, outMat)
	blas64.Ger(1, leftVecR, rightVec, outMatR)
	blas64.Ger(1, leftVec, rightVecR, outMatR)
	return &outerProductRResult{
		OutputVec:  outMat.Data,
		ROutputVec: outMatR.Data,
		LeftIn:     left,
		RightIn:    right,
	}
}

func (o *outerProductRResult) Output() linalg.Vector {
	return o.OutputVec
}

func (o *outerProductRResult) ROutput() linalg.Vector {
	return o.ROutputVec
}

func (o *outerProductRResult) Constant(rg RGradient, g Gradient) bool {
	return o.LeftIn.Constant(rg, g) && o.RightIn.Constant(rg, g)
}

func (o *outerProductRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	upstreamMatrix := blas64.General{
		Data:   upstream,
		Rows:   len(o.LeftIn.Output()),
		Cols:   len(o.RightIn.Output()),
		Stride: len(o.RightIn.Output()),
	}
	upstreamMatrixR := blas64.General{
		Data:   upstreamR,
		Rows:   len(o.LeftIn.Output()),
		Cols:   len(o.RightIn.Output()),
		Stride: len(o.RightIn.Output()),
	}
	if !o.LeftIn.Constant(rgrad, grad) {
		rightVec := blas64.Vector{
			Data: o.RightIn.Output(),
			Inc:  1,
		}
		rightVecR := blas64.Vector{
			Data: o.RightIn.ROutput(),
			Inc:  1,
		}
		leftDownstream := blas64.Vector{
			Data: make(linalg.Vector, len(o.LeftIn.Output())),
			Inc:  1,
		}
		leftDownstreamR := blas64.Vector{
			Data: make(linalg.Vector, len(o.LeftIn.Output())),
			Inc:  1,
		}
		blas64.Gemv(blas.NoTrans, 1, upstreamMatrix, rightVec, 0, leftDownstream)
		blas64.Gemv(blas.NoTrans, 1, upstreamMatrix, rightVecR, 0, leftDownstreamR)
		blas64.Gemv(blas.NoTrans, 1, upstreamMatrixR, rightVec, 1, leftDownstreamR)
		o.LeftIn.PropagateRGradient(leftDownstream.Data, leftDownstreamR.Data,
			rgrad, grad)
	}
	if !o.RightIn.Constant(rgrad, grad) {
		leftVec := blas64.Vector{
			Data: o.LeftIn.Output(),
			Inc:  1,
		}
		leftVecR := blas64.Vector{
			Data: o.LeftIn.ROutput(),
			Inc:  1,
		}
		rightDownstream := blas64.Vector{
			Data: make(linalg.Vector, len(o.RightIn.Output())),
			Inc:  1,
		}
		rightDownstreamR := blas64.Vector{
			Data: make(linalg.Vector, len(o.RightIn.Output())),
			Inc:  1,
		}
		blas64.Gemv(blas.Trans, 1, upstreamMatrix, leftVec, 0, rightDownstream)
		blas64.Gemv(blas.Trans, 1, upstreamMatrix, leftVecR, 0, rightDownstreamR)
		blas64.Gemv(blas.Trans, 1, upstreamMatrixR, leftVec, 1, rightDownstreamR)
		o.RightIn.PropagateRGradient(rightDownstream.Data, rightDownstreamR.Data,
			rgrad, grad)
	}
}

type transposeResult struct {
	OutputVec linalg.Vector
	In        Result
	InRows    int
	InCols    int
}

// Transpose transposes a row-major matrix.
func Transpose(in Result, rows, cols int) Result {
	return &transposeResult{
		OutputVec: transposeVector(in.Output(), rows, cols),
		In:        in,
		InRows:    rows,
		InCols:    cols,
	}
}

func (t *transposeResult) Output() linalg.Vector {
	return t.OutputVec
}

func (t *transposeResult) Constant(g Gradient) bool {
	return t.In.Constant(g)
}

func (t *transposeResult) PropagateGradient(u linalg.Vector, g Gradient) {
	uTrans := transposeVector(u, t.InCols, t.InRows)
	t.In.PropagateGradient(uTrans, g)
}

type transposeRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	In         RResult
	InRows     int
	InCols     int
}

// TransposeR transposes a row-major matrix.
func TransposeR(in RResult, rows, cols int) RResult {
	return &transposeRResult{
		OutputVec:  transposeVector(in.Output(), rows, cols),
		ROutputVec: transposeVector(in.ROutput(), rows, cols),
		In:         in,
		InRows:     rows,
		InCols:     cols,
	}
}

func (t *transposeRResult) Output() linalg.Vector {
	return t.OutputVec
}

func (t *transposeRResult) ROutput() linalg.Vector {
	return t.ROutputVec
}

func (t *transposeRResult) Constant(rg RGradient, g Gradient) bool {
	return t.In.Constant(rg, g)
}

func (t *transposeRResult) PropagateRGradient(u, uR linalg.Vector, rg RGradient, g Gradient) {
	uTrans := transposeVector(u, t.InCols, t.InRows)
	uTransR := transposeVector(uR, t.InCols, t.InRows)
	t.In.PropagateRGradient(uTrans, uTransR, rg, g)
}

func transposeVector(vec linalg.Vector, rows, cols int) linalg.Vector {
	m := &linalg.Matrix{Data: vec, Rows: rows, Cols: cols}
	return m.Transpose().Data
}

type scaleRowsResult struct {
	Matrix    Result
	Scalers   Result
	OutputVec linalg.Vector
}

// ScaleRows scales each row of a row-major matrix by a
// different scaler.
//
// The number of entries in the matrix must be divisible
// by the number of scalers.
func ScaleRows(mat, scalers Result) Result {
	matData := mat.Output()
	scalerData := scalers.Output()
	if len(matData)%len(scalerData) != 0 {
		panic("scaler count must divide entry count")
	}
	cols := len(matData) / len(scalerData)
	res := make(linalg.Vector, len(matData))
	copy(res, matData)
	for i, scaler := range scalerData {
		dest := res[i*cols : (i+1)*cols]
		blas64.Scal(len(dest), scaler, blas64.Vector{
			Data: dest,
			Inc:  1,
		})
	}
	return &scaleRowsResult{
		Matrix:    mat,
		Scalers:   scalers,
		OutputVec: res,
	}
}

func (s *scaleRowsResult) Output() linalg.Vector {
	return s.OutputVec
}

func (s *scaleRowsResult) Constant(g Gradient) bool {
	return s.Scalers.Constant(g) && s.Matrix.Constant(g)
}

func (s *scaleRowsResult) PropagateGradient(u linalg.Vector, g Gradient) {
	matData := s.Matrix.Output()
	if !s.Scalers.Constant(g) {
		su := make(linalg.Vector, len(s.Scalers.Output()))
		cols := len(matData) / len(su)
		for row := range su {
			matRow := matData[row*cols : (row+1)*cols]
			upstreamRow := u[row*cols : (row+1)*cols]
			su[row] = matRow.DotFast(upstreamRow)
		}
		s.Scalers.PropagateGradient(su, g)
	}
	if !s.Matrix.Constant(g) {
		scalers := s.Scalers.Output()
		cols := len(matData) / len(scalers)
		for row, scaler := range scalers {
			blas64.Scal(cols, scaler, blas64.Vector{
				Data: u[row*cols : (row+1)*cols],
				Inc:  1,
			})
		}
		s.Matrix.PropagateGradient(u, g)
	}
}

type scaleRowsRResult struct {
	Matrix     RResult
	Scalers    RResult
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
}

// ScaleRowsR is like ScaleRows but for RResults.
func ScaleRowsR(mat, scalers RResult) RResult {
	matData := mat.Output()
	matDataR := mat.ROutput()
	scalerData := scalers.Output()
	scalerDataR := scalers.ROutput()
	if len(matData)%len(scalerData) != 0 {
		panic("scaler count must divide entry count")
	}
	cols := len(matData) / len(scalerData)
	res := make(linalg.Vector, len(matData))
	resR := make(linalg.Vector, len(matData))
	copy(res, matData)
	for i, scaler := range scalerData {
		scalerR := scalerDataR[i]
		dest := res[i*cols : (i+1)*cols]
		destR := resR[i*cols : (i+1)*cols]
		source := matData[i*cols : (i+1)*cols]
		sourceR := matDataR[i*cols : (i+1)*cols]
		blas64.Scal(len(dest), scaler, blas64.Vector{
			Data: dest,
			Inc:  1,
		})
		destR.Add(source).Scale(scalerR)
		destR.Add(sourceR.Copy().Scale(scaler))
	}
	return &scaleRowsRResult{
		Matrix:     mat,
		Scalers:    scalers,
		OutputVec:  res,
		ROutputVec: resR,
	}
}

func (s *scaleRowsRResult) Output() linalg.Vector {
	return s.OutputVec
}

func (s *scaleRowsRResult) ROutput() linalg.Vector {
	return s.ROutputVec
}

func (s *scaleRowsRResult) Constant(rg RGradient, g Gradient) bool {
	return s.Scalers.Constant(rg, g) && s.Matrix.Constant(rg, g)
}

func (s *scaleRowsRResult) PropagateRGradient(u, uR linalg.Vector, rg RGradient, g Gradient) {
	matData := s.Matrix.Output()
	matDataR := s.Matrix.ROutput()

	if !s.Scalers.Constant(rg, g) {
		su := make(linalg.Vector, len(s.Scalers.Output()))
		suR := make(linalg.Vector, len(su))
		cols := len(matData) / len(su)
		for row := range su {
			matRow := matData[row*cols : (row+1)*cols]
			matRowR := matDataR[row*cols : (row+1)*cols]
			upstreamRow := u[row*cols : (row+1)*cols]
			upstreamRowR := uR[row*cols : (row+1)*cols]
			su[row] = matRow.DotFast(upstreamRow)
			suR[row] = matRowR.DotFast(upstreamRow) + matRow.DotFast(upstreamRowR)
		}
		s.Scalers.PropagateRGradient(su, suR, rg, g)
	}
	if !s.Matrix.Constant(rg, g) {
		scalers := s.Scalers.Output()
		scalersR := s.Scalers.ROutput()
		cols := len(matData) / len(scalers)
		for row, scaler := range scalers {
			scalerR := scalersR[row]
			blas64.Scal(cols, scaler, blas64.Vector{
				Data: uR[row*cols : (row+1)*cols],
				Inc:  1,
			})
			uRow := u[row*cols : (row+1)*cols]
			uR[row*cols : (row+1)*cols].Add(uRow.Copy().Scale(scalerR))
			blas64.Scal(cols, scaler, blas64.Vector{Data: uRow, Inc: 1})
		}
		s.Matrix.PropagateRGradient(u, uR, rg, g)
	}
}
