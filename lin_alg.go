package autofunc

import "github.com/unixpickle/num-analysis/linalg"

type matMulResult struct {
	MatIn  Result
	MatVar *Variable
	Res    Result
}

// MatMulVec multiplies a row-major matrix by a column
// vector.
func MatMulVec(mat Result, rows, cols int, vec Result) Result {
	if len(mat.Output()) != rows*cols {
		panic("invalid matrix data size")
	}
	v := &Variable{Vector: mat.Output()}
	lt := &LinTran{
		Data: v,
		Rows: rows,
		Cols: cols,
	}
	return &matMulResult{
		MatIn:  mat,
		MatVar: v,
		Res:    lt.Apply(vec),
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
	if len(mat.Output()) != rows*cols {
		panic("invalid matrix data size")
	}
	v := &Variable{Vector: mat.Output()}
	lt := &LinTran{
		Data: v,
		Rows: rows,
		Cols: cols,
	}
	return &matMulRResult{
		MatIn:  mat,
		MatVar: v,
		Res:    lt.ApplyR(RVector{v: mat.ROutput()}, vec),
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
