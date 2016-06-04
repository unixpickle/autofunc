package autofunc

import "github.com/unixpickle/num-analysis/linalg"

type ResultSum struct {
	OutputVec linalg.Vector

	R1 Result
	R2 Result
}

// Add adds two Results.
func Add(r1, r2 Result) *ResultSum {
	return &ResultSum{
		OutputVec: r1.Output().Copy().Add(r2.Output()),
		R1:        r1,
		R2:        r2,
	}
}

func (r *ResultSum) Output() linalg.Vector {
	return r.OutputVec
}

func (r *ResultSum) Constant(grad Gradient) bool {
	return r.R1.Constant(grad) && r.R2.Constant(grad)
}

func (r *ResultSum) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if !r.R1.Constant(grad) {
		r.R1.PropagateGradient(upstream, grad)
	}
	if !r.R2.Constant(grad) {
		r.R2.PropagateGradient(upstream, grad)
	}
}

type RResultSum struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector

	R1 RResult
	R2 RResult
}

// AddR adds two RResults.
func AddR(r1, r2 RResult) *RResultSum {
	return &RResultSum{
		OutputVec:  r1.Output().Copy().Add(r2.Output()),
		ROutputVec: r1.ROutput().Copy().Add(r2.ROutput()),
		R1:         r1,
		R2:         r2,
	}
}

func (r *RResultSum) Output() linalg.Vector {
	return r.OutputVec
}

func (r *RResultSum) ROutput() linalg.Vector {
	return r.ROutputVec
}

func (r *RResultSum) Constant(rg RGradient, g Gradient) bool {
	return r.R1.Constant(rg, g) && r.R2.Constant(rg, g)
}

func (r *RResultSum) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	if !r.R1.Constant(rgrad, grad) {
		r.R1.PropagateRGradient(upstream, upstreamR, rgrad, grad)
	}
	if !r.R2.Constant(rgrad, grad) {
		r.R2.PropagateRGradient(upstream, upstreamR, rgrad, grad)
	}
}

type ResultProduct struct {
	OutputVec linalg.Vector

	R1 Result
	R2 Result
}

// Mul multiplies two Results component-wise.
func Mul(r1, r2 Result) *ResultProduct {
	r1Output := r1.Output()
	r2Output := r2.Output()
	if len(r1Output) != len(r2Output) {
		panic("vector sizes do not match")
	}
	product := make(linalg.Vector, len(r1Output))
	for i, x := range r1Output {
		product[i] = x * r2Output[i]
	}
	return &ResultProduct{
		OutputVec: product,
		R1:        r1,
		R2:        r2,
	}
}

func (r *ResultProduct) Output() linalg.Vector {
	return r.OutputVec
}

func (r *ResultProduct) Constant(g Gradient) bool {
	return r.R1.Constant(g) && r.R2.Constant(g)
}

func (r *ResultProduct) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	downstream := make(linalg.Vector, len(upstream))

	if !r.R1.Constant(grad) {
		r2Out := r.R2.Output()
		for i, x := range upstream {
			downstream[i] = x * r2Out[i]
		}
		r.R1.PropagateGradient(downstream, grad)
	}

	if !r.R2.Constant(grad) {
		r1Out := r.R1.Output()
		for i, x := range upstream {
			downstream[i] = x * r1Out[i]
		}
		r.R2.PropagateGradient(downstream, grad)
	}
}

type RResultProduct struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector

	R1 RResult
	R2 RResult
}

// MulR multiplies two RResults component-wise.
func MulR(r1, r2 RResult) *RResultProduct {
	r1Output := r1.Output()
	r1OutputR := r1.ROutput()
	r2Output := r2.Output()
	r2OutputR := r2.ROutput()
	if len(r1Output) != len(r2Output) {
		panic("vector sizes do not match")
	}
	product := make(linalg.Vector, len(r1Output))
	productR := make(linalg.Vector, len(r1Output))
	for i, x := range r1Output {
		y := r2Output[i]
		product[i] = x * y
		productR[i] = x*r2OutputR[i] + r1OutputR[i]*y
	}
	return &RResultProduct{
		OutputVec:  product,
		ROutputVec: productR,
		R1:         r1,
		R2:         r2,
	}
}

func (r *RResultProduct) Output() linalg.Vector {
	return r.OutputVec
}

func (r *RResultProduct) ROutput() linalg.Vector {
	return r.ROutputVec
}

func (r *RResultProduct) Constant(rg RGradient, g Gradient) bool {
	return r.R1.Constant(rg, g) && r.R2.Constant(rg, g)
}

func (r *RResultProduct) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	downstream := make(linalg.Vector, len(upstream))
	downstreamR := make(linalg.Vector, len(upstream))

	if !r.R1.Constant(rgrad, grad) {
		r2Out := r.R2.Output()
		r2OutR := r.R2.ROutput()
		for i, x := range upstream {
			otherOut := r2Out[i]
			downstream[i] = x * otherOut
			downstreamR[i] = x*r2OutR[i] + upstreamR[i]*otherOut
		}
		r.R1.PropagateRGradient(downstream, downstreamR, rgrad, grad)
	}

	if !r.R2.Constant(rgrad, grad) {
		r1Out := r.R1.Output()
		r1OutR := r.R1.ROutput()
		for i, x := range upstream {
			otherOut := r1Out[i]
			downstream[i] = x * otherOut
			downstreamR[i] = x*r1OutR[i] + upstreamR[i]*otherOut
		}
		r.R2.PropagateRGradient(downstream, downstreamR, rgrad, grad)
	}
}

type ScaledResult struct {
	OutputVec linalg.Vector
	Scaler    float64
	Input     Result
}

// Scale scales a Result component-wise.
func Scale(r Result, f float64) *ScaledResult {
	return &ScaledResult{
		OutputVec: r.Output().Copy().Scale(f),
		Scaler:    f,
		Input:     r,
	}
}

func (s *ScaledResult) Output() linalg.Vector {
	return s.OutputVec
}

func (s *ScaledResult) Constant(g Gradient) bool {
	return s.Input.Constant(g)
}

func (s *ScaledResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if !s.Input.Constant(grad) {
		s.Input.PropagateGradient(upstream.Scale(s.Scaler), grad)
	}
}

type ScaledRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Scaler     float64
	Input      RResult
}

// ScaleR scales an RResult component-wise.
func ScaleR(r RResult, f float64) *ScaledRResult {
	return &ScaledRResult{
		OutputVec:  r.Output().Copy().Scale(f),
		ROutputVec: r.ROutput().Copy().Scale(f),
		Scaler:     f,
		Input:      r,
	}
}

func (s *ScaledRResult) Output() linalg.Vector {
	return s.OutputVec
}

func (s *ScaledRResult) ROutput() linalg.Vector {
	return s.ROutputVec
}

func (s *ScaledRResult) Constant(rg RGradient, g Gradient) bool {
	return s.Input.Constant(rg, g)
}

func (s *ScaledRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	if !s.Input.Constant(rgrad, grad) {
		s.Input.PropagateRGradient(upstream.Scale(s.Scaler), upstreamR.Scale(s.Scaler),
			rgrad, grad)
	}
}

type ResultSquare struct {
	OutputVec linalg.Vector
	Input     Result
}

// Square squares every component of a Result.
func Square(r Result) *ResultSquare {
	rVec := r.Output()
	out := make(linalg.Vector, len(rVec))
	for i, x := range rVec {
		out[i] = x * x
	}
	return &ResultSquare{
		OutputVec: out,
		Input:     r,
	}
}

func (r *ResultSquare) Output() linalg.Vector {
	return r.OutputVec
}

func (r *ResultSquare) Constant(g Gradient) bool {
	return r.Input.Constant(g)
}

func (r *ResultSquare) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if !r.Input.Constant(grad) {
		for i, x := range r.Input.Output() {
			upstream[i] *= x * 2
		}
		r.Input.PropagateGradient(upstream, grad)
	}
}

type RResultSquare struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      RResult
}

// SquareR squares every component of an RResult.
func SquareR(r RResult) *RResultSquare {
	vec := r.Output()
	vecR := r.ROutput()
	out := make(linalg.Vector, len(vec))
	outR := make(linalg.Vector, len(vec))
	for i, x := range vec {
		out[i] = x * x
		outR[i] = 2 * x * vecR[i]
	}
	return &RResultSquare{
		OutputVec:  out,
		ROutputVec: outR,
		Input:      r,
	}
}

func (r *RResultSquare) Output() linalg.Vector {
	return r.OutputVec
}

func (r *RResultSquare) ROutput() linalg.Vector {
	return r.ROutputVec
}

func (r *RResultSquare) Constant(rg RGradient, g Gradient) bool {
	return r.Input.Constant(rg, g)
}

func (r *RResultSquare) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	if !r.Input.Constant(rgrad, grad) {
		inR := r.Input.ROutput()
		for i, x := range r.Input.Output() {
			upstreamR[i] = 2 * (upstreamR[i]*x + upstream[i]*inR[i])
			upstream[i] *= x * 2
		}
		r.Input.PropagateRGradient(upstream, upstreamR, rgrad, grad)
	}
}

type ResultInverse struct {
	OutputVec linalg.Vector
	Input     Result
}

// Inverse computes component-wise reciprocals.
// NaNs or Infs will result from 0-divisions.
func Inverse(r Result) *ResultInverse {
	inVec := r.Output()
	outVec := make(linalg.Vector, len(inVec))
	for i, x := range inVec {
		outVec[i] = 1 / x
	}
	return &ResultInverse{
		OutputVec: outVec,
		Input:     r,
	}
}

func (r *ResultInverse) Output() linalg.Vector {
	return r.OutputVec
}

func (r *ResultInverse) Constant(g Gradient) bool {
	return r.Input.Constant(g)
}

func (r *ResultInverse) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if !r.Input.Constant(grad) {
		for i, x := range r.OutputVec {
			upstream[i] *= -x * x
		}
		r.Input.PropagateGradient(upstream, grad)
	}
}

type RResultInverse struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	SquaredOut linalg.Vector
	Input      RResult
}

// InverseR is like Inverse, but for RResults.
func InverseR(r RResult) *RResultInverse {
	inVec := r.Output()
	inVecR := r.ROutput()
	outVec := make(linalg.Vector, len(inVec))
	outVecR := make(linalg.Vector, len(inVec))
	squaredOut := make(linalg.Vector, len(inVec))
	for i, x := range inVec {
		recip := 1 / x
		squared := recip * recip
		outVec[i] = recip
		squaredOut[i] = squared
		outVecR[i] = -squared * inVecR[i]
	}
	return &RResultInverse{
		OutputVec:  outVec,
		ROutputVec: outVecR,
		SquaredOut: squaredOut,
		Input:      r,
	}
}

func (r *RResultInverse) Output() linalg.Vector {
	return r.OutputVec
}

func (r *RResultInverse) ROutput() linalg.Vector {
	return r.ROutputVec
}

func (r *RResultInverse) Constant(rg RGradient, g Gradient) bool {
	return r.Input.Constant(rg, g)
}

func (r *RResultInverse) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	if !r.Input.Constant(rgrad, grad) {
		inR := r.Input.ROutput()
		for i, u := range upstream {
			uR := upstreamR[i]
			squared := -r.SquaredOut[i]
			output := r.OutputVec[i]
			xR := inR[i]
			upstream[i] = squared * u
			upstreamR[i] = squared*uR + -2*(squared*output)*xR*u
		}
		r.Input.PropagateRGradient(upstream, upstreamR, rgrad, grad)
	}
}
