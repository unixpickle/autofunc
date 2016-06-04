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
