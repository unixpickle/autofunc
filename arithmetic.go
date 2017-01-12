package autofunc

import (
	"math"

	"github.com/unixpickle/num-analysis/linalg"
)

type resultSum struct {
	OutputVec linalg.Vector

	R1 Result
	R2 Result
}

// Add adds two Results.
func Add(r1, r2 Result) Result {
	return &resultSum{
		OutputVec: r1.Output().Copy().Add(r2.Output()),
		R1:        r1,
		R2:        r2,
	}
}

func (r *resultSum) Output() linalg.Vector {
	return r.OutputVec
}

func (r *resultSum) Constant(grad Gradient) bool {
	return r.R1.Constant(grad) && r.R2.Constant(grad)
}

func (r *resultSum) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	r2Const := r.R2.Constant(grad)
	if !r.R1.Constant(grad) {
		if r2Const {
			r.R1.PropagateGradient(upstream, grad)
		} else {
			backup := make(linalg.Vector, len(upstream))
			copy(backup, upstream)
			r.R1.PropagateGradient(backup, grad)
		}
	}
	if !r2Const {
		r.R2.PropagateGradient(upstream, grad)
	}
}

type rresultSum struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector

	R1 RResult
	R2 RResult
}

// AddR adds two RResults.
func AddR(r1, r2 RResult) RResult {
	return &rresultSum{
		OutputVec:  r1.Output().Copy().Add(r2.Output()),
		ROutputVec: r1.ROutput().Copy().Add(r2.ROutput()),
		R1:         r1,
		R2:         r2,
	}
}

func (r *rresultSum) Output() linalg.Vector {
	return r.OutputVec
}

func (r *rresultSum) ROutput() linalg.Vector {
	return r.ROutputVec
}

func (r *rresultSum) Constant(rg RGradient, g Gradient) bool {
	return r.R1.Constant(rg, g) && r.R2.Constant(rg, g)
}

func (r *rresultSum) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	r2Const := r.R2.Constant(rgrad, grad)
	if !r.R1.Constant(rgrad, grad) {
		if r2Const {
			r.R1.PropagateRGradient(upstream, upstreamR, rgrad, grad)
		} else {
			backup := make(linalg.Vector, len(upstream))
			backupR := make(linalg.Vector, len(upstreamR))
			copy(backup, upstream)
			copy(backupR, upstreamR)
			r.R1.PropagateRGradient(backup, backupR, rgrad, grad)
		}
	}
	if !r2Const {
		r.R2.PropagateRGradient(upstream, upstreamR, rgrad, grad)
	}
}

type resultDiff struct {
	OutputVec linalg.Vector
	R1        Result
	R2        Result
}

// Sub subtracts b from a (componentwise).
func Sub(a, b Result) Result {
	aOut := a.Output()
	bOut := b.Output()
	res := make(linalg.Vector, len(aOut))
	for i, x := range aOut {
		res[i] = x - bOut[i]
	}
	return &resultDiff{
		OutputVec: res,
		R1:        a,
		R2:        b,
	}
}

func (r *resultDiff) Output() linalg.Vector {
	return r.OutputVec
}

func (r *resultDiff) Constant(g Gradient) bool {
	return r.R1.Constant(g) && r.R2.Constant(g)
}

func (r *resultDiff) PropagateGradient(u linalg.Vector, g Gradient) {
	if !r.R1.Constant(g) {
		r.R1.PropagateGradient(u.Copy(), g)
	}
	if !r.R2.Constant(g) {
		r.R2.PropagateGradient(u.Scale(-1), g)
	}
}

type rresultDiff struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	R1         RResult
	R2         RResult
}

// SubR subtracts b from a (componentwise).
func SubR(a, b RResult) RResult {
	aOut := a.Output()
	bOut := b.Output()
	aOutR := a.ROutput()
	bOutR := b.ROutput()
	res := make(linalg.Vector, len(aOut))
	resR := make(linalg.Vector, len(aOut))
	for i, x := range aOut {
		res[i] = x - bOut[i]
		resR[i] = aOutR[i] - bOutR[i]
	}
	return &rresultDiff{
		OutputVec:  res,
		ROutputVec: resR,
		R1:         a,
		R2:         b,
	}
}

func (r *rresultDiff) Output() linalg.Vector {
	return r.OutputVec
}

func (r *rresultDiff) ROutput() linalg.Vector {
	return r.ROutputVec
}

func (r *rresultDiff) Constant(rg RGradient, g Gradient) bool {
	return r.R1.Constant(rg, g) && r.R2.Constant(rg, g)
}

func (r *rresultDiff) PropagateRGradient(u, uR linalg.Vector, rg RGradient, g Gradient) {
	if !r.R1.Constant(rg, g) {
		r.R1.PropagateRGradient(u.Copy(), uR.Copy(), rg, g)
	}
	if !r.R2.Constant(rg, g) {
		r.R2.PropagateRGradient(u.Scale(-1), uR.Scale(-1), rg, g)
	}
}

type addScalerResult struct {
	OutputVec linalg.Vector
	Scaler    float64
	Input     Result
}

// AddScaler adds a scaler to every component of a vector.
func AddScaler(r Result, f float64) Result {
	inVec := r.Output()
	res := make(linalg.Vector, len(inVec))
	for i, x := range inVec {
		res[i] = x + f
	}
	return &addScalerResult{
		OutputVec: res,
		Scaler:    f,
		Input:     r,
	}
}

func (a *addScalerResult) Output() linalg.Vector {
	return a.OutputVec
}

func (a *addScalerResult) Constant(g Gradient) bool {
	return a.Input.Constant(g)
}

func (a *addScalerResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	a.Input.PropagateGradient(upstream, grad)
}

type addScalerRResult struct {
	OutputVec linalg.Vector
	Scaler    float64
	Input     RResult
}

// AddScalerR is like AddScaler, but with RResults.
func AddScalerR(r RResult, f float64) RResult {
	inVec := r.Output()
	res := make(linalg.Vector, len(inVec))
	for i, x := range inVec {
		res[i] = x + f
	}
	return &addScalerRResult{
		OutputVec: res,
		Scaler:    f,
		Input:     r,
	}
}

func (a *addScalerRResult) Output() linalg.Vector {
	return a.OutputVec
}

func (a *addScalerRResult) ROutput() linalg.Vector {
	return a.Input.ROutput()
}

func (a *addScalerRResult) Constant(rg RGradient, g Gradient) bool {
	return a.Input.Constant(rg, g)
}

func (a *addScalerRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	a.Input.PropagateRGradient(upstream, upstreamR, rgrad, grad)
}

type resultProduct struct {
	OutputVec linalg.Vector

	R1 Result
	R2 Result
}

// Mul multiplies two Results component-wise.
func Mul(r1, r2 Result) Result {
	r1Output := r1.Output()
	r2Output := r2.Output()
	if len(r1Output) != len(r2Output) {
		panic("vector sizes do not match")
	}
	product := make(linalg.Vector, len(r1Output))
	for i, x := range r1Output {
		product[i] = x * r2Output[i]
	}
	return &resultProduct{
		OutputVec: product,
		R1:        r1,
		R2:        r2,
	}
}

func (r *resultProduct) Output() linalg.Vector {
	return r.OutputVec
}

func (r *resultProduct) Constant(g Gradient) bool {
	return r.R1.Constant(g) && r.R2.Constant(g)
}

func (r *resultProduct) PropagateGradient(upstream linalg.Vector, grad Gradient) {
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

type rresultProduct struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector

	R1 RResult
	R2 RResult
}

// MulR multiplies two RResults component-wise.
func MulR(r1, r2 RResult) RResult {
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
	return &rresultProduct{
		OutputVec:  product,
		ROutputVec: productR,
		R1:         r1,
		R2:         r2,
	}
}

func (r *rresultProduct) Output() linalg.Vector {
	return r.OutputVec
}

func (r *rresultProduct) ROutput() linalg.Vector {
	return r.ROutputVec
}

func (r *rresultProduct) Constant(rg RGradient, g Gradient) bool {
	return r.R1.Constant(rg, g) && r.R2.Constant(rg, g)
}

func (r *rresultProduct) PropagateRGradient(upstream, upstreamR linalg.Vector,
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

type resultQuotient struct {
	OutputVec linalg.Vector
	Num       Result
	Denom     Result
}

// Div computes a/b (elementwise).
func Div(a, b Result) Result {
	aOut := a.Output()
	bOut := b.Output()
	out := make(linalg.Vector, len(aOut))
	for i, x := range aOut {
		out[i] = x / bOut[i]
	}
	return &resultQuotient{
		OutputVec: out,
		Num:       a,
		Denom:     b,
	}
}

func (r *resultQuotient) Output() linalg.Vector {
	return r.OutputVec
}

func (r *resultQuotient) Constant(g Gradient) bool {
	return r.Num.Constant(g) && r.Denom.Constant(g)
}

func (r *resultQuotient) PropagateGradient(u linalg.Vector, g Gradient) {
	denomOut := r.Denom.Output()
	if !r.Num.Constant(g) {
		uScaled := u.Copy()
		for i, x := range denomOut {
			uScaled[i] /= x
		}
		r.Num.PropagateGradient(uScaled, g)
	}
	if !r.Denom.Constant(g) {
		for i, x := range r.OutputVec {
			u[i] *= -x / denomOut[i]
		}
		r.Denom.PropagateGradient(u, g)
	}
}

// DivR computes a/b (elementwise).
func DivR(a, b RResult) RResult {
	return MulR(a, InverseR(b))
}

type scaledResult struct {
	OutputVec linalg.Vector
	Scaler    float64
	Input     Result
}

// Scale scales a Result component-wise.
func Scale(r Result, f float64) Result {
	return &scaledResult{
		OutputVec: r.Output().Copy().Scale(f),
		Scaler:    f,
		Input:     r,
	}
}

func (s *scaledResult) Output() linalg.Vector {
	return s.OutputVec
}

func (s *scaledResult) Constant(g Gradient) bool {
	return s.Input.Constant(g)
}

func (s *scaledResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if !s.Input.Constant(grad) {
		s.Input.PropagateGradient(upstream.Scale(s.Scaler), grad)
	}
}

type scaledRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Scaler     float64
	Input      RResult
}

// ScaleR scales an RResult component-wise.
func ScaleR(r RResult, f float64) RResult {
	return &scaledRResult{
		OutputVec:  r.Output().Copy().Scale(f),
		ROutputVec: r.ROutput().Copy().Scale(f),
		Scaler:     f,
		Input:      r,
	}
}

func (s *scaledRResult) Output() linalg.Vector {
	return s.OutputVec
}

func (s *scaledRResult) ROutput() linalg.Vector {
	return s.ROutputVec
}

func (s *scaledRResult) Constant(rg RGradient, g Gradient) bool {
	return s.Input.Constant(rg, g)
}

func (s *scaledRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	if !s.Input.Constant(rgrad, grad) {
		s.Input.PropagateRGradient(upstream.Scale(s.Scaler), upstreamR.Scale(s.Scaler),
			rgrad, grad)
	}
}

type scaleFirstResult struct {
	OutputVec linalg.Vector
	Scaler    Result
	Input     Result
}

// ScaleFirst scales all the elements of a
// vector by the first element of a vector.
func ScaleFirst(in Result, scaler Result) Result {
	f := scaler.Output()[0]
	return &scaleFirstResult{
		OutputVec: in.Output().Copy().Scale(f),
		Scaler:    scaler,
		Input:     in,
	}
}

func (s *scaleFirstResult) Output() linalg.Vector {
	return s.OutputVec
}

func (s *scaleFirstResult) Constant(g Gradient) bool {
	return s.Input.Constant(g) && s.Scaler.Constant(g)
}

func (s *scaleFirstResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if !s.Scaler.Constant(grad) {
		downstream := make(linalg.Vector, len(s.Scaler.Output()))
		downstream[0] = upstream.DotFast(s.Input.Output())
		s.Scaler.PropagateGradient(downstream, grad)
	}

	// This logic is intentionally done after propagating
	// through s.Scaler so we can scale upstream in place.
	if !s.Input.Constant(grad) {
		scaler := s.Scaler.Output()[0]
		s.Input.PropagateGradient(upstream.Scale(scaler), grad)
	}
}

type scaleFirstRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Scaler     RResult
	Input      RResult
}

// ScaleFirstR is like ScaleFirst, but for RResults.
func ScaleFirstR(in RResult, scaler RResult) RResult {
	f := scaler.Output()[0]
	fR := scaler.ROutput()[0]
	return &scaleFirstRResult{
		OutputVec:  in.Output().Copy().Scale(f),
		ROutputVec: in.Output().Copy().Scale(fR).Add(in.ROutput().Copy().Scale(f)),
		Scaler:     scaler,
		Input:      in,
	}
}

func (s *scaleFirstRResult) Output() linalg.Vector {
	return s.OutputVec
}

func (s *scaleFirstRResult) ROutput() linalg.Vector {
	return s.ROutputVec
}

func (s *scaleFirstRResult) Constant(rg RGradient, g Gradient) bool {
	return s.Input.Constant(rg, g) && s.Scaler.Constant(rg, g)
}

func (s *scaleFirstRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	if !s.Scaler.Constant(rgrad, grad) {
		downstream := make(linalg.Vector, len(s.Scaler.Output()))
		downstreamR := make(linalg.Vector, len(s.Scaler.ROutput()))
		downstream[0] = upstream.DotFast(s.Input.Output())
		downstreamR[0] = upstreamR.DotFast(s.Input.Output()) +
			upstream.DotFast(s.Input.ROutput())
		s.Scaler.PropagateRGradient(downstream, downstreamR, rgrad, grad)
	}

	// This is intentionally done after propagating s.Scaler.
	// See scaleFirstResult.PropagateGradient() for more.
	if !s.Input.Constant(rgrad, grad) {
		scaler := s.Scaler.Output()[0]
		scalerR := s.Scaler.ROutput()[0]
		upstreamR.Scale(scaler).Add(upstream.Copy().Scale(scalerR))
		upstream.Scale(scaler)
		s.Input.PropagateRGradient(upstream, upstreamR, rgrad, grad)
	}
}

type addFirstResult struct {
	OutputVec linalg.Vector
	Input     Result
	Scaler    Result
}

// AddFirst adds the first element of v2 to each
// element of v1.
// This is basically like AddScaler, but instead
// of a constant, the first element of v2 is used.
func AddFirst(v1 Result, v2 Result) Result {
	inVec := v1.Output()
	outVec := make(linalg.Vector, len(inVec))
	scaler := v2.Output()[0]
	for i, x := range inVec {
		outVec[i] = scaler + x
	}
	return &addFirstResult{
		OutputVec: outVec,
		Input:     v1,
		Scaler:    v2,
	}
}

func (a *addFirstResult) Output() linalg.Vector {
	return a.OutputVec
}

func (a *addFirstResult) Constant(g Gradient) bool {
	return a.Input.Constant(g) && a.Scaler.Constant(g)
}

func (a *addFirstResult) PropagateGradient(upstream linalg.Vector, g Gradient) {
	if !a.Scaler.Constant(g) {
		scalerUpstream := make(linalg.Vector, len(a.Scaler.Output()))
		for _, x := range upstream {
			scalerUpstream[0] += x
		}
		a.Scaler.PropagateGradient(scalerUpstream, g)
	}
	if !a.Input.Constant(g) {
		a.Input.PropagateGradient(upstream, g)
	}
}

type addFirstRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      RResult
	Scaler     RResult
}

// AddFirstR is like AddFirst, but for RResults.
func AddFirstR(v1 RResult, v2 RResult) RResult {
	inVec := v1.Output()
	inVecR := v1.ROutput()
	outVec := make(linalg.Vector, len(inVec))
	outVecR := make(linalg.Vector, len(inVec))
	scaler := v2.Output()[0]
	scalerR := v2.ROutput()[0]
	for i, x := range inVec {
		outVec[i] = scaler + x
	}
	for i, x := range inVecR {
		outVecR[i] = scalerR + x
	}
	return &addFirstRResult{
		OutputVec:  outVec,
		ROutputVec: outVecR,
		Input:      v1,
		Scaler:     v2,
	}
}

func (a *addFirstRResult) Output() linalg.Vector {
	return a.OutputVec
}

func (a *addFirstRResult) ROutput() linalg.Vector {
	return a.ROutputVec
}

func (a *addFirstRResult) Constant(rg RGradient, g Gradient) bool {
	return a.Input.Constant(rg, g) && a.Scaler.Constant(rg, g)
}

func (a *addFirstRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rg RGradient, g Gradient) {
	if !a.Scaler.Constant(rg, g) {
		scalerUpstream := make(linalg.Vector, len(a.Scaler.Output()))
		scalerUpstreamR := make(linalg.Vector, len(a.Scaler.Output()))
		for i, x := range upstream {
			scalerUpstream[0] += x
			scalerUpstreamR[0] += upstreamR[i]
		}
		a.Scaler.PropagateRGradient(scalerUpstream, scalerUpstreamR, rg, g)
	}
	if !a.Input.Constant(rg, g) {
		a.Input.PropagateRGradient(upstream, upstreamR, rg, g)
	}
}

type resultSquare struct {
	OutputVec linalg.Vector
	Input     Result
}

// Square squares every component of a Result.
func Square(r Result) Result {
	rVec := r.Output()
	out := make(linalg.Vector, len(rVec))
	for i, x := range rVec {
		out[i] = x * x
	}
	return &resultSquare{
		OutputVec: out,
		Input:     r,
	}
}

func (r *resultSquare) Output() linalg.Vector {
	return r.OutputVec
}

func (r *resultSquare) Constant(g Gradient) bool {
	return r.Input.Constant(g)
}

func (r *resultSquare) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if !r.Input.Constant(grad) {
		for i, x := range r.Input.Output() {
			upstream[i] *= x * 2
		}
		r.Input.PropagateGradient(upstream, grad)
	}
}

type rresultSquare struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      RResult
}

// SquareR squares every component of an RResult.
func SquareR(r RResult) RResult {
	vec := r.Output()
	vecR := r.ROutput()
	out := make(linalg.Vector, len(vec))
	outR := make(linalg.Vector, len(vec))
	for i, x := range vec {
		out[i] = x * x
		outR[i] = 2 * x * vecR[i]
	}
	return &rresultSquare{
		OutputVec:  out,
		ROutputVec: outR,
		Input:      r,
	}
}

func (r *rresultSquare) Output() linalg.Vector {
	return r.OutputVec
}

func (r *rresultSquare) ROutput() linalg.Vector {
	return r.ROutputVec
}

func (r *rresultSquare) Constant(rg RGradient, g Gradient) bool {
	return r.Input.Constant(rg, g)
}

func (r *rresultSquare) PropagateRGradient(upstream, upstreamR linalg.Vector,
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

type resultInverse struct {
	OutputVec linalg.Vector
	Input     Result
}

// Inverse computes component-wise reciprocals.
// NaNs or Infs will result from 0-divisions.
func Inverse(r Result) Result {
	inVec := r.Output()
	outVec := make(linalg.Vector, len(inVec))
	for i, x := range inVec {
		outVec[i] = 1 / x
	}
	return &resultInverse{
		OutputVec: outVec,
		Input:     r,
	}
}

func (r *resultInverse) Output() linalg.Vector {
	return r.OutputVec
}

func (r *resultInverse) Constant(g Gradient) bool {
	return r.Input.Constant(g)
}

func (r *resultInverse) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if !r.Input.Constant(grad) {
		for i, x := range r.OutputVec {
			upstream[i] *= -x * x
		}
		r.Input.PropagateGradient(upstream, grad)
	}
}

type rresultInverse struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	SquaredOut linalg.Vector
	Input      RResult
}

// InverseR is like Inverse, but for RResults.
func InverseR(r RResult) RResult {
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
	return &rresultInverse{
		OutputVec:  outVec,
		ROutputVec: outVecR,
		SquaredOut: squaredOut,
		Input:      r,
	}
}

func (r *rresultInverse) Output() linalg.Vector {
	return r.OutputVec
}

func (r *rresultInverse) ROutput() linalg.Vector {
	return r.ROutputVec
}

func (r *rresultInverse) Constant(rg RGradient, g Gradient) bool {
	return r.Input.Constant(rg, g)
}

func (r *rresultInverse) PropagateRGradient(upstream, upstreamR linalg.Vector,
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

type resultPow struct {
	OutputVec linalg.Vector
	Power     float64
	Input     Result
}

// Pow raises each component of r to a given power.
func Pow(r Result, pow float64) Result {
	input := r.Output()
	output := make(linalg.Vector, len(input))
	for i, x := range input {
		output[i] = math.Pow(x, pow)
	}
	return &resultPow{
		OutputVec: output,
		Power:     pow,
		Input:     r,
	}
}

func (r *resultPow) Output() linalg.Vector {
	return r.OutputVec
}

func (r *resultPow) Constant(g Gradient) bool {
	return r.Power != 0 && r.Input.Constant(g)
}

func (r *resultPow) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if !r.Constant(grad) {
		if r.Power != 1 {
			for i, x := range r.Input.Output() {
				upstream[i] *= r.Power * math.Pow(x, r.Power-1)
			}
		}
		r.Input.PropagateGradient(upstream, grad)
	}
}

type rresultPow struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Power      float64
	Input      RResult
}

// PowR is like Pow, but for RResults.
func PowR(r RResult, pow float64) RResult {
	input := r.Output()
	inputR := r.ROutput()
	output := make(linalg.Vector, len(input))
	outputR := make(linalg.Vector, len(input))
	for i, x := range input {
		output[i] = math.Pow(x, pow)
	}
	if pow != 0 {
		for i, x := range input {
			xR := inputR[i]
			outputR[i] = pow * math.Pow(x, pow-1) * xR
		}
	}
	return &rresultPow{
		OutputVec:  output,
		ROutputVec: outputR,
		Power:      pow,
		Input:      r,
	}
}

func (r *rresultPow) Output() linalg.Vector {
	return r.OutputVec
}

func (r *rresultPow) ROutput() linalg.Vector {
	return r.ROutputVec
}

func (r *rresultPow) Constant(rg RGradient, g Gradient) bool {
	return r.Power != 0 && r.Input.Constant(rg, g)
}

func (r *rresultPow) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	if !r.Constant(rgrad, grad) {
		if r.Power != 1 {
			inputR := r.Input.ROutput()
			for i, x := range r.Input.Output() {
				u := upstream[i]
				uR := upstreamR[i]
				funcDeriv := r.Power * math.Pow(x, r.Power-1)
				funcDerivDeriv := r.Power * (r.Power - 1) * math.Pow(x, r.Power-2) * inputR[i]
				upstream[i] = u * funcDeriv
				upstreamR[i] = uR*funcDeriv + u*funcDerivDeriv
			}
		}
		r.Input.PropagateRGradient(upstream, upstreamR, rgrad, grad)
	}
}

type sumAllResult struct {
	OutputVec linalg.Vector
	Input     Result
}

// SumAll adds up all the components of r and
// returns a vector with that sum as its one
// and only element.
func SumAll(r Result) Result {
	var sum float64
	for _, x := range r.Output() {
		sum += x
	}
	return &sumAllResult{
		OutputVec: linalg.Vector{sum},
		Input:     r,
	}
}

func (s *sumAllResult) Output() linalg.Vector {
	return s.OutputVec
}

func (s *sumAllResult) Constant(g Gradient) bool {
	return s.Input.Constant(g)
}

func (s *sumAllResult) PropagateGradient(upstream linalg.Vector, grad Gradient) {
	if !s.Input.Constant(grad) {
		inLen := len(s.Input.Output())
		downstream := make(linalg.Vector, inLen)
		for i := range downstream {
			downstream[i] = upstream[0]
		}
		s.Input.PropagateGradient(downstream, grad)
	}
}

type sumAllRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      RResult
}

// SumAllR is like SumAll, but for an RResult.
func SumAllR(r RResult) RResult {
	var sum, rsum float64
	routput := r.ROutput()
	for i, x := range r.Output() {
		sum += x
		rsum += routput[i]
	}
	return &sumAllRResult{
		OutputVec:  linalg.Vector{sum},
		ROutputVec: linalg.Vector{rsum},
		Input:      r,
	}
}

func (s *sumAllRResult) Output() linalg.Vector {
	return s.OutputVec
}

func (s *sumAllRResult) ROutput() linalg.Vector {
	return s.ROutputVec
}

func (s *sumAllRResult) Constant(rg RGradient, g Gradient) bool {
	return s.Input.Constant(rg, g)
}

func (s *sumAllRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad RGradient, grad Gradient) {
	if !s.Input.Constant(rgrad, grad) {
		inLen := len(s.Input.Output())
		downstream := make(linalg.Vector, inLen)
		downstreamR := make(linalg.Vector, inLen)
		for i := range downstream {
			downstream[i] = upstream[0]
			downstreamR[i] = upstreamR[0]
		}
		s.Input.PropagateRGradient(downstream, downstreamR, rgrad, grad)
	}
}

// AddLogDomain adds values which are expressed as natural
// logarithms of their actual values.
// In other words, it is like exponentiating the elements
// of v1 and v2, adding them together, and then taking the
// natural log of the result.
// However, the above procedure is vulnerable to overflow
// issues, whereas AddLogDomain is not.
func AddLogDomain(v1, v2 Result) Result {
	maxVal := math.Max(v1.Output().MaxAbs(), v2.Output().MaxAbs())
	exp1 := Exp{}.Apply(AddScaler(v1, -maxVal))
	exp2 := Exp{}.Apply(AddScaler(v2, -maxVal))
	expSum := Add(exp1, exp2)
	return AddScaler(Log{}.Apply(expSum), maxVal)
}

// AddLogDomainR is like AddLogDomain, but for RResults.
func AddLogDomainR(v1, v2 RResult) RResult {
	rv := RVector{}
	maxVal := math.Max(v1.Output().MaxAbs(), v2.Output().MaxAbs())
	exp1 := Exp{}.ApplyR(rv, AddScalerR(v1, -maxVal))
	exp2 := Exp{}.ApplyR(rv, AddScalerR(v2, -maxVal))
	expSum := AddR(exp1, exp2)
	return AddScalerR(Log{}.ApplyR(rv, expSum), maxVal)
}

// SumAllLogDomain is a numerically-stable way to exponentiate
// the components of a vector, add the results, then take the
// natural log.
func SumAllLogDomain(v Result) Result {
	maxVal := v.Output().MaxAbs()
	exp := Exp{}.Apply(AddScaler(v, -maxVal))
	sum := SumAll(exp)
	return AddScaler(Log{}.Apply(sum), maxVal)
}

// SumAllLogDomainR is like SumAllLogDomain but for RResults.
func SumAllLogDomainR(v RResult) RResult {
	maxVal := v.Output().MaxAbs()
	exp := Exp{}.ApplyR(nil, AddScalerR(v, -maxVal))
	sum := SumAllR(exp)
	return AddScalerR(Log{}.ApplyR(nil, sum), maxVal)
}
