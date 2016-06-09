package autofunc

import (
	"math"

	"github.com/unixpickle/num-analysis/linalg"
)

// Arithmetic performs arithmetic operations using
// a given VectorCache.
type Arithmetic struct {
	Cache *VectorCache
}

var DefaultArithmetic = Arithmetic{Cache: DefaultVectorCache}

type resultSum struct {
	Cache     *VectorCache
	OutputVec linalg.Vector

	R1 Result
	R2 Result
}

// Add adds two Results.
func (a *Arithmetic) Add(r1, r2 Result) Result {
	r1Vec := r1.Output()
	r2Vec := r2.Output()
	result := a.Cache.Alloc(len(r1Vec))
	for i, x := range r1Vec {
		result[i] = x + r2Vec[i]
	}
	return &resultSum{
		Cache:     a.Cache,
		OutputVec: result,
		R1:        r1,
		R2:        r2,
	}
}

// Add is equivalent to DefaultArithmetic.Add.
func Add(r1, r2 Result) Result {
	return DefaultArithmetic.Add(r1, r2)
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

func (r *resultSum) Release() {
	r.Cache.Free(r.OutputVec)
	r.OutputVec = nil
	r.R1.Release()
	r.R2.Release()
}

type rresultSum struct {
	Cache      *VectorCache
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector

	R1 RResult
	R2 RResult
}

// AddR adds two RResults.
func (a *Arithmetic) AddR(r1, r2 RResult) RResult {
	r1Vec := r1.Output()
	r1VecR := r1.ROutput()
	r2Vec := r2.Output()
	r2VecR := r2.ROutput()
	result := a.Cache.Alloc(len(r1Vec))
	resultR := a.Cache.Alloc(len(r1Vec))
	for i, x := range r1Vec {
		result[i] = x + r2Vec[i]
		resultR[i] = r1VecR[i] + r2VecR[i]
	}
	return &rresultSum{
		Cache:      a.Cache,
		OutputVec:  result,
		ROutputVec: resultR,
		R1:         r1,
		R2:         r2,
	}
}

// AddR is equivalent to DefaultArithmetic.AddR.
func AddR(r1, r2 RResult) RResult {
	return DefaultArithmetic.AddR(r1, r2)
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

func (r *rresultSum) Release() {
	r.Cache.Free(r.OutputVec)
	r.Cache.Free(r.ROutputVec)
	r.OutputVec = nil
	r.ROutputVec = nil
	r.R1.Release()
	r.R2.Release()
}

type addScalerResult struct {
	Cache     *VectorCache
	OutputVec linalg.Vector
	Scaler    float64
	Input     Result
}

// AddScaler adds a scaler to every component of a vector.
func (a *Arithmetic) AddScaler(r Result, f float64) Result {
	inVec := r.Output()
	res := a.Cache.Alloc(len(inVec))
	for i, x := range inVec {
		res[i] = x + f
	}
	return &addScalerResult{
		Cache:     a.Cache,
		OutputVec: res,
		Scaler:    f,
		Input:     r,
	}
}

// AddScaler is equivalent to DefaultArithmetic.AddScaler.
func AddScaler(r Result, f float64) Result {
	return DefaultArithmetic.AddScaler(r, f)
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

func (a *addScalerResult) Release() {
	a.Cache.Free(a.OutputVec)
	a.OutputVec = nil
	a.Input.Release()
}

type addScalerRResult struct {
	Cache     *VectorCache
	OutputVec linalg.Vector
	Scaler    float64
	Input     RResult
}

// AddScalerR is like AddScaler, but with RResults.
func (a *Arithmetic) AddScalerR(r RResult, f float64) RResult {
	inVec := r.Output()
	res := a.Cache.Alloc(len(inVec))
	for i, x := range inVec {
		res[i] = x + f
	}
	return &addScalerRResult{
		Cache:     a.Cache,
		OutputVec: res,
		Scaler:    f,
		Input:     r,
	}
}

// AddScalerR is equivalent to DefaultArithmetic.AddScalerR.
func AddScalerR(r RResult, f float64) RResult {
	return DefaultArithmetic.AddScalerR(r, f)
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

func (a *addScalerRResult) Release() {
	a.Cache.Free(a.OutputVec)
	a.OutputVec = nil
	a.Input.Release()
}

type resultProduct struct {
	Cache     *VectorCache
	OutputVec linalg.Vector

	R1 Result
	R2 Result
}

// Mul multiplies two Results component-wise.
func (a *Arithmetic) Mul(r1, r2 Result) Result {
	r1Output := r1.Output()
	r2Output := r2.Output()
	if len(r1Output) != len(r2Output) {
		panic("vector sizes do not match")
	}
	product := a.Cache.Alloc(len(r1Output))
	for i, x := range r1Output {
		product[i] = x * r2Output[i]
	}
	return &resultProduct{
		Cache:     a.Cache,
		OutputVec: product,
		R1:        r1,
		R2:        r2,
	}
}

// Mul is equivalent to DefaultArithmetic.Mul.
func Mul(r1, r2 Result) Result {
	return DefaultArithmetic.Mul(r1, r2)
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

func (r *resultProduct) Release() {
	r.Cache.Free(r.OutputVec)
	r.OutputVec = nil
	r.R1.Release()
	r.R2.Release()
}

type rresultProduct struct {
	Cache      *VectorCache
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector

	R1 RResult
	R2 RResult
}

// MulR multiplies two RResults component-wise.
func (a *Arithmetic) MulR(r1, r2 RResult) RResult {
	r1Output := r1.Output()
	r1OutputR := r1.ROutput()
	r2Output := r2.Output()
	r2OutputR := r2.ROutput()
	if len(r1Output) != len(r2Output) {
		panic("vector sizes do not match")
	}
	product := a.Cache.Alloc(len(r1Output))
	productR := a.Cache.Alloc(len(r1Output))
	for i, x := range r1Output {
		y := r2Output[i]
		product[i] = x * y
		productR[i] = x*r2OutputR[i] + r1OutputR[i]*y
	}
	return &rresultProduct{
		Cache:      a.Cache,
		OutputVec:  product,
		ROutputVec: productR,
		R1:         r1,
		R2:         r2,
	}
}

// MulR is equivalent to DefaultArithmetic.MulR.
func MulR(r1, r2 RResult) RResult {
	return DefaultArithmetic.MulR(r1, r2)
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

func (r *rresultProduct) Release() {
	r.Cache.Free(r.OutputVec)
	r.Cache.Free(r.ROutputVec)
	r.OutputVec = nil
	r.ROutputVec = nil
	r.R1.Release()
	r.R2.Release()
}

type scaledResult struct {
	Cache     *VectorCache
	OutputVec linalg.Vector
	Scaler    float64
	Input     Result
}

// Scale scales a Result component-wise.
func (a *Arithmetic) Scale(r Result, f float64) Result {
	input := r.Output()
	result := a.Cache.Alloc(len(input))
	for i, x := range input {
		result[i] = x * f
	}
	return &scaledResult{
		Cache:     a.Cache,
		OutputVec: result,
		Scaler:    f,
		Input:     r,
	}
}

// Scale is equivalent to DefaultArithmetic.Scale.
func Scale(r Result, f float64) Result {
	return DefaultArithmetic.Scale(r, f)
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

func (s *scaledResult) Release() {
	s.Cache.Free(s.OutputVec)
	s.OutputVec = nil
	s.Input.Release()
}

type scaledRResult struct {
	Cache      *VectorCache
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Scaler     float64
	Input      RResult
}

// ScaleR scales an RResult component-wise.
func (a *Arithmetic) ScaleR(r RResult, f float64) RResult {
	input := r.Output()
	inputR := r.ROutput()
	result := a.Cache.Alloc(len(input))
	resultR := a.Cache.Alloc(len(input))
	for i, x := range input {
		result[i] = x * f
	}
	for i, x := range inputR {
		resultR[i] = x * f
	}
	return &scaledRResult{
		Cache:      a.Cache,
		OutputVec:  result,
		ROutputVec: resultR,
		Scaler:     f,
		Input:      r,
	}
}

// ScaleR is equivalent to DefaultArithmetic.ScaleR.
func ScaleR(r RResult, f float64) RResult {
	return DefaultArithmetic.ScaleR(r, f)
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

func (s *scaledRResult) Release() {
	s.Cache.Free(s.OutputVec)
	s.Cache.Free(s.ROutputVec)
	s.OutputVec = nil
	s.ROutputVec = nil
	s.Input.Release()
}

type scaleFirstResult struct {
	Cache     *VectorCache
	OutputVec linalg.Vector
	Scaler    Result
	Input     Result
}

// scaleFirstResult scales all the elements of
// a vector by the first element of a vector.
func (a *Arithmetic) ScaleFirst(in, scaler Result) Result {
	f := scaler.Output()[0]
	input := in.Output()
	result := a.Cache.Alloc(len(input))
	for i, x := range input {
		result[i] = x * f
	}
	return &scaleFirstResult{
		Cache:     a.Cache,
		OutputVec: result,
		Scaler:    scaler,
		Input:     in,
	}
}

// ScaleFirst is equivalent to DefaultArithmetic.ScaleFirst.
func ScaleFirst(in, scaler Result) Result {
	return DefaultArithmetic.ScaleFirst(in, scaler)
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

func (s *scaleFirstResult) Release() {
	s.Cache.Free(s.OutputVec)
	s.OutputVec = nil
	s.Input.Release()
	s.Scaler.Release()
}

type scaleFirstRResult struct {
	Cache      *VectorCache
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Scaler     RResult
	Input      RResult
}

// ScaleFirstR is like ScaleFirst, but for RResults.
func (a *Arithmetic) ScaleFirstR(in, scaler RResult) RResult {
	f := scaler.Output()[0]
	fR := scaler.ROutput()[0]

	input := in.Output()
	output := a.Cache.Alloc(len(input))
	for i, x := range input {
		output[i] = x * f
	}

	inputR := in.ROutput()
	outputR := a.Cache.Alloc(len(input))
	for i, x := range input {
		xR := inputR[i]
		outputR[i] = x*fR + xR*f
	}

	return &scaleFirstRResult{
		Cache:      a.Cache,
		OutputVec:  output,
		ROutputVec: outputR,
		Scaler:     scaler,
		Input:      in,
	}
}

// ScaleFirstR is equivalent to DefaultArithmetic.ScaleFirstR.
func ScaleFirstR(in, scaler RResult) RResult {
	return DefaultArithmetic.ScaleFirstR(in, scaler)
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

func (s *scaleFirstRResult) Release() {
	s.Cache.Free(s.OutputVec)
	s.Cache.Free(s.ROutputVec)
	s.OutputVec = nil
	s.ROutputVec = nil
	s.Scaler.Release()
	s.Input.Release()
}

type resultSquare struct {
	Cache     *VectorCache
	OutputVec linalg.Vector
	Input     Result
}

// Square squares every component of a Result.
func (a *Arithmetic) Square(r Result) Result {
	rVec := r.Output()
	out := a.Cache.Alloc(len(rVec))
	for i, x := range rVec {
		out[i] = x * x
	}
	return &resultSquare{
		Cache:     a.Cache,
		OutputVec: out,
		Input:     r,
	}
}

// Square is equivalent to DefaultArithmetic.Square.
func Square(r Result) Result {
	return DefaultArithmetic.Square(r)
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

func (r *resultSquare) Release() {
	r.Cache.Free(r.OutputVec)
	r.OutputVec = nil
	r.Input.Release()
}

type rresultSquare struct {
	Cache      *VectorCache
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      RResult
}

// SquareR squares every component of an RResult.
func (a *Arithmetic) SquareR(r RResult) RResult {
	vec := r.Output()
	vecR := r.ROutput()
	out := a.Cache.Alloc(len(vec))
	outR := a.Cache.Alloc(len(vec))
	for i, x := range vec {
		out[i] = x * x
		outR[i] = 2 * x * vecR[i]
	}
	return &rresultSquare{
		Cache:      a.Cache,
		OutputVec:  out,
		ROutputVec: outR,
		Input:      r,
	}
}

// SquareR is equivalent to DefaultArithmetic.SquareR.
func SquareR(r RResult) RResult {
	return DefaultArithmetic.SquareR(r)
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

func (r *rresultSquare) Release() {
	r.Cache.Free(r.OutputVec)
	r.Cache.Free(r.ROutputVec)
	r.OutputVec = nil
	r.ROutputVec = nil
	r.Input.Release()
}

type resultInverse struct {
	Cache     *VectorCache
	OutputVec linalg.Vector
	Input     Result
}

// Inverse computes component-wise reciprocals.
// NaNs or Infs will result from 0-divisions.
func (a *Arithmetic) Inverse(r Result) Result {
	inVec := r.Output()
	outVec := a.Cache.Alloc(len(inVec))
	for i, x := range inVec {
		outVec[i] = 1 / x
	}
	return &resultInverse{
		Cache:     a.Cache,
		OutputVec: outVec,
		Input:     r,
	}
}

// Inverse is equivalent to DefaultArithmetic.Inverse.
func Inverse(r Result) Result {
	return DefaultArithmetic.Inverse(r)
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

func (r *resultInverse) Release() {
	r.Cache.Free(r.OutputVec)
	r.OutputVec = nil
	r.Input.Release()
}

type rresultInverse struct {
	Cache      *VectorCache
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	SquaredOut linalg.Vector
	Input      RResult
}

// InverseR is like Inverse, but for RResults.
func (a *Arithmetic) InverseR(r RResult) RResult {
	inVec := r.Output()
	inVecR := r.ROutput()
	outVec := a.Cache.Alloc(len(inVec))
	outVecR := a.Cache.Alloc(len(inVec))
	squaredOut := a.Cache.Alloc(len(inVec))
	for i, x := range inVec {
		recip := 1 / x
		squared := recip * recip
		outVec[i] = recip
		squaredOut[i] = squared
		outVecR[i] = -squared * inVecR[i]
	}
	return &rresultInverse{
		Cache:      a.Cache,
		OutputVec:  outVec,
		ROutputVec: outVecR,
		SquaredOut: squaredOut,
		Input:      r,
	}
}

// InverseR is equivalent to DefaultArithmetic.InverseR.
func InverseR(r RResult) RResult {
	return DefaultArithmetic.InverseR(r)
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

func (r *rresultInverse) Release() {
	r.Cache.Free(r.OutputVec)
	r.Cache.Free(r.ROutputVec)
	r.Cache.Free(r.SquaredOut)
	r.OutputVec = nil
	r.ROutputVec = nil
	r.SquaredOut = nil
	r.Input.Release()
}

type resultPow struct {
	Cache     *VectorCache
	OutputVec linalg.Vector
	Power     float64
	Input     Result
}

// Pow raises each component of r to a given power.
func (a *Arithmetic) Pow(r Result, pow float64) Result {
	input := r.Output()
	output := a.Cache.Alloc(len(input))
	for i, x := range input {
		output[i] = math.Pow(x, pow)
	}
	return &resultPow{
		Cache:     a.Cache,
		OutputVec: output,
		Power:     pow,
		Input:     r,
	}
}

// Pow is equivalent to DefaultArithmetic.Pow.
func Pow(r Result, pow float64) Result {
	return DefaultArithmetic.Pow(r, pow)
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

func (r *resultPow) Release() {
	r.Cache.Free(r.OutputVec)
	r.OutputVec = nil
	r.Input.Release()
}

type rresultPow struct {
	Cache      *VectorCache
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Power      float64
	Input      RResult
}

// PowR is like Pow, but for RResults.
func (a *Arithmetic) PowR(r RResult, pow float64) RResult {
	input := r.Output()
	inputR := r.ROutput()
	output := a.Cache.Alloc(len(input))
	outputR := a.Cache.Alloc(len(input))
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
		Cache:      a.Cache,
		OutputVec:  output,
		ROutputVec: outputR,
		Power:      pow,
		Input:      r,
	}
}

// PowR is equivalent to DefaultArithmetic.PowR.
func PowR(r RResult, pow float64) RResult {
	return DefaultArithmetic.PowR(r, pow)
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

func (r *rresultPow) Release() {
	r.Cache.Free(r.OutputVec)
	r.Cache.Free(r.ROutputVec)
	r.OutputVec = nil
	r.ROutputVec = nil
	r.Input.Release()
}

type sumAllResult struct {
	Cache     *VectorCache
	OutputVec linalg.Vector
	Input     Result
}

// SumAll adds up all the components of r and
// returns a vector with that sum as its one
// and only element.
func (a *Arithmetic) SumAll(r Result) Result {
	var sum float64
	for _, x := range r.Output() {
		sum += x
	}
	return &sumAllResult{
		Cache:     a.Cache,
		OutputVec: linalg.Vector{sum},
		Input:     r,
	}
}

// SumAll is equivalent to DefaultArithmetic.SumAll.
func SumAll(r Result) Result {
	return DefaultArithmetic.SumAll(r)
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
		downstream := s.Cache.Alloc(inLen)
		for i := range downstream {
			downstream[i] = upstream[0]
		}
		s.Input.PropagateGradient(downstream, grad)
		s.Cache.Free(downstream)
	}
}

func (s *sumAllResult) Release() {
	s.Input.Release()
}

type sumAllRResult struct {
	Cache      *VectorCache
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      RResult
}

// SumAllR is like SumAll, but for an RResult.
func (a *Arithmetic) SumAllR(r RResult) RResult {
	var sum, rsum float64
	routput := r.ROutput()
	for i, x := range r.Output() {
		sum += x
		rsum += routput[i]
	}
	return &sumAllRResult{
		Cache:      a.Cache,
		OutputVec:  linalg.Vector{sum},
		ROutputVec: linalg.Vector{rsum},
		Input:      r,
	}
}

// SumAllR is equivalent to DefaultArithmetic.SumAllR.
func SumAllR(r RResult) RResult {
	return DefaultArithmetic.SumAllR(r)
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
		downstream := s.Cache.Alloc(inLen)
		downstreamR := s.Cache.Alloc(inLen)
		for i := range downstream {
			downstream[i] = upstream[0]
			downstreamR[i] = upstreamR[0]
		}
		s.Input.PropagateRGradient(downstream, downstreamR, rgrad, grad)
		s.Cache.Free(downstream)
		s.Cache.Free(downstreamR)
	}
}

func (s *sumAllRResult) Release() {
	s.Input.Release()
}
