package autofunc

// A Batcher is like a Func, except it can be run
// on multiple inputs simultaneously.
type Batcher interface {
	// Batch evaluates the function on n inputs, where
	// the inputs are side-by-side in the inputs.Output()
	// vector.
	// Outputs are packed the same way as inputs, where
	// the first group of entries corresponds to the first
	// output, the next group to the next output, etc.
	//
	// For example, if two vectors were {1,2} and {3,4},
	// the packed vector containing these two vectors
	// would be {1,2,3,4}.
	//
	// The n argument must evenly divide the lengths of
	// the input and output vectors.
	Batch(inputs Result, n int) Result
}

// An RBatcher is like a Batcher, but it can also
// operate on RResults.
type RBatcher interface {
	Batcher

	// BatchR is like Batcher.Batch, but for RResults.
	BatchR(v RVector, inputs RResult, n int) RResult
}

// A FuncBatcher converts a Func to a Batcher by
// naively splitting and re-joining batched inputs
// and outputs respectively.
type FuncBatcher struct {
	F Func

	Cache *VectorCache
}

func (f *FuncBatcher) Batch(in Result, n int) Result {
	inLen := len(in.Output())
	if inLen%n != 0 {
		panic("input count does not divide input length")
	}
	sampleLen := inLen / n

	return Pool(in, func(in Result) Result {
		results := make([]Result, n)
		for i := range results {
			startIdx := i * sampleLen
			slicedIn := SliceCache(f.Cache, in, startIdx, startIdx+sampleLen)
			results[i] = f.F.Apply(slicedIn)
		}
		return ConcatCache(f.Cache, results...)
	})
}

// An RFuncBatcher is like a FuncBatcher, but for RFuncs.
type RFuncBatcher struct {
	F RFunc

	Cache *VectorCache
}

func (f *RFuncBatcher) Batch(in Result, n int) Result {
	b := FuncBatcher{F: f.F, Cache: f.Cache}
	return b.Batch(in, n)
}

func (f *RFuncBatcher) BatchR(v RVector, in RResult, n int) RResult {
	inLen := len(in.Output())
	if inLen%n != 0 {
		panic("input count does not divide input length")
	}
	sampleLen := inLen / n

	return PoolR(in, func(in RResult) RResult {
		results := make([]RResult, n)
		for i := range results {
			startIdx := i * sampleLen
			slicedIn := SliceCacheR(f.Cache, in, startIdx, startIdx+sampleLen)
			results[i] = f.F.ApplyR(v, slicedIn)
		}
		return ConcatCacheR(f.Cache, results...)
	})
}

// A ComposedBatcher is a Batcher which propagates
// through a list of sub-Batchers.
type ComposedBatcher []Batcher

func (c ComposedBatcher) Batch(in Result, n int) Result {
	for _, b := range c {
		in = b.Batch(in, n)
	}
	return in
}

// A ComposedRBatcher is an RBatcher which propagates
// through a list of sub-RBatchers.
type ComposedRBatcher []RBatcher

func (c ComposedRBatcher) Batch(in Result, n int) Result {
	for _, b := range c {
		in = b.Batch(in, n)
	}
	return in
}

func (c ComposedRBatcher) BatchR(v RVector, in RResult, n int) RResult {
	for _, b := range c {
		in = b.BatchR(v, in, n)
	}
	return in
}
