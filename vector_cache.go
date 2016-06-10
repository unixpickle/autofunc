package autofunc

import (
	"sync"

	"github.com/unixpickle/num-analysis/linalg"
)

var DefaultVectorCache = NewVectorCache(1 << 20)

// A VectorCache is a concurrency-safe cache for
// reusing linalg.Vector instances.
// Using a VectorCache can greatly improve
// performance compared to pure make() calls.
type VectorCache struct {
	maxFloats int

	lock       sync.Mutex
	sizeCaches map[int][]linalg.Vector
	floatCount int
}

// NewVectorCache creates a VectorCache which will
// never store more than maxFloats float64s at once.
// If maxFloats is 0, the cache will be unlimited.
func NewVectorCache(maxFloats int) *VectorCache {
	return &VectorCache{
		maxFloats:  maxFloats,
		sizeCaches: map[int][]linalg.Vector{},
	}
}

// Alloc creates or reuses a 0-initialized vector.
//
// If v is nil, DefaultVectorCache is used.
func (v *VectorCache) Alloc(size int) linalg.Vector {
	if v == nil {
		v = DefaultVectorCache
	}

	v.lock.Lock()
	if len(v.sizeCaches[size]) == 0 {
		v.lock.Unlock()
		return make(linalg.Vector, size)
	}
	vecs := v.sizeCaches[size]
	res := vecs[len(vecs)-1]
	v.sizeCaches[size] = vecs[:len(vecs)-1]
	v.floatCount -= size
	v.lock.Unlock()
	for i := range res {
		res[i] = 0
	}
	return res
}

// Free gives a vector back to the cache for reuse.
//
// If v is nil, DefaultVectorCache is used.
// If vec is nil, this does nothing.
func (v *VectorCache) Free(vec linalg.Vector) {
	if vec == nil {
		return
	} else if v == nil {
		v = DefaultVectorCache
	}

	vec = vec[:cap(vec)]
	v.lock.Lock()
	defer v.lock.Unlock()
	if v.maxFloats != 0 && v.floatCount+len(vec) > v.maxFloats {
		return
	}
	v.sizeCaches[len(vec)] = append(v.sizeCaches[len(vec)], vec)
	v.floatCount += len(vec)
}

// FloatCount returns the current number of float64s
// allocated in this cache.
//
// If v is nil, DefaultVectorCache is used.
func (v *VectorCache) FloatCount() int {
	if v == nil {
		v = DefaultVectorCache
	}
	v.lock.Lock()
	defer v.lock.Unlock()
	return v.floatCount
}

// Clear clears the entire cache.
//
// If v is nil, DefaultVectorCache is used.
func (v *VectorCache) Clear() {
	if v == nil {
		v = DefaultVectorCache
	}
	v.lock.Lock()
	v.floatCount = 0
	v.sizeCaches = map[int][]linalg.Vector{}
	v.lock.Unlock()
}

// MaxFloats returns the maximum number of floats
// allowed in this cache.
//
// If v is nil, DefaultVectorCache is used.
func (v *VectorCache) MaxFloats() int {
	if v == nil {
		v = DefaultVectorCache
	}
	v.lock.Lock()
	defer v.lock.Unlock()
	return v.maxFloats
}

// SetMaxFloats changes the maximum number of floats
// allowed in this cache.
// If the threshold is lower than the current number
// of floats, vectors will be evicted.
//
// If v is nil, DefaultVectorCache is used.
func (v *VectorCache) SetMaxFloats(m int) {
	if v == nil {
		v = DefaultVectorCache
	}
	v.lock.Lock()
	defer v.lock.Unlock()

	v.maxFloats = m
	if v.floatCount > m {
		v.sizeCaches = map[int][]linalg.Vector{}
		v.floatCount = 0
	}
}

// UsageHistogram returns a mapping from vector sizes
// to the number of floats worth of memory reserved
// for that vector size.
//
// If v is nil, DefaultVectorCache is used.
func (v *VectorCache) UsageHistogram() map[int]int {
	if v == nil {
		v = DefaultVectorCache
	}
	v.lock.Lock()
	defer v.lock.Unlock()

	m := map[int]int{}
	for key, val := range v.sizeCaches {
		for _, slice := range val {
			m[key] += len(slice)
		}
	}

	return m
}
