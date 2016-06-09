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
// If v or vec are nil, this does nothing.
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
func (v *VectorCache) FloatCount() int {
	v.lock.Lock()
	defer v.lock.Unlock()
	return v.floatCount
}

// Clear clears the entire cache.
func (v *VectorCache) Clear() {
	v.lock.Lock()
	v.floatCount = 0
	v.sizeCaches = map[int][]linalg.Vector{}
	v.lock.Unlock()
}
