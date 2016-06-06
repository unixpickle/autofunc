package benchcblas

import (
	"testing"

	"github.com/gonum/blas/blas64"
	"github.com/gonum/blas/cgo"
	"github.com/unixpickle/autofunc/bench"
)

func BenchmarkLSTMForward(b *testing.B) {
	blas64.Use(cgo.Implementation{})
	bench.DefaultLSTMBenchmark.Run(b, false)
}

func BenchmarkLSTMBothWays(b *testing.B) {
	blas64.Use(cgo.Implementation{})
	bench.DefaultLSTMBenchmark.Run(b, true)
}
