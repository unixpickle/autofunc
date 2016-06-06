package benchcblas

import (
	"testing"

	"github.com/gonum/blas/blas64"
	"github.com/gonum/blas/cgo"
	"github.com/unixpickle/autofunc/bench"
)

func BenchmarkMLPForward(b *testing.B) {
	blas64.Use(cgo.Implementation{})
	bench.DefaultMLPBenchmark.Run(b, false)
}

func BenchmarkMLPBothWays(b *testing.B) {
	blas64.Use(cgo.Implementation{})
	bench.DefaultMLPBenchmark.Run(b, true)
}

func BenchmarkMLPHessian(b *testing.B) {
	blas64.Use(cgo.Implementation{})
	bench.DefaultMLPBenchmark.RunR(b, true)
}
