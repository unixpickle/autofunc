package benchcblas

import (
	"testing"

	"github.com/gonum/blas/blas64"
	"github.com/gonum/blas/cgo"
	"github.com/unixpickle/autofunc/bench"
)

func BenchmarkLinTranForward(b *testing.B) {
	blas64.Use(cgo.Implementation{})
	bench.DefaultLinTranBenchmark.Run(b, false)
}

func BenchmarkLinTranBackward(b *testing.B) {
	blas64.Use(cgo.Implementation{})
	bench.DefaultLinTranBenchmark.Run(b, true)
}

func BenchmarkLinTranForwardR(b *testing.B) {
	blas64.Use(cgo.Implementation{})
	bench.DefaultLinTranBenchmark.RunR(b, false)
}

func BenchmarkLinTranBackwardR(b *testing.B) {
	blas64.Use(cgo.Implementation{})
	bench.DefaultLinTranBenchmark.RunR(b, true)
}
