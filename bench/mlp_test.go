package bench

import "testing"

func BenchmarkMLPForward(b *testing.B) {
	DefaultMLPBenchmark.Run(b, false)
}

func BenchmarkMLPBothWays(b *testing.B) {
	DefaultMLPBenchmark.Run(b, true)
}

func BenchmarkMLPHessian(b *testing.B) {
	DefaultMLPBenchmark.RunR(b, true)
}
