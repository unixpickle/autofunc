package bench

import "testing"

func BenchmarkLinTranForward(b *testing.B) {
	DefaultLinTranBenchmark.Run(b, false)
}

func BenchmarkLinTranBackward(b *testing.B) {
	DefaultLinTranBenchmark.Run(b, true)
}

func BenchmarkLinTranForwardR(b *testing.B) {
	DefaultLinTranBenchmark.RunR(b, false)
}

func BenchmarkLinTranBackwardR(b *testing.B) {
	DefaultLinTranBenchmark.RunR(b, true)
}
