package bench

import "testing"

func BenchmarkLSTMForward(b *testing.B) {
	DefaultLSTMBenchmark.Run(b, false)
}

func BenchmarkLSTMBothWays(b *testing.B) {
	DefaultLSTMBenchmark.Run(b, true)
}
