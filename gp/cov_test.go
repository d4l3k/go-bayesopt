package gp

import (
	"math"
	"testing"
)

func TestMaternCov(t *testing.T) {
	cases := []struct {
		a, b []float64
		want float64
	}{
		{
			[]float64{0},
			[]float64{0},
			1,
		},
		{
			[]float64{0, 1, 3},
			[]float64{0, 1, 2},
			0.828649,
		},
		{
			[]float64{0, 1, 4},
			[]float64{0, 1, 2},
			0.523994,
		},
	}
	for i, c := range cases {
		out := MaternCov{}.Cov(c.a, c.b)
		if math.Abs(out-c.want) > 0.00001 {
			t.Errorf("%d. MaternCov(%+v, %+v) = %f; not %f", i, c.a, c.b, out, c.want)
		}
	}
}
