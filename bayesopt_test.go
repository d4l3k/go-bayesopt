package bayesopt

import (
	"math"
	"testing"

	"github.com/d4l3k/go-bayesopt/gp/plot"
	"gonum.org/v1/gonum/floats"
)

func TestOptimizer(t *testing.T) {
	X := LinearParam{
		Max: 10,
		Min: -10,
	}
	o := New(
		[]Param{
			X,
		},
	)
	x, y, err := o.Optimize(func(params map[Param]float64) float64 {
		return math.Pow(params[X], 2) + 1
	})
	if err != nil {
		t.Errorf("%+v", err)
	}

	{
		x, y := o.GP().RawData()
		t.Logf("x %+v\ny %+v", x, y)
	}
	if _, err := plot.SaveAll(o.GP()); err != nil {
		t.Errorf("plot error: %+v", err)
	}

	{
		got := x[X]
		want := 0.0
		if !floats.EqualWithinAbs(got, want, 0.0001) {
			t.Errorf("got x = %f; not %f", got, want)
		}
	}
	{
		got := y
		want := 1.0
		if !floats.EqualWithinAbs(got, want, 0.0001) {
			t.Errorf("got x = %f; not %f", got, want)
		}
	}
}
