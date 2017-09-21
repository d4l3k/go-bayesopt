package bayesopt

import (
	"math"
	"testing"

	"github.com/d4l3k/go-bayesopt/gp/plot"
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
	err := o.Optimize(func(params map[Param]float64) float64 {
		return math.Pow(params[X], 2)
	})

	t.Log("done")

	x, y := o.GP().RawData()
	t.Logf("x %+v\ny %+v", x, y)
	if _, err := plot.SaveAll(o.GP()); err != nil {
		t.Errorf("plot error: %+v", err)
	}

	if err != nil {
		t.Logf("%+v", err)
	}

	if IsFatalErr(err) {
		t.Fatalf("%+v", err)
	}
}
