package gp_test

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/floats"

	"github.com/d4l3k/go-bayesopt/gp"
	"github.com/d4l3k/go-bayesopt/gp/plot"
)

func f(x, y float64) float64 {
	return math.Cos(x/2)/2 + math.Sin(y/4)
}

func gpAdd(gp *gp.GP, x, y float64) {
	gp.Add([]float64{x, y}, f(x, y))
}

func TestKnown(t *testing.T) {
	gp := gp.New(gp.MaternCov, 0)

	gpAdd(gp, 0.25, 0.75)

	for i := 0; i < 20; i++ {
		gpAdd(gp, rand.Float64()*2*math.Pi-math.Pi, rand.Float64()*2*math.Pi-math.Pi)
	}

	if _, err := plot.SaveAll(gp); err != nil {
		t.Fatal(err)
	}
	mean, variance, err := gp.Estimate([]float64{0.25, 0.75})
	if err != nil {
		t.Fatal(err)
	}
	if !floats.EqualWithinAbs(mean, f(0.25, 0.75), 0.0001) {
		t.Fatalf("got mean = %f; not 1", mean)
	}
	if !floats.EqualWithinAbs(variance, 0, 0.0001) {
		t.Fatalf("got variance = %f; not 0", variance)
	}
}
