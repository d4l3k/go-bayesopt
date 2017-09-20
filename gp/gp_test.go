package gp_test

import (
	"testing"

	"github.com/d4l3k/go-bayesopt/gp"
	"github.com/d4l3k/go-bayesopt/gp/plot"
	"github.com/gonum/floats"
)

func TestKnown(t *testing.T) {
	gp := gp.New(gp.MaternCov, 0)
	gp.Add([]float64{1}, 1)
	gp.Add([]float64{2}, 2)
	gp.Add([]float64{3}, 3)
	gp.Add([]float64{4}, 4)
	gp.Add([]float64{5}, 5)
	gp.Add([]float64{10}, 10)
	if _, err := plot.SaveAll(gp); err != nil {
		t.Fatal(err)
	}
	mean, variance, err := gp.Estimate([]float64{1})
	if err != nil {
		t.Fatal(err)
	}
	if !floats.EqualWithinAbs(mean, 1, 0.0001) {
		t.Fatalf("got mean = %f; not 1", mean)
	}
	if !floats.EqualWithinAbs(variance, 0, 0.0001) {
		t.Fatalf("got variance = %f; not 0", variance)
	}
}
