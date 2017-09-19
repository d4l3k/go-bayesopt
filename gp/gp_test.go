package gp

import "testing"

func TestKnown(t *testing.T) {
	gp := New(MaternCov, 0)
	gp.Add([]float64{1}, 1)
	//gp.Add([]float64{1}, 1)
	//gp.Add([]float64{2}, 2)
	mean, variance, err := gp.Estimate([]float64{1})
	if err != nil {
		t.Fatal(err)
	}
	if mean != 1 {
		t.Fatalf("got mean = %f; not 1", mean)
	}
	if variance != 0 {
		t.Fatalf("got variance = %f; not 0", variance)
	}
}
