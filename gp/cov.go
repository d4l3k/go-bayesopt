package gp

import (
	"math"

	"gonum.org/v1/gonum/floats"
)

// Cov calculates the covariance between a and b.
type Cov interface {
	Cov(a, b []float64) float64
	Grad(a, b []float64) []float64
}

// MaternCov calculates the covariance between a and b. nu = 2.5
// https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function#Simplification_for_.CE.BD_half_integer
type MaternCov struct{}

func (MaternCov) Cov(a, b []float64) float64 {
	const p = 2
	d := floats.Distance(a, b, 2)
	return (1 + math.Sqrt(5)*d/p + 5*d*d/(3*p*p)) * math.Exp(-math.Sqrt(5)*d/p)
}

// Grad computes the gradient of the matern covariance between a
// and b with respect to a. nu = 2.5.
func (MaternCov) Grad(a, b []float64) []float64 {
	d2 := floats.Distance(a, b, 2)
	d := make([]float64, len(a))
	floats.Add(d, a)
	floats.Sub(d, b)
	/*
		tmp := math.Sqrt(5 * floats.Sum(d))
		floats.Scale(5.0/3.0*(tmp+1)*math.Exp(-tmp), d)
	*/

	floats.Scale(math.Sqrt(5)+5.0/3.0*d2+math.Sqrt(5)*math.Exp(-math.Sqrt(5)/2.0*d2), d)
	return d
}
