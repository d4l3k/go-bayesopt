package gp

import (
	"math"

	"github.com/gonum/floats"
)

// Cov calculates the covariance between a and b.
type Cov func(a, b []float64) float64

// MaternCov calculates the covariance between a and b. nu = 2.5
// https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function#Simplification_for_.CE.BD_half_integer
func MaternCov(a, b []float64) float64 {
	const p = 2
	d := floats.Distance(a, b, 2)
	return (1 + math.Sqrt(5)*d/p + 5*d*d/(3*p*p)) * math.Exp(-math.Sqrt(5)*d/p)
}
