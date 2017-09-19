package gp

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Cov calculates the covariance between a and b.
type Cov func(a, b []float64) float64

// MaternCov calculates the covariance between a and b. nu = 2.5
// https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function#Simplification_for_.CE.BD_half_integer
func MaternCov(a, b []float64) float64 {
	const p = 2

	av := mat.NewVecDense(len(a), a)
	bv := mat.NewVecDense(len(b), b)
	diff := mat.NewVecDense(len(a), nil)
	diff.SubVec(av, bv)
	d := math.Sqrt(mat.Dot(diff, diff))

	return (1 + math.Sqrt(5)*d/p + 5*d*d/(3*p*p)) * math.Exp(-math.Sqrt(5)*d/p)
}
