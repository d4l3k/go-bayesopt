// gp is a library for computing Gaussian processes in Go/Golang.
// Algorithm adapted from:
// http://www.gaussianprocess.org/gpml/chapters/RW2.pdf
package gp

import (
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// GP represents a gaussian process.
type GP struct {
	inputs  [][]float64
	outputs []float64
	cov     Cov
	noise   float64

	alpha        *mat.VecDense
	l            *mat.Cholesky
	mean, stddev float64
	dirty        bool
}

// New creates a new Gaussian process with the specified covariance function
// (cov) and noise level (variance).
func New(cov Cov, noise float64) *GP {
	return &GP{
		cov:   cov,
		noise: noise,
	}
}

// Add bulk adds XY pairs.
func (gp *GP) Add(x []float64, y float64) {
	gp.dirty = true
	gp.inputs = append(gp.inputs, x)
	gp.outputs = append(gp.outputs, y)
}

func (gp *GP) compute() error {
	n := len(gp.inputs)
	k := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			v := gp.cov(gp.inputs[i], gp.inputs[j])
			if i == j {
				v += gp.noise
			}
			k.SetSym(i, j, v)
		}
	}
	var L mat.Cholesky
	if ok := L.Factorize(k); !ok {
		return errors.Errorf("failed to factorize")
	}
	b := mat.NewVecDense(n, gp.normOutputs())
	gp.alpha = mat.NewVecDense(n, nil)
	if err := L.SolveVec(gp.alpha, b); err != nil {
		return err
	}
	gp.l = &L
	gp.dirty = false
	return nil
}

func (gp *GP) normOutputs() []float64 {
	gp.mean, gp.stddev = stat.MeanStdDev(gp.outputs, nil)
	out := make([]float64, len(gp.outputs))
	for i, v := range gp.outputs {
		out[i] = (v - gp.mean) / gp.stddev
	}
	return out
}

// Estimate returns the mean and variance at the point x.
func (gp *GP) Estimate(x []float64) (float64, float64, error) {
	if gp.dirty {
		if err := gp.compute(); err != nil {
			return 0, 0, err
		}
	}
	n := len(gp.inputs)
	kstar := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		kstar.SetVec(i, gp.cov(gp.inputs[i], x))
	}
	mean := (mat.Dot(kstar, gp.alpha) + gp.mean) * gp.stddev
	v := mat.NewVecDense(n, nil)
	if err := gp.l.SolveVec(v, kstar); err != nil {
		return 0, 0, err
	}
	variance := gp.cov(x, x) - mat.Dot(v, v)
	return mean, variance, nil
}
