package bayesopt

import (
	"math"

	"github.com/d4l3k/go-bayesopt/gp"
)

// Exploration is the strategy to use for exploring the Gaussian process.
type Exploration interface {
	Estimate(gp *gp.GP, minimize bool, x []float64) (float64, error)
}

// UCB implements upper confidence bound exploration.
type UCB struct {
	Kappa float64
}

// Estimate implements Exploration.
func (e UCB) Estimate(gp *gp.GP, minimize bool, x []float64) (float64, error) {
	mean, sd, err := gp.Estimate(x)
	if err != nil {
		return 0, err
	}
	if minimize {
		return mean - e.Kappa*sd, nil
	}
	return mean + e.Kappa*sd, nil
}

// BarrierFunc returns a value that is added to the value to bound the
// optimization.
type BarrierFunc func(x []float64, params []Param) float64

// BasicBarrier returns -Inf if an x value is outside the param range.
func BasicBarrier(x []float64, params []Param) float64 {
	for i, p := range params {
		v := x[i]
		if v < p.GetMin() || v > p.GetMax() {
			return math.Inf(-1)
		}
	}
	return 0
}

// LogBarrier implements a logarithmic barrier function.
func LogBarrier(x []float64, params []Param) float64 {
	v := 0.0
	for i, p := range params {
		v += math.Log2(p.GetMax() - x[i])
		v += math.Log2(x[i] - p.GetMin())
	}
	if math.IsNaN(v) {
		return math.Inf(-1)
	}
	return v
}
