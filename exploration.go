package bayesopt

import "github.com/d4l3k/go-bayesopt/gp"

type Exploration interface {
	Estimate(gp *gp.GP, x []float64) (float64, error)
}

// UCB implements upper confidence bound exploration.
type UCB struct {
	Kappa float64
}

func (e UCB) Estimate(gp *gp.GP, x []float64) (float64, error) {
	mean, sd, err := gp.Estimate(x)
	if err != nil {
		return 0, err
	}
	return mean + e.Kappa*sd, nil
}
