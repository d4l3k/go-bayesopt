package bayesopt

import (
	"errors"
	"log"
	"sync"

	"github.com/d4l3k/go-bayesopt/gp"
	"github.com/gonum/optimize"
	"gonum.org/v1/gonum/diff/fd"
)

const (
	// DefaultRounds is the default number of rounds to run.
	DefaultRounds = 20
	// DefaultRandomRounds is the default number of random rounds to run.
	DefaultRandomRounds = 5

	NumRandPoints = 100000
	NumGradPoints = 256
)

var (
	// DefaultExploration uses UCB with 95 confidence interval.
	DefaultExploration = UCB{Kappa: 1.96}
)

// Optimizer is a blackbox gaussian process optimizer.
type Optimizer struct {
	mu struct {
		sync.Mutex
		gp                          *gp.GP
		params                      []Param
		round, randomRounds, rounds int
		exploration                 Exploration

		running bool
	}
}

// OptimizerOption sets an option on the optimizer.
type OptimizerOption func(*Optimizer)

// WithOutputName sets the outputs name. Only really matters if you're planning
// on using gp/plot.
func WithOutputName(name string) OptimizerOption {
	return func(o *Optimizer) {
		o.updateNames(name)
	}
}

// WithRandomRounds sets the number of random rounds to run.
func WithRandomRounds(rounds int) OptimizerOption {
	return func(o *Optimizer) {
		o.mu.randomRounds = rounds
	}
}

// WithRounds sets the total number of rounds to run.
func WithRounds(rounds int) OptimizerOption {
	return func(o *Optimizer) {
		o.mu.rounds = rounds
	}
}

// WithExploration sets the exploration function to use.
func WithExploration(exploration Exploration) OptimizerOption {
	return func(o *Optimizer) {
		o.mu.exploration = exploration
	}
}

// New creates a new optimizer with the specified optimizable parameters and
// options.
func New(params []Param, opts ...OptimizerOption) *Optimizer {
	o := &Optimizer{}
	o.mu.gp = gp.New(gp.MaternCov, 0)
	o.mu.params = params

	// Set default values.
	o.mu.randomRounds = DefaultRandomRounds
	o.mu.rounds = DefaultRounds
	o.mu.exploration = DefaultExploration

	o.updateNames("")

	for _, opt := range opts {
		opt(o)
	}

	return o
}

// updateNames sets the gaussian process names.
func (o *Optimizer) updateNames(outputName string) {
	o.mu.Lock()
	defer o.mu.Unlock()

	var inputNames []string
	for _, p := range o.mu.params {
		inputNames = append(inputNames, p.GetName())
	}
	o.mu.gp.SetNames(inputNames, outputName)
}

// GP returns the underlying gaussian process. Primary for use with plotting
// behavior.
func (o *Optimizer) GP() *gp.GP {
	o.mu.Lock()
	defer o.mu.Unlock()

	return o.mu.gp
}

func sampleParams(params []Param) []float64 {
	x := make([]float64, len(params))
	for i, p := range params {
		x[i] = p.Sample()
	}
	return x
}

func sampleParamsMap(params []Param) map[Param]float64 {
	x := map[Param]float64{}
	for i, v := range sampleParams(params) {
		x[params[i]] = v
	}
	return x
}

type randerFunc func([]float64) []float64

func (f randerFunc) Rand(x []float64) []float64 {
	return f(x)
}

// Next returns the next best x values to explore. If more than rounds have
// elapsed, nil is returned. If parallel is true, that round can happen in
// parallel to other rounds.
func (o *Optimizer) Next() (x map[Param]float64, parallel bool, err error) {
	o.mu.Lock()
	defer o.mu.Unlock()

	if o.mu.round >= o.mu.rounds {
		return nil, false, nil
	}

	// If we don't have enough random rounds, run more.
	if o.mu.round < o.mu.randomRounds {
		x = sampleParamsMap(o.mu.params)
		o.mu.round += 1
		return x, true, nil
	}

	f := func(x []float64) float64 {
		v, err := o.mu.exploration.Estimate(o.mu.gp, x)
		if err != nil {
			log.Printf("error %+v", err)
		}
		return v
	}
	problem := optimize.Problem{
		Func: f,
		Grad: func(grad, x []float64) {
			fd.Gradient(grad, f, x, nil)
		},
	}

	// Randomly query a bunch of points to get a good estimate of maximum.
	result, err := optimize.Global(problem, len(o.mu.params), &optimize.Settings{
		FuncEvaluations: NumRandPoints,
	}, &optimize.GuessAndCheck{
		Rander: randerFunc(func(x []float64) []float64 {
			return sampleParams(o.mu.params)
		}),
	})
	if err != nil {
		return nil, false, err
	}
	min := result.F
	minX := result.X

	// Run gradient descent on the best point.
	grad := optimize.LBFGS{}
	{
		result, err := optimize.Local(problem, minX, nil, &grad)
		if err != nil {
			return nil, false, err
		}
		if result.F < min {
			min = result.F
			minX = result.X
		}
	}

	// Attempt to use gradient descent on random points.
	for i := 0; i < NumGradPoints; i++ {
		x := sampleParams(o.mu.params)
		result, err := optimize.Local(problem, x, nil, &grad)
		if err != nil {
			return nil, false, err
		}
		if result.F < min {
			min = result.F
			minX = result.X
		}
	}

	m := map[Param]float64{}
	for i, x := range minX {
		m[o.mu.params[i]] = x
	}

	return m, false, nil
}

func (o *Optimizer) Log(x map[Param]float64, y float64) {
	o.mu.Lock()
	defer o.mu.Unlock()

	var xa []float64
	for _, p := range o.mu.params {
		xa = append(xa, x[p])
	}
	o.mu.gp.Add(xa, y)
}

// Optimize will call f the fewest times as possible while trying to maximize
// the output value. It blocks until all rounds have elapsed, or Stop is called.
func (o *Optimizer) Optimize(f func(map[Param]float64) float64) error {
	o.mu.Lock()
	if o.mu.running {
		o.mu.Unlock()
		return errors.New("optimizer is already running")
	}
	o.mu.running = true
	o.mu.Unlock()

	var wg sync.WaitGroup
	for {
		if !o.Running() {
			return errors.New("optimizer got stop signal")
		}

		x, parallel, err := o.Next()
		if err != nil {
			return err
		}
		if x == nil {
			break
		}
		if parallel {
			wg.Add(1)
			go func() {
				defer wg.Done()

				o.Log(x, f(x))
			}()
		} else {
			wg.Wait()
			o.Log(x, f(x))
		}
	}

	o.mu.Lock()
	o.mu.running = false
	o.mu.Unlock()

	return nil
}

// Stop stops Optimize.
func (o *Optimizer) Stop() {
	o.mu.Lock()
	defer o.mu.Unlock()

	o.mu.running = false
}

// Running returns whether or not the optimizer is running.
func (o *Optimizer) Running() bool {
	o.mu.Lock()
	defer o.mu.Unlock()

	return o.mu.running
}
