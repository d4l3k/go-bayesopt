package bayesopt

import (
	"sync"

	"github.com/pkg/errors"

	"gonum.org/v1/gonum/optimize"

	"github.com/d4l3k/go-bayesopt/gp"
)

const (
	// DefaultRounds is the default number of rounds to run.
	DefaultRounds = 20
	// DefaultRandomRounds is the default number of random rounds to run.
	DefaultRandomRounds = 5
	// DefaultMinimize is the default value of minimize.
	DefaultMinimize = true

	NumRandPoints = 100000
	NumGradPoints = 256
)

var (
	// DefaultExploration uses UCB with 95 confidence interval.
	DefaultExploration = UCB{Kappa: 1.96}
	// DefaultBarrierFunc sets the default barrier function to use.
	DefaultBarrierFunc = LogBarrier{}
)

// Optimizer is a blackbox gaussian process optimizer.
type Optimizer struct {
	mu struct {
		sync.Mutex
		gp                          *gp.GP
		params                      []Param
		round, randomRounds, rounds int
		exploration                 Exploration
		minimize                    bool
		barrierFunc                 BarrierFunc

		running        bool
		explorationErr error
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

// WithMinimize sets whether or not to minimize. Passing false, maximizes
// instead.
func WithMinimize(minimize bool) OptimizerOption {
	return func(o *Optimizer) {
		o.mu.minimize = minimize
	}
}

// WithBarrierFunc sets the barrier function to use.
func WithBarrierFunc(bf BarrierFunc) OptimizerOption {
	return func(o *Optimizer) {
		o.mu.barrierFunc = bf
	}
}

// New creates a new optimizer with the specified optimizable parameters and
// options.
func New(params []Param, opts ...OptimizerOption) *Optimizer {
	o := &Optimizer{}
	o.mu.gp = gp.New(gp.MaternCov{}, 0)
	o.mu.params = params

	// Set default values.
	o.mu.randomRounds = DefaultRandomRounds
	o.mu.rounds = DefaultRounds
	o.mu.exploration = DefaultExploration
	o.mu.minimize = DefaultMinimize
	o.mu.barrierFunc = DefaultBarrierFunc

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

func isFatalErr(err error) bool {
	if err == nil {
		return false
	}

	// Only recurse 100 times before breaking.
	for i := 0; i < 100; i++ {
		parent := errors.Cause(err)
		if parent == err {
			break
		}
		err = parent
	}

	if _, ok := err.(optimize.ErrFunc); ok {
		return false
	}
	switch err {
	case optimize.ErrLinesearcherFailure, optimize.ErrNoProgress:
		return false
	default:
		return true
	}
}

// Next returns the next best x values to explore. If more than rounds have
// elapsed, nil is returned. If parallel is true, that round can happen in
// parallel to other rounds.
func (o *Optimizer) Next() (x map[Param]float64, parallel bool, err error) {
	o.mu.Lock()
	defer o.mu.Unlock()

	// Return if we've exceeded max # of rounds, or if there was an error while
	// doing exploration which is likely caused by numerical precision errors.
	if o.mu.round >= o.mu.rounds || o.mu.explorationErr != nil {
		return nil, false, nil
	}

	// If we don't have enough random rounds, run more.
	if o.mu.round < o.mu.randomRounds {
		x = sampleParamsMap(o.mu.params)
		o.mu.round += 1
		// Don't return parallel on the last random round.
		return x, o.mu.round != o.mu.randomRounds, nil
	}

	var fErr error
	f := func(x []float64) float64 {
		v, err := o.mu.exploration.Estimate(o.mu.gp, o.mu.minimize, x)
		if err != nil {
			fErr = errors.Wrap(err, "exploration error")
		}

		if o.mu.minimize {
			return v
		}
		return -v
	}
	problem := optimize.Problem{
		Func: f,
		Grad: func(grad, x []float64) {
			g, err := o.mu.gp.Gradient(x)
			if err != nil {
				fErr = errors.Wrap(err, "gradient error")
			}
			copy(grad, g)
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
		return nil, false, errors.Wrapf(err, "random sample failed")
	}
	if fErr != nil {
		o.mu.explorationErr = fErr
	}
	min := result.F
	minX := result.X

	// Run gradient descent on the best point.
	method := optimize.LBFGS{}
	grad := BoundsMethod{
		Method: &method,
		Bounds: o.mu.params,
	}
	// TODO(d4l3k): Bounded line searcher.
	{
		result, err := optimize.Local(problem, minX, nil, grad)
		if isFatalErr(err) {
			o.mu.explorationErr = errors.Wrapf(err, "random sample optimize failed")
		}
		if fErr != nil {
			o.mu.explorationErr = fErr
		}
		if result != nil && result.F < min {
			min = result.F
			minX = result.X
		}
	}

	// Attempt to use gradient descent on random points.
	for i := 0; i < NumGradPoints; i++ {
		x := sampleParams(o.mu.params)
		result, err := optimize.Local(problem, x, nil, grad)
		if isFatalErr(err) {
			o.mu.explorationErr = errors.Wrapf(err, "gradient descent failed: i %d, x %+v, result%+v", i, x, result)
		}
		if fErr != nil {
			o.mu.explorationErr = fErr
		}
		if result != nil && result.F < min {
			min = result.F
			minX = result.X
		}
	}

	if o.mu.explorationErr != nil {
		return nil, false, nil
	}

	m := map[Param]float64{}
	for i, x := range minX {
		m[o.mu.params[i]] = x
	}

	o.mu.round += 1
	return m, false, nil
}

func (o *Optimizer) ExplorationErr() error {
	o.mu.Lock()
	defer o.mu.Unlock()

	return o.mu.explorationErr
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
func (o *Optimizer) Optimize(f func(map[Param]float64) float64) (x map[Param]float64, y float64, err error) {
	o.mu.Lock()
	if o.mu.running {
		o.mu.Unlock()
		return nil, 0, errors.New("optimizer is already running")
	}
	o.mu.running = true
	o.mu.Unlock()

	var wg sync.WaitGroup
	for {
		if !o.Running() {
			return nil, 0, errors.New("optimizer got stop signal")
		}

		x, parallel, err := o.Next()
		if err != nil {
			return nil, 0, errors.Wrapf(err, "failed to get next point")
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

	var xa []float64
	if o.mu.minimize {
		xa, y = o.mu.gp.Minimum()
	} else {
		xa, y = o.mu.gp.Maximum()
	}
	x = map[Param]float64{}
	for i, v := range xa {
		x[o.mu.params[i]] = v
	}

	return x, y, nil
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

// Rounds is the number of rounds that have been run.
func (o *Optimizer) Rounds() int {
	o.mu.Lock()
	defer o.mu.Unlock()

	return o.mu.round
}
