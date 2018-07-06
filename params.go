package bayesopt

import (
	"math"
	"math/rand"
)

// SampleTries is the number of tries a sample function should try before
// truncating the samples to the boundaries.
var SampleTries = 1000

// Param represents a parameter that can be optimized.
type Param interface {
	// GetName returns the name of the parameter.
	GetName() string
	// GetMax returns the maximum value.
	GetMax() float64
	// GetMin returns the minimum value.
	GetMin() float64
	// Sample returns a random point within the bounds. It doesn't have to be
	// uniformly distributed.
	Sample() float64
}

var _ Param = UniformParam{}

// LinearParam is a UniformParam. Deprecated.
type LinearParam = UniformParam

// UniformParam is a uniformly distributed parameter between Max and Min.
type UniformParam struct {
	Name     string
	Max, Min float64
}

// GetName implements Param.
func (p UniformParam) GetName() string {
	return p.Name
}

// GetMax implements Param.
func (p UniformParam) GetMax() float64 {
	return p.Max
}

// GetMin implements Param.
func (p UniformParam) GetMin() float64 {
	return p.Min
}

// Sample implements Param.
func (p UniformParam) Sample() float64 {
	return rand.Float64()*(p.Max-p.Min) + p.Min
}

var _ Param = NormalParam{}

// NormalParam is a normally distributed parameter with Mean and StdDev between
// Max and Min. The Max and Min parameters use discard sampling to find a point
// between them. Set them to be math.Inf(1) and math.Inf(-1) to disable the
// bounds.
type NormalParam struct {
	Name         string
	Max, Min     float64
	Mean, StdDev float64
}

// GetName implements Param.
func (p NormalParam) GetName() string {
	return p.Name
}

// GetMax implements Param.
func (p NormalParam) GetMax() float64 {
	return p.Max
}

// GetMin implements Param.
func (p NormalParam) GetMin() float64 {
	return p.Min
}

// Sample implements Param.
func (p NormalParam) Sample() float64 {
	return truncateSample(p, func() float64 {
		return rand.NormFloat64()*p.StdDev + p.Mean
	})
}

func truncateSample(p Param, f func() float64) float64 {
	max := p.GetMax()
	min := p.GetMin()

	var sample float64
	for i := 0; i < SampleTries; i++ {
		sample = f()
		if sample >= min && sample <= max {
			return sample
		}
	}
	return math.Min(math.Max(sample, min), max)
}
