package bayesopt

import "math/rand"

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

var _ Param = LinearParam{}

// LinearParam is a uniformly distributed parameter between Max and Min.
type LinearParam struct {
	Name     string
	Max, Min float64
}

// GetName implements Param.
func (p LinearParam) GetName() string {
	return p.Name
}

// GetMax implements Param.
func (p LinearParam) GetMax() float64 {
	return p.Max
}

// GetMin implements Param.
func (p LinearParam) GetMin() float64 {
	return p.Min
}

// Sample implements Param.
func (p LinearParam) Sample() float64 {
	return rand.Float64()*(p.Max-p.Min) + p.Min
}
