package bayesopt

import (
	"gonum.org/v1/gonum/optimize"
)

type BoundsMethod struct {
	Method optimize.Method
	Bounds []Param
}

func (m BoundsMethod) constrain(loc *optimize.Location) {
	for i, param := range m.Bounds {
		max := param.GetMax()
		min := param.GetMin()
		if loc.X[i] > max {
			loc.X[i] = max
		} else if loc.X[i] < min {
			loc.X[i] = min
		}
	}
}

func (m BoundsMethod) Init(loc *optimize.Location) (optimize.Operation, error) {
	m.constrain(loc)
	op, err := m.Method.Init(loc)
	m.constrain(loc)
	return op, err
}

func (m BoundsMethod) Iterate(loc *optimize.Location) (optimize.Operation, error) {
	m.constrain(loc)
	op, err := m.Method.Iterate(loc)
	m.constrain(loc)
	return op, err
}

func (m BoundsMethod) Needs() struct {
	Gradient bool
	Hessian  bool
} {
	return m.Method.Needs()
}
