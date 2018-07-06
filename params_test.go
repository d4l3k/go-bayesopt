package bayesopt

import "testing"

func TestParams(t *testing.T) {
	t.Parallel()

	cases := []struct {
		p        Param
		name     string
		max, min float64
	}{
		{
			p: UniformParam{
				Name: "uniform",
				Max:  10,
				Min:  1,
			},
			name: "uniform",
			max:  10,
			min:  1,
		},
		{
			p: NormalParam{
				Name:   "normal",
				Max:    10,
				Min:    -10,
				Mean:   0,
				StdDev: 10,
			},
			name: "normal",
			max:  10,
			min:  -10,
		},
		{
			p: NormalParam{
				Name:   "normal",
				Max:    10,
				Min:    0,
				Mean:   1,
				StdDev: 5,
			},
			name: "normal",
			max:  10,
			min:  0,
		},
	}

	for i, c := range cases {
		{
			out := c.p.GetName()
			want := c.name
			if out != want {
				t.Errorf("%d. %+v.GetName() = %q; wanted %q", i, c.p, out, want)
			}
		}
		{
			out := c.p.GetMax()
			want := c.max
			if out != want {
				t.Errorf("%d. %+v.GetMax() = %v; wanted %v", i, c.p, out, want)
			}
		}
		{
			out := c.p.GetMin()
			want := c.min
			if out != want {
				t.Errorf("%d. %+v.GetMin() = %v; wanted %v", i, c.p, out, want)
			}
		}
		for j := 0; j < 1000; j++ {
			sample := c.p.Sample()
			if sample < c.min || sample > c.max {
				t.Errorf("%d. %+v.Sample() = %v; outside bounds", i, c.p, sample)
			}
		}
	}
}

func TestTruncateSample(t *testing.T) {
	t.Parallel()

	var count int

	cases := []struct {
		p     Param
		f     func() float64
		want  float64
		count int
	}{
		{
			p: UniformParam{"", 10, 5},
			f: func() float64 {
				return 0
			},
			want:  5,
			count: 1000,
		},
		{
			p: UniformParam{"", 10, 5},
			f: func() float64 {
				return 100
			},
			want:  10,
			count: 1000,
		},
		{
			p: UniformParam{"", 10, 5},
			f: func() float64 {
				return 7
			},
			want:  7,
			count: 1,
		},
		{
			p: UniformParam{"", 10, 5},
			f: func() float64 {
				return float64(count)
			},
			want:  5,
			count: 5,
		},
	}

	for i, c := range cases {
		count = 0
		out := truncateSample(c.p, func() float64 {
			count++
			return c.f()
		})
		if out != c.want {
			t.Errorf("%d. truncateSample(%+v, ...) = %f; not %f", i, c.p, out, c.want)
		}
		if count != c.count {
			t.Errorf("%d. count = %d; not %d", i, count, c.count)
		}
	}
}
