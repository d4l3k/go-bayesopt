package plot

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"
	"sort"

	"github.com/pkg/errors"
	"github.com/wcharczuk/go-chart"

	"gonum.org/v1/gonum/floats"

	"github.com/d4l3k/go-bayesopt/gp"
)

// SaveAll saves all dimension graphs of the gaussian process to a temp
// directory and prints their file names.
func SaveAll(gp *gp.GP) (string, error) {
	dir, err := ioutil.TempDir("", "gp-plots")
	if err != nil {
		return "", err
	}
	dims := gp.Dims()
	for i := 0; i < dims; i++ {
		name := fmt.Sprintf("%d.svg", i)
		fpath := path.Join(dir, name)
		f, err := os.OpenFile(fpath, os.O_CREATE|os.O_WRONLY, 0755)
		if err != nil {
			return "", err
		}
		defer f.Close()
		if err := GP(gp, f, i); err != nil {
			return "", err
		}
		f.Close()
	}
	return dir, nil
}

// GP renders a plot of the gaussian process for the specified dimension.
func GP(gp *gp.GP, w io.Writer, dim int) error {
	dims := gp.Dims()
	if dim >= dims {
		return errors.Errorf("requested graph of dimension %d; only %d dimensions", dim, dims)
	}

	inputs, outputs := gp.RawData()

	type pair struct {
		x []float64
		y float64
	}

	pairs := make([]pair, len(inputs))
	for i := range pairs {
		pairs[i].x = inputs[i]
		pairs[i].y = outputs[i]
	}

	sort.Slice(pairs, func(a, b int) bool {
		return pairs[a].x[dim] < pairs[b].x[dim]
	})

	knownX := make([]float64, len(pairs))
	knownY := make([]float64, len(pairs))
	for i, p := range pairs {
		knownX[i] = p.x[dim]
		knownY[i] = p.y
	}

	graph := chart.Chart{
		Title:      fmt.Sprintf("%s vs. %s", gp.Name(dim), gp.OutputName()),
		TitleStyle: chart.StyleShow(),
		XAxis: chart.XAxis{
			Name:      gp.Name(dim),
			NameStyle: chart.StyleShow(),
			Style:     chart.StyleShow(),
		},
		YAxis: chart.YAxis{
			Name:      gp.OutputName(),
			NameStyle: chart.StyleShow(),
			Style:     chart.StyleShow(),
		},
		Background: chart.Style{
			Padding: chart.Box{
				Top:    20,
				Left:   20,
				Bottom: 20,
				Right:  20,
			},
		},
	}
	graph.Elements = []chart.Renderable{
		chart.Legend(&graph),
	}

	max := floats.Max(knownX)
	min := floats.Min(knownX)
	const steps = chart.DefaultChartWidth
	x := make([]float64, steps)
	means := make([]float64, steps)
	uppers := make([]float64, steps)
	lowers := make([]float64, steps)
	stepSize := (max - min) / steps

	pairI := 0

outer:
	for j := range x {
		xi := stepSize*float64(j) + min
		x[j] = xi

		var lowerPair, upperPair pair
		for upperPair.x == nil || upperPair.x[dim] < xi {
			if upperPair.x != nil {
				pairI += 1
			}
			if pairI+1 >= len(pairs) {
				break outer
			}
			lowerPair = pairs[pairI]
			upperPair = pairs[pairI+1]
		}

		mid := (xi - lowerPair.x[dim]) / (upperPair.x[dim] - lowerPair.x[dim])
		args := make([]float64, dims)
		floats.AddScaled(args, 1-mid, lowerPair.x)
		floats.AddScaled(args, mid, upperPair.x)
		mean, sd, err := gp.Estimate(args)
		if err != nil {
			return err
		}
		means[j] = mean
		uppers[j] = mean + sd
		lowers[j] = mean - sd
	}

	graph.Series = append(
		graph.Series,
		chart.ContinuousSeries{
			Name:    "Mean",
			XValues: x,
			YValues: means,
		},
		chart.ContinuousSeries{
			Name:    "+1σ",
			XValues: x,
			YValues: uppers,
		},
		chart.ContinuousSeries{
			Name:    "-1σ",
			XValues: x,
			YValues: lowers,
		},
	)

	graph.Series = append(
		graph.Series,
		chart.ContinuousSeries{
			Name:    "Known",
			XValues: knownX,
			YValues: knownY,
			Style: chart.Style{
				Show:        true,
				StrokeWidth: chart.Disabled,
				DotWidth:    5,
			},
		},
	)

	if err := graph.Render(chart.SVG, w); err != nil {
		return err
	}
	return nil
}
