package plot_test

import (
	"fmt"
	"os"
	"path"
	"testing"

	"github.com/d4l3k/go-bayesopt/gp"
	"github.com/d4l3k/go-bayesopt/gp/plot"
)

func TestPlot(t *testing.T) {
	gp := gp.New(gp.MaternCov, 0)
	gp.Add([]float64{1, 5}, 1)
	gp.Add([]float64{2, 4}, 2)
	gp.Add([]float64{3, 3}, 3)
	gp.Add([]float64{4, 2}, 4)
	gp.Add([]float64{5, 1}, 5)
	gp.Add([]float64{10, -5}, 10)

	dir, err := plot.SaveAll(gp)
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	for i := 0; i < 2; i++ {
		fpath := path.Join(dir, fmt.Sprintf("%d.svg", i))
		info, err := os.Stat(fpath)
		if err != nil {
			t.Fatal(err)
		}
		if info.Size() == 0 {
			t.Errorf("%q: size = 0", fpath)
		}
	}
}
