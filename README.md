# go-bayesopt [![Build
Status](https://travis-ci.org/d4l3k/go-bayesopt.svg?branch=master)](https://travis-ci.org/d4l3k/go-bayesopt)
A library for doing Bayesian Optimization using Gaussian Processes (blackbox optimizer) in Go/Golang.

## How does it work?

From https://github.com/fmfn/BayesianOptimization:

Bayesian optimization works by constructing a posterior distribution of functions (gaussian process) that best describes the function you want to optimize. As the number of observations grows, the posterior distribution improves, and the algorithm becomes more certain of which regions in parameter space are worth exploring and which are not, as seen in the picture below.

![BayesianOptimization in action](https://github.com/fmfn/BayesianOptimization/blob/master/examples/bo_example.png)

As you iterate over and over, the algorithm balances its needs of exploration and exploitation taking into account what it knows about the target function. At each step a Gaussian Process is fitted to the known samples (points previously explored), and the posterior distribution, combined with a exploration strategy (such as UCB (Upper Confidence Bound), or EI (Expected Improvement)), are used to determine the next point that should be explored (see the gif below).

![BayesianOptimization in action](https://github.com/fmfn/BayesianOptimization/blob/master/examples/bayesian_optimization.gif)

This process is designed to minimize the number of steps required to find a combination of parameters that are close to the optimal combination. To do so, this method uses a proxy optimization problem (finding the maximum of the acquisition function) that, albeit still a hard problem, is cheaper (in the computational sense) and common tools can be employed. Therefore Bayesian Optimization is most adequate for situations where sampling the function to be optimized is a very expensive endeavor. See the references for a proper discussion of this method.

This project is under active development, if you find a bug, or anything that
needs correction, please let me know.
