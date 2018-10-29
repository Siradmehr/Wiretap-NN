// Copyright ©2016 The Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/btracey/mixent"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/gonum/stat"
	//"gonum.org/v1/gonum/stat/distuv"
	"bufio"
	"os"
	"encoding/csv"
	"log"
	//"reflect"
	"strings"
	"strconv"
)

//const codebookdir string = "polarcodebooks"
var codebookdir string
const conditional bool = false

func main() {
	f, err := os.Create("results-config1-full.dat")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	names := []string{
		"[0.75, 0.25]",
		"[0.8343844692039338, 0.16561553079606614]",
		"[0.8902859838365491, 0.10971401616345094]",
		"[0.9273186199092875, 0.07268138009071241]",
		"[0.9518513386291441, 0.04814866137085589]",
		"[0.9681033355597826, 0.03189666444021738]",
		"[0.9788696679524376, 0.021130332047562435]",
		"[0.9860019553681832, 0.013998044631816713]",
		"[0.9907268256327786, 0.009273174367221405]",
		"[0.9938568732200326, 0.006143126779967388]",
		"[0.9959304111903526, 0.004069588809647309]",
		"[0.9973040515566741, 0.0026959484433258826]",
		"[0.998214036270239, 0.0017859637297609765]",
		"[0.9988168666756525, 0.0011831333243475443]",
		"[0.9992162189859427, 0.000783781014057331]",
		"[0.999480774765316, 0.0005192252346839358]",
		"[0.9996560329486204, 0.00034396705137956806]",
		"[0.9997721348568377, 0.00022786514316233962]",
		"[0.999849047973461, 0.00015095202653901002]",
		"[0.9999, 0.0001]",
		"[1, 0]",
	}
	for _, combination := range names {
		codebookdir = fmt.Sprintf("results/T5/config1-codewords-%s", combination)
		fmt.Println(codebookdir)


	mcEnt := make([]float64, 16) // store of the entropy from Monte Carlo.
	//for _, name := range names {
	seed := int64(rand.Int())
	//fmt.Println(seed)
	rnd := rand.New(rand.NewSource(seed))
	for messageNumber := 0; messageNumber < 16; messageNumber++ {
		//fmt.Println("Running ", name)
		fmt.Println("Running ", messageNumber)
		run := GetRun("polar_wiretap")
		if conditional == true {
			codebookFile := fmt.Sprintf("%s/codebook-%d.csv", codebookdir, messageNumber)
			_, codewords := readCodebook(codebookFile)
			run.SetHypers(codewords)
		}

		// Fix the random samples for each run. Keep the randomness consistent
		// across the hyper sweep to reduce noise.
		//rnd := rand.New(rand.NewSource(2))
		//rnd := rand.New(rand.NewSource(int64(messageNumber)))

		// The code allows the dimension of the problem to adjust with the
		// hyperparameters. Find the maximum dimension used.
		var maxDim int = run.DistGen.CompDim()
		//for _, _ := range run.Hypers {
		//	dim := run.DistGen.CompDim()
		//	if dim > maxDim {
		//		maxDim = dim
		//	}
		//}
		fmt.Println("NumComponents: ", run.NumComponents)

		// Generate random samples for computing the MC entropy.
		randComps := make([]int, run.MCEntropySamples)
		for i := range randComps {
			randComps[i] = rnd.Intn(run.NumComponents)
		}
		//fmt.Println("randComp: ", len(randComps))
		mcSamps := mat.NewDense(run.MCEntropySamples, maxDim, nil)
		for i := 0; i < run.MCEntropySamples; i++ {
			for j := 0; j < maxDim; j++ {
				mcSamps.Set(i, j, rnd.Float64())
			}
		}

		// Generate the random numbers for components.
		nRand := run.DistGen.NumRandom()
		compSamps := mat.NewDense(run.NumComponents, nRand, nil)
		for i := 0; i < run.NumComponents; i++ {
			for j := 0; j < nRand; j++ {
				compSamps.Set(i, j, rnd.Float64())
			}
		}

		// Sweep over the hyperparameter, and estimate the entropy with all of the
		// estimators
		entComponents := make([]mixent.Component, run.NumComponents)
		components := make([]Component, run.NumComponents)
		//mcEnt := make([]float64, 1) // store of the entropy from Monte Carlo.

		estEnts := mat.NewDense(1, len(run.Estimators), nil) // entropy from estimators.
		for j, mean := range run.Hypers {
			//fmt.Println(mean)
			// Construct the components given the random samples and hyperparameter.
			components[j] = run.DistGen.ComponentFrom(compSamps.RawRowView(j), mean)
			entComponents[j] = components[j]
		}
		// Estimate the entropy with all the estimators.
		for j, estimator := range run.Estimators {
			v := estimator.MixtureEntropy(entComponents, nil)
			estEnts.Set(0, j, v)
		}
		//fmt.Println(estEnts)

		// Estimate the entropy from Monte Carlo.
		dim := run.DistGen.CompDim()
		sv := mcSamps.Slice(0, run.MCEntropySamples, 0, dim).(*mat.Dense)
		mcEnt[messageNumber] = mcEntropy(components, randComps, sv)
		fmt.Println(mcEnt)

		// Plot the results.
		//err := makePlots(run, mcEnt, estEnts)
		//if err != nil {
		//	log.Fatal(err)
		//}
	}
	entropyMean := stat.Mean(mcEnt, nil)
	fmt.Println(entropyMean)
	f.WriteString(fmt.Sprintf("%s: %f\n", combination, entropyMean))
	f.Sync()
}
}

// mcEntropy estimates the entropy of the mixture distribution given the pre-drawn
// random components and samples.
func mcEntropy(components []Component, randComps []int, mcSamps *mat.Dense) float64 {
	nSamp, dim := mcSamps.Dims()
	if len(randComps) != nSamp {
		panic("rand mismatch")
	}
	var ent float64
	lw := -math.Log(float64(len(components))) // probability of chosing component.
	x := make([]float64, dim)
	lps := make([]float64, len(components))
	for i := range randComps {
		// Extract the sampled x location.
		comp := randComps[i]
		components[comp].Quantile(x, mcSamps.RawRowView(i))
		//fmt.Println(x)
		// Compute \sum_i w_i p(x_i).
		for j := range lps {
			lps[j] = components[j].LogProb(x) + lw
		}
		ent += floats.LogSumExp(lps)
	}
	return -ent / float64(nSamp)
}

type Run struct {
	Name             string
	DistGen          DistributionGenerator
	MCEntropySamples int
	NumComponents    int

	Hypers     [][]float64
	Estimators []mixent.Estimator

	XLabel string
	LogX   bool
}

func (r *Run) SetHypers(hypers [][]float64){
	r.Hypers = hypers
	r.NumComponents = len(hypers)
}

func GetRun(name string) Run {
	var isUniform bool

	c := Run{
		Name:          name,
		//NumComponents: int(math.Pow(2, 7)),
		//NumComponents: 128,
	}

	switch name {
	default:
		panic("unknown case name")
	case "polar_wiretap":
		dim := 16
		//hypers := make([][]float64, 3)
		//_, codewords := readCodebook("polar_wtc_codebook.csv")
		_, codewords := readCodebook(codebookdir + "/codebook-all.csv")
		hypers := codewords
		//hypers := [][]float64{
		//	{0, 1, 2, 3},
		//	{4, 5, 6, 7},
		//}
		c.DistGen = GaussianFixedCenter{dim, 3.614031611621005} //Polar
		//c.DistGen = GaussianFixedCenter{dim, 3.5801165469883025} //AE
		c.Hypers = hypers
		c.XLabel = "Polar Wiretap"
		c.LogX = true
		c.NumComponents = len(codewords)
	}
	if isUniform {
		c.Estimators = []mixent.Estimator{
			mixent.JointEntropy{},
			mixent.AvgEnt{},
			mixent.PairwiseDistance{mixent.UniformDistance{distmv.KullbackLeibler{}}},
			mixent.PairwiseDistance{mixent.UniformDistance{distmv.Bhattacharyya{}}},
			mixent.ComponentCenters{},
			mixent.ELK{},
		}
		c.MCEntropySamples = 5000
	} else {
		c.Estimators = []mixent.Estimator{
			mixent.JointEntropy{},
			mixent.AvgEnt{},
			mixent.PairwiseDistance{mixent.NormalDistance{distmv.KullbackLeibler{}}},
			mixent.PairwiseDistance{mixent.NormalDistance{distmv.Bhattacharyya{}}},
			mixent.ComponentCenters{},
			mixent.ELK{},
		}
		//c.MCEntropySamples = 20000
		c.MCEntropySamples = 200
	}
	return c
}

// Component is the same as a mixent.Component, except can also compute probabilities
// and quantiles
type Component interface {
	mixent.Component
	distmv.Quantiler
	distmv.LogProber
}

// DistributionGenerator is a type for generating the Components of a mixture model.
type DistributionGenerator interface {
	// NumRandom is the amount of random numbers needed to generate a component
	// (random location of the mean, etc.)
	NumRandom() int
	// ComponentFrom takes a set of random numbers and turns it into a component
	ComponentFrom(rands []float64, hyper []float64) Component
	// CompDim returns the dimension as a function of the hyperparameter.
	CompDim() int
}

// GaussianFixedCenter generates Gaussian components with a fixed center location
// and covariance drawn from a Wishart distribution. As the hyperparameter increases,
// the covariances tend to the identity matrix.
type GaussianFixedCenter struct {
	Dim int;
	Sigma float64
}

func (w GaussianFixedCenter) CompDim() int {
	return w.Dim
}

func (w GaussianFixedCenter) NumRandom() int {
	// Center location + chi^2 variables + normal variables
	return w.Dim + w.Dim + (w.Dim-1)*w.Dim/2
}

func (w GaussianFixedCenter) ComponentFrom(rands []float64, mu []float64) Component {
	dim := w.Dim

	// The center is just fixed, the first rands.
	//mu := make([]float64, w.Dim)
	//for i := 0; i < dim; i++ {
	//	//mu[i] = distuv.Normal{Mu: 0, Sigma: 1}.Quantile(rands[i])
	//	mu[i] = float64(i)
	//}
	//rands = rands[dim:]

	cov := mat.NewSymDense(dim, nil)
	for i := 0; i < dim; i++ {
		cov.SetSym(i, i, w.Sigma)
	}
	norm, ok := distmv.NewNormal(mu, cov, nil)

	// TODO(btracey): Can set directly from Cholesky.
	//norm, ok := distmv.NewNormal(mu, &cov, nil)
	if !ok {
		panic("bad norm")
	}
	return norm
}

func eyeSym(dim int) *mat.SymDense {
	m := mat.NewSymDense(dim, nil)
	for i := 0; i < dim; i++ {
		m.SetSym(i, i, 1)
	}
	return m
}

func readCodebook(codebookfile string) ([][]float64, [][]float64) {
	f, _ := os.Open(codebookfile)
	r := csv.NewReader(bufio.NewReader(f))
	r.Comma = ','
	//r := csv.NewReader(strings.NewReader(in))
	records, err := r.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	//fmt.Print(records)
	messages := make([][]float64, len(records))
	codewords := make([][]float64, len(records))
	//fmt.Println(len(records))
	for value := range records{
		messages[value] = parseRecords(records[value][0])
		codewords[value] = parseRecords(records[value][1])
		//fmt.Println(messages)
		//fmt.Println(records[value][0], reflect.TypeOf(records[value][0]))
	}
	return messages, codewords
}

func parseRecords(record string) []float64 {
	splits := splitTags(record)
	var mess = []float64{}
	for _, i := range splits {
		j, err := strconv.ParseFloat(i, 64)
		if err != nil {
			panic(err)
		}
		mess = append(mess, j)
	}
	//fmt.Println(mess)
	return mess
}

func splitTags(entry string) []string {
	entry = strings.Trim(entry, "[")
	entry = strings.Trim(entry, "]")
	tags := strings.Split(entry, ", ")
	return tags
}
