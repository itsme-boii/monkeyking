package main

import (
	"fmt"
	"math/rand"

	"github.com/consensys/gnark/frontend"
)

// Function to approximate hyperbolic tangent (tanh)
func approxTanh(api frontend.API, x frontend.Variable) frontend.Variable {
	x2 := api.Mul(x, x)
	x3 := api.Mul(x2, x)
	x5 := api.Mul(x3, x2)
	x7 := api.Mul(x5, x2)

	term1 := x
	term2 := api.Div(x3, 3)
	term3 := api.Div(api.Mul(x5, 2), 15)
	term4 := api.Div(api.Mul(x7, 17), 315)

	return api.Sub(api.Sub(api.Add(term1, term3), term2), term4)
}

// Function to compute square root using Newton's method
func sqrt(api frontend.API, x frontend.Variable) frontend.Variable {
	z := x
	for i := 0; i < 10; i++ {
		z = api.Div(api.Add(z, api.Div(x, z)), 2)
	}
	return z
}

// Function to compute GELU activation
func newGELU(api frontend.API, x frontend.Variable) frontend.Variable {
	fmt.Println("newGelu in")
	constB := frontend.Variable(797884)
	constC := frontend.Variable(44715)
	scale := frontend.Variable(1000000)
	xCube := api.Mul(api.Mul(x, x), x)
	inner := api.Add(x, api.Div(api.Mul(constC, xCube), scale))
	tanhInput := api.Div(api.Mul(constB, inner), scale)
	tanhOutput := approxTanh(api, tanhInput)
	constAVal := frontend.Variable(1)
	fmt.Println("newGelu out")

	return api.Div(api.Mul(x, api.Add(constAVal, tanhOutput)), 2)
}

// Function to perform layer normalization
func layerNorm(api frontend.API, x []frontend.Variable, weight, bias, epsilon frontend.Variable) []frontend.Variable {
	fmt.Println("layer Norm in")
	var mean frontend.Variable
	mean = frontend.Variable(0)
	var variance frontend.Variable
	variance = frontend.Variable(0)

	if x == nil {
		return nil
	}

	for _, xi := range x {
		mean = api.Add(mean, xi)
	}
	mean = api.Div(mean, frontend.Variable(len(x)))

	var xSubMean []frontend.Variable
	for _, xi := range x {
		xSubMean = append(xSubMean, api.Sub(xi, mean))
		variance = api.Add(variance, api.Mul(xi, xi))
	}
	variance = api.Div(variance, frontend.Variable(len(x)))

	add := api.Add(variance, epsilon)
	invStdDev := api.Div(1, sqrt(api, add))

	fmt.Println("mean is ", mean)
	fmt.Println("mean is ", xSubMean)
	fmt.Println("mean is ", variance)

	var normalized []frontend.Variable
	for _, xi := range xSubMean {
		normalized = append(normalized, api.Add(api.Mul(xi, invStdDev), bias))
	}
	fmt.Println("layer Norm out")

	return normalized
}

// Function to perform linear layer computation
func linearLayer(api frontend.API, x []frontend.Variable, weights [][]frontend.Variable, bias frontend.Variable) []frontend.Variable {
	fmt.Println("linear layer in")
	var output []frontend.Variable

	if x == nil || weights == nil {
		return output
	}

	for _, weight := range weights {
		var sum frontend.Variable
		sum = frontend.Variable(0)

		for j, w := range weight {
			if j < len(x) {
				sum = api.Add(sum, api.Mul(x[j], w))
			}
		}

		output = append(output, api.Add(sum, bias))
	}
	fmt.Println("linear layer out")

	return output
}

// Function to perform MLP block computation
func mlpBlock(api frontend.API, x []frontend.Variable, config map[string]int) []frontend.Variable {
	fmt.Println("Mlp Block in")
	hiddenDim := 4 * config["nEmbd"]

	weights1 := make([][]frontend.Variable, config["nEmbd"])
	for i := range weights1 {
		weights1[i] = make([]frontend.Variable, hiddenDim)
		for j := range weights1[i] {
			weights1[i][j] = frontend.Variable(rand.Intn(10) + 1)
		}
	}
	weights2 := make([][]frontend.Variable, hiddenDim)
	for i := range weights2 {
		weights2[i] = make([]frontend.Variable, config["nEmbd"])
		for j := range weights2[i] {
			weights2[i][j] = frontend.Variable(rand.Intn(10) + 1)
		}
	}
	bias1 := frontend.Variable(rand.Intn(10))
	bias2 := frontend.Variable(rand.Intn(10))

	// Perform linear layer computations
	fc1 := linearLayer(api, x, weights1, bias1)
	gelu1 := newGELU(api, fc1[0])
	fc2 := linearLayer(api, []frontend.Variable{gelu1}, weights2, bias2)
	fmt.Println("Mlp Block out")

	return fc2
}

// Function to perform transformer block computation
func transformerBlock(api frontend.API, x []frontend.Variable, config map[string]int) []frontend.Variable {
	fmt.Println("Transformer Block in")
	attn := x
	xAddAttn := append([]frontend.Variable{}, x...)
	for i := range xAddAttn {
		xAddAttn[i] = api.Add(x[i], attn[i])
	}
	ln2 := layerNorm(api, xAddAttn, frontend.Variable(rand.Intn(5)), frontend.Variable(rand.Intn(5)), frontend.Variable(10))
	mlp := mlpBlock(api, ln2, config)
	xAddMlp := append([]frontend.Variable{}, xAddAttn...)
	for i := range xAddMlp {
		xAddMlp[i] = api.Add(xAddAttn[i], mlp[i])
	}
	fmt.Println("Transformer Block out")

	return xAddMlp
}

// GPT model
func gpt(api frontend.API, input frontend.Variable, config map[string]int) frontend.Variable {
	fmt.Println("gpt in")
	tokenEmbeddings := []frontend.Variable{input}
	positionEmbeddings := []frontend.Variable{frontend.Variable(rand.Intn(100))}
	x := append([]frontend.Variable{}, tokenEmbeddings...)
	for i := range x {
		x[i] = api.Add(x[i], positionEmbeddings[i])
	}
	for i := 0; i < config["nLayer"]; i++ {
		x = transformerBlock(api, x, config)
	}
	ln := layerNorm(api, x, frontend.Variable(rand.Intn(5)), frontend.Variable(rand.Intn(5)), frontend.Variable(10))
	logits := linearLayer(api, ln, [][]frontend.Variable{{frontend.Variable(rand.Intn(10))}}, frontend.Variable(rand.Intn(10)))
	fmt.Println("gpt out")

	return logits[0]
}

type MyCircuit struct {
	X frontend.Variable `gnark:"x"`
	Y frontend.Variable `gnark:",public"`
	Z frontend.Variable `gnark:",public"`
}

func (circuit *MyCircuit) Define(api frontend.API) error {
	config := map[string]int{
		"blockSize": 512, // Increase blockSize
		"vocabSize": 513, // Increase vocabSize
		"nLayer":    32,  // Increase nLayer
		"nHead":     32,
		"nEmbd":     512,
		"dropout":   0,
	}

	result := gpt(api, circuit.X, config)

	ln := layerNorm(api, []frontend.Variable{circuit.Y, circuit.Z}, frontend.Variable(rand.Intn(5)), frontend.Variable(rand.Intn(5)), frontend.Variable(10))
	api.AssertIsEqual(ln[0], ln[1])
	xSquared := api.Mul(circuit.X, circuit.X)
	xCubed := api.Mul(xSquared, circuit.X)
	xSquaredRoot := sqrt(api, xSquared)
	xCubedRoot := sqrt(api, xCubed)
	xPowFour := api.Mul(xSquared, xSquared)
	xPowSix := api.Mul(xCubed, xSquared)

	for i := 0; i < 10000; i++ {
		api.AssertIsLessOrEqual(xSquaredRoot, xCubedRoot)
		api.AssertIsDifferent(xPowSix, xPowFour)
	}

	// Example of using model output in constraints
	api.AssertIsLessOrEqual(result, frontend.Variable(rand.Intn(10000)))

	return nil
}
