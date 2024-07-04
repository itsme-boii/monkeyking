package main

import (
	"github.com/consensys/gnark/frontend"
)

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

func NewGELU(api frontend.API, x frontend.Variable) frontend.Variable {
	constB := frontend.Variable(797884)
	constC := frontend.Variable(44715)
	scale := frontend.Variable(1000000)

	xCube := api.Mul(api.Mul(x, x), x)
	inner := api.Add(x, api.Div(api.Mul(constC, xCube), scale))
	tanhInput := api.Div(api.Mul(constB, inner), scale)
	tanhOutput := approxTanh(api, tanhInput)
	constAVal := frontend.Variable(1)

	return api.Div(api.Mul(x, api.Add(constAVal, tanhOutput)), 2)
}

func LinearLayer(api frontend.API, input frontend.Variable, weight []frontend.Variable, bias frontend.Variable) frontend.Variable {
	var sum frontend.Variable
	sum = bias
	for i := 0; i < len(weight); i++ {
		sum = api.Add(sum, api.Mul(input, weight[i]))
	}
	return sum
}

func MLP(api frontend.API, input frontend.Variable) frontend.Variable {
	constAVal := frontend.Variable(500000)
	constBVal := frontend.Variable(-300000)
	constcVal := frontend.Variable(200000)
	constdVal := frontend.Variable(700000)
	weight1 := []frontend.Variable{constAVal, constBVal}
	bias1 := frontend.Variable(100000)
	linear1 := LinearLayer(api, input, weight1, bias1)
	gelu1 := NewGELU(api, linear1)

	weight2 := []frontend.Variable{constcVal, constdVal}
	bias2 := frontend.Variable(50000)
	linear2 := LinearLayer(api, gelu1, weight2, bias2)

	return linear2
}

type MyCircuit struct {
	X frontend.Variable `gnark:"x"`
	Y frontend.Variable `gnark:",public"`
}

func (circuit *MyCircuit) Define(api frontend.API) error {
	result := MLP(api, frontend.Variable(circuit.X))
	api.AssertIsEqual(circuit.Y, result)
	return nil
}
