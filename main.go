package main

import (
	"fmt"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

func main() {
	var circuit MyCircuit

	r1cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		fmt.Println("Error compiling circuit:", err)
		return
	}

	pk, vk, err := groth16.Setup(r1cs)
	if err != nil {
		fmt.Println("Error setting up keys:", err)
		return
	}

	var assignment MyCircuit
	assignment.X = frontend.Variable(20)
	assignment.Y = frontend.Variable(1)

	witness, err := frontend.NewWitness(&assignment, ecc.BN254.ScalarField())
	if err != nil {
		fmt.Println("Error creating witness:", err)
		return
	}

	proof, err := groth16.Prove(r1cs, pk, witness)
	if err != nil {
		fmt.Println("Error generating proof:", err)
		return
	}

	publicWitness, err := frontend.NewWitness(&assignment, ecc.BN254.ScalarField(), frontend.PublicOnly())
	if err != nil {
		fmt.Println("Error creating public witness:", err)
		return
	}

	err = groth16.Verify(proof, vk, publicWitness)
	if err != nil {
		fmt.Println("Proof verification failed:", err)
	} else {
		fmt.Println("Proof verified successfully")
	}
}
