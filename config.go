package main

type GPTConfig struct {
	BlockSize int
	VocabSize int
	NLayer    int
	NHead     int
	NEmbd     int
	Dropout   float64
	Bias      bool
}
