package main

import (
	"fmt"
	"strings"

	"github.com/waygo/bleu"
)

func main() {
	references := []bleu.Sentence{
		strings.Split("the cat is on the mat", " "),
		strings.Split("there is a cat on the mat", " "),
	}
	weights := []float64{0.25, 0.25, 0.25, 0.25}
	sentence := strings.Split("cat on mat", " ")

	score := bleu.Compute(sentence, references, weights)
	fmt.Printf("%.2f\n", score)
	// Output: 0.31
}
