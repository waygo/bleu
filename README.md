# BLEU

This package implements a library for using the [BLEU method](https://en.wikipedia.org/wiki/BLEU), used to evaluate the quality of machine translation. It is written in Go.

## Installation

With a valid Go installation, simply use `go get`:

```
go get github.com/waygo/bleu
```

## Usage

```
import (
	"github.com/waygo/bleu"
	"strings"
)

func main() {
	references := []bleu.Sentence{
		strings.Split("the cat is on the mat", " "),
		strings.Split("there is a cat on the mat", " "),
	}
	weights := []float64{0.25, 0.25, 0.25, 0.25}
	sentence := strings.Split("cat on mat", " ")

	score := bleu.Compute(sentence, references, weights)
	fmt.Printf("%.2f", score)
	// Output: 0.31
}
```

## Documentation

See [documentation on godoc](https://godoc.org/github.com/waygo/bleu).