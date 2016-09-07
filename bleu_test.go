package bleu

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

func split(s string) []string {
	return strings.Split(s, " ")
}

var references1 = []Sentence{
	split("the cat is on the mat"),
	split("there is a cat on the mat"),
}

var references2 = []Sentence{
	split("It is a guide to action that ensures that the military will forever heed Party commands."),
	split("It is the guiding principle which guarantees the military forces always being under the command of the Party."),
	split("It is the practical guide for the army always to heed the directions of the party"),
}

func TestModifiedPrecision(t *testing.T) {
	var cases = []struct {
		candidate  Sentence
		references []Sentence
		n          int
		want       float64
	}{
		{
			candidate: split("cat mat"),
			references: []Sentence{
				split("cat on the mat"),
			},
			n:    1,
			want: 1,
		},
		{
			candidate: split("cat mat"),
			references: []Sentence{
				split("cat on the mat"),
			},
			n:    2,
			want: 0,
		},
		{
			candidate:  split("the the the the the the the"),
			references: references1,
			n:          1,
			want:       0.2857,
		},
		{
			candidate:  split("the the the the the the the"),
			references: references1,
			n:          2,
			want:       0.0,
		},
		{
			candidate:  split("of the"),
			references: references2,
			n:          1,
			want:       1.0,
		},
		{
			candidate:  split("of the"),
			references: references2,
			n:          2,
			want:       1.0,
		},
	}

	for _, tt := range cases {
		mp := modifiedPrecision(tt.candidate, tt.references, tt.n, false)
		if math.Abs(mp-tt.want) > 0.0001 {
			t.Errorf("Case with:\n\tcandidate: %q\n\treferences: %q\n\tn: %v\nGot precision = %v, want %v", tt.candidate, tt.references, tt.n, mp, tt.want)
		}
	}
}

func TestBrevityPenalty(t *testing.T) {
	bp := brevityPenalty(split("cat mat"), []Sentence{split("cat on the mat")})
	if math.Abs(bp-0.3678) > 0.01 {
		t.Errorf("brevityPenalty got %v, want %v", bp, 0.3678)
	}

	bp = brevityPenalty(split("1 2 3 4 5 6 7 8 9 10 11 12"), []Sentence{split("1 2 3 4 5 6 7 8 9 10 11 12 13"), split("1 2")})
	if math.Abs(bp-0.92) > 0.01 {
		t.Errorf("brevityPenalty got %v, want %v", bp, 0.92)
	}
}

func TestBLEUScore(t *testing.T) {
	var cases = []struct {
		candidate  Sentence
		references []Sentence
		weights    []float64
		want       float64
	}{
		{
			candidate: split("cat mat"),
			references: []Sentence{
				split("cat on the mat"),
			},
			weights: []float64{0.5, 0.5},
			want:    0.36787,
		},
		{
			candidate: split("garlic mushrooms"),
			references: []Sentence{
				split("garlic mushrooms"),
			},
			weights: []float64{0.5, 0.5, 0.5, 0.5},
			want:    1.0,
		},
		{
			candidate: split("Champinones sizzled"),
			references: []Sentence{
				split("garlic mushrooms"),
			},
			weights: []float64{0.5, 0.5, 0.5, 0.5},
			want:    0.0,
		},
		{
			candidate:  split("It is a guide to action which ensures that the military always obeys the commands of the party"),
			references: references2,
			weights:    []float64{0.25, 0.25, 0.25, 0.25},
			want:       0.504,
		},
		{
			candidate:  split("It is to insure the troops forever hearing the activity guidebook that party direct"),
			references: references2,
			weights:    []float64{0.25, 0.25, 0.25, 0.25},
			want:       0.396,
		},
		{
			candidate:  split("ham and egg"),
			references: []Sentence{split("Ham and Eggs")},
			weights:    []float64{0.25, 0.25, 0.25, 0.25},
			want:       0.7598,
		},
		{
			candidate:  split("artichokes with the butter"),
			references: []Sentence{split("hearts of artichoke in butter sauce")},
			weights:    []float64{0.25, 0.25, 0.25, 0.25},
			want:       0.4288,
		},
	}
	for _, tt := range cases {
		score := Compute(tt.candidate, tt.references, tt.weights)
		if math.Abs(score-tt.want) > 0.01 {
			t.Errorf("Case with:\n\tcandidate: %q\n\treferences: %q\n\tweights: %v\nGot BLEU score = %v, want %v", tt.candidate, tt.references, tt.weights, score, tt.want)
		}
	}
}

func TestSmoothBLEUScore(t *testing.T) {
	var cases = []struct {
		candidate  Sentence
		references []Sentence
		weights    []float64
		want       float64
	}{
		{
			candidate: split("artichokes with the butter"),
			references: []Sentence{
				split("hearts of artichoke in butter sauce"),
			},
			weights: []float64{0.5, 0.5},
			want:    0.30326,
		},
		{
			candidate: split("butter artichokes"),
			references: []Sentence{
				split("hearts of artichoke in butter sauce"),
			},
			weights: []float64{0.5, 0.5},
			want:    0.09569,
		},
	}
	for _, tt := range cases {
		score := Compute(tt.candidate, tt.references, tt.weights)
		if math.Abs(score-tt.want) > 0.01 {
			t.Errorf("Case with:\n\tcandidate: %q\n\treferences: %q\n\tweights: %v\nGot smooth BLEU score = %v, want %v", tt.candidate, tt.references, tt.weights, score, tt.want)
		}
	}
}

func ExampleCompute() {
	references := []Sentence{
		strings.Split("the cat is on the mat", " "),
		strings.Split("there is a cat on the mat", " "),
	}

	weights := []float64{0.25, 0.25, 0.25, 0.25}

	score := Compute(strings.Split("cat on mat", " "), references, weights)
	fmt.Printf("%.2f", score)
	// Output: 0.31
}
