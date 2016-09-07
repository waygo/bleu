// Package bleu implements the BLEU method, which is used to evaluate
// the quality of machine translation. [1]
//
// The code in this package was largely ported from the corresponding package
// in Python NLTK. [2]
//
// [1] Papineni, Kishore, et al. "BLEU: a method for automatic evaluation of
//     machine translation." Proceedings of the 40th annual meeting on
//     association for computational linguistics. Association for Computational
//     Linguistics, 2002.
//
// [2] http://www.nltk.org/_modules/nltk/align/bleu.html
package bleu

import (
	"encoding/json"
	"log"
	"math"
	"strings"
)

// Sentence represents a series of tokens used to compute the BLEU score in this package
type Sentence []string

// Compute calculates the BLEU score for a given candidate, references and ngram weights.
func Compute(candidate Sentence, references []Sentence, weights []float64) float64 {
	return computeBleu(candidate, references, weights, false)
}

// Smooth adds a smoothing factor so that any missing ngrams do not result in a score of 0.
// This is useful in settings where the bleu score is being calcuated on the individual sentence
// level. See section 4 of Lin and Och 2004 (http://acl.ldc.upenn.edu/C/C04/C04-1072.pdf)
func Smooth(candidate Sentence, references []Sentence, weights []float64) float64 {
	return computeBleu(candidate, references, weights, true)
}

func computeBleu(candidate Sentence, references []Sentence, weights []float64, smoothing bool) float64 {
	// convert candidate tokens to lower case
	for i := range candidate {
		candidate[i] = strings.ToLower(candidate[i])
	}

	// convert reference tokens to lower case
	for i := range references {
		for u := range references[i] {
			references[i][u] = strings.ToLower(references[i][u])
		}
	}

	// calculate BLEU modified precision
	ps := make([]float64, len(weights))
	for i := range weights {
		ps[i] = modifiedPrecision(candidate, references, i+1, smoothing)
	}

	s := 0.0
	overlap := 0
	for i := range weights {
		w := weights[i]
		pn := ps[i]
		if pn > 0.0 {
			overlap++
			s += w * math.Log(pn)
		}
	}

	// if none of the ngrams have any overlap with the reference translations,
	// return 0. See https://github.com/nltk/nltk/issues/1268 for discussion.
	if overlap == 0 {
		return 0
	}

	bp := brevityPenalty(candidate, references)
	return bp * math.Exp(s)
}

type phrase []string

func (p phrase) String() string {
	// var b bytes.Buffer
	// enc := gob.NewEncoder(&b)
	// err := enc.Encode(p)
	// if err != nil {
	// 	log.Fatal("encode error:", err)
	// }
	// return b.String()
	b, err := json.Marshal(p)
	if err != nil {
		log.Fatal("encode error:", err)
	}
	return string(b)
}

func getNgrams(s Sentence, n int) []phrase {
	ngrams := []phrase{}
	for i := 0; i < len(s)-n+1; i++ {
		ngrams = append(ngrams, phrase(s[i:i+n]))
	}
	return ngrams
}

func countNgrams(ngrams []phrase) map[string]int {
	counts := map[string]int{}
	for _, gram := range ngrams {
		counts[gram.String()]++
	}
	return counts
}

func sum(m map[string]int) int {
	s := 0
	for _, v := range m {
		s += v
	}
	return s
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// modifiedPrecision implements a version of precision that rectifies a problem with
// the normal precision method, where some wrong translations achieve high precision,
// e.g., the translation, in which a word of reference repeats several times,
// has very high precision. So in the modified n-gram precision, a reference
// word will be considered exhausted after a matching candidate word is identified.
func modifiedPrecision(candidate Sentence, references []Sentence, n int, smoothing bool) float64 {
	ngrams := getNgrams(candidate, n)
	if len(ngrams) == 0 {
		return 0.0
	}

	counts := countNgrams(ngrams)

	if len(counts) == 0 {
		return 0.0
	}

	maxCounts := map[string]int{}
	for i := range references {
		referenceCounts := countNgrams(getNgrams(references[i], n))
		for ngram := range counts {
			if v, ok := maxCounts[ngram]; !ok {
				maxCounts[ngram] = referenceCounts[ngram]
			} else if v < referenceCounts[ngram] {
				maxCounts[ngram] = referenceCounts[ngram]
			}
		}
	}

	clippedCounts := map[string]int{}
	for ngram, count := range counts {
		clippedCounts[ngram] = min(count, maxCounts[ngram])
	}

	// we add smoothing to these so that we never return 0.0
	smoothingFactor := 0.0
	if smoothing {
		smoothingFactor = 1.0
	}
	return (float64(sum(clippedCounts)) + smoothingFactor) / (float64(sum(counts)) + smoothingFactor)
}

// brevityPenalty applies a penalty to translation candidates that are longer
// than the reference translations.
// As the modified n-gram precision still has the problem from the short
// length sentence, brevity penalty is used to modify the overall BLEU
// score according to length.
func brevityPenalty(candidate Sentence, references []Sentence) float64 {
	c := len(candidate)
	refLens := []int{}
	for i := range references {
		refLens = append(refLens, len(references[i]))
	}
	minDiffInd, minDiff := 0, -1
	for i := range refLens {
		if minDiff == -1 || abs(refLens[i]-c) < minDiff {
			minDiffInd = i
			minDiff = abs(refLens[i] - c)
		}
	}
	r := refLens[minDiffInd]
	if c > r {
		return 1
	}
	return math.Exp(float64(1 - float64(r)/float64(c)))
}
