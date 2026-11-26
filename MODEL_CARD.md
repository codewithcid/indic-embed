---
license: mit
base_model: intfloat/multilingual-e5-small
library_name: sentence-transformers
pipeline_tag: sentence-similarity
language: [hi, bn, ta, te, ml, mr, gu, kn, pa, or, as, ur, ne, en]
tags: [sentence-transformers, embeddings, indic, romanized, transliteration, code-mixing, hinglish, tanglish]
datasets: [ai4bharat/sangraha]
---

<!-- DRAFT: numbers are for models/indic-e5-small-phaseA2; set {{repo}} before publishing -->

# indic-e5-small

A 47M-parameter sentence-embedding model for 13 Indian languages and English that understands Indic text **in its
native script and typed in the Latin alphabet** ("mujhe kal office jaana hai", "naan nalaikku varuven").
It is [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) with the vocabulary trimmed
to the scripts that matter here (250K → 64K tokens, 118M → 47M parameters), then distilled so that romanized,
casually spelled and code-mixed text lands where the original model puts the native-script text.

| model | params | MTEB(Indic) | Dakshina roman→native R@1 | native→roman R@1 | TanglishSTS |
|---|---|---|---|---|---|
| **indic-e5-small** | **47M** | 0.648 | **0.776** | **0.879** | 0.703 |
| multilingual-e5-small | 118M | 0.648 | 0.396 | 0.498 | 0.729 |
| L3Cube indic-sentence-similarity-sbert | 238M | 0.592 | 0.772 | 0.819 | 0.692 |
| bge-m3 | 568M | 0.686 | 0.534 | 0.586 | 0.758 |
| LaBSE | 471M | 0.617 | 0.478 | 0.436 | 0.750 |

MTEB(Indic, v1): mean over its 20 tasks. Dakshina: 2,000 human-romanized sentences per language, 10 languages,
retrieval of the native sentence among 2,000. TanglishSTS: Spearman on 325 Tamil–English code-mixed pairs.
Neither Dakshina nor TanglishSTS was used for training or for choosing checkpoints. TanglishSTS is small: the
difference to multilingual-e5-small is -0.027, 95% paired-bootstrap CI [-0.076, +0.020], i.e. not significant.
English is unchanged (STSB 0.838 vs 0.841).

## Usage

Same prefixes as e5: `query: ` for queries and symmetric tasks, `passage: ` for documents.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("{{repo}}")
q = model.encode(["query: mujhe kal office jaana hai"], normalize_embeddings=True)
d = model.encode(["passage: मुझे कल ऑफिस जाना है", "passage: I have to go to the office tomorrow"],
                 normalize_embeddings=True)
print(q @ d.T)
```

## How it was made

1. **Vocabulary trimming.** Token frequencies from Wikipedia and Sangraha in 13 languages plus romanized text,
   balanced per language; the 64K most useful pieces (and every single character of the target scripts) are kept
   and the embedding rows gathered. MTEB(Indic) 0.6471 vs 0.6484 for the full model.
2. **Distillation** ([Reimers & Gurevych, 2020](https://arxiv.org/abs/2004.09813)). Teacher: the original
   multilingual-e5-small, frozen. On 256K aligned native/romanized paragraphs from Sangraha's synthetic split, the
   student is trained so that S(native), S(romanized), S(casually spelled romanized) and S(code-mixed romanized)
   all match T(native), and S(English) matches T(English). Code-mixing swaps words for English using a lexicon
   induced from the teacher's own embeddings (margin-based mutual nearest neighbours). One epoch on a 4 GB laptop
   GPU (RTX 3050), about an hour.

## Limitations

- Romanization in the training data is synthetic; very informal spellings and heavy code-mixing are only
  approximated by rules and an induced lexicon.
- Tamil is the weakest language for romanized input (Dakshina R@1 0.586).
- Max sequence length 512 as in e5-small, but training used 128 tokens.

## Data and licenses

Base model: multilingual-e5-small (MIT). Training text: [Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha)
(AI4Bharat, CC BY 4.0). Evaluation only: Dakshina (CC BY-SA 4.0), TanglishSTS (CC BY 4.0), MTEB.
