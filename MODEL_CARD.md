---
license: mit
base_model: intfloat/multilingual-e5-small
library_name: sentence-transformers
pipeline_tag: sentence-similarity
language: [hi, bn, ta, te, ml, mr, gu, kn, pa, or, as, ur, ne, en]
tags: [sentence-transformers, embeddings, indic, romanized, transliteration, code-mixing, hinglish, tanglish]
datasets: [ai4bharat/sangraha]
---

# indic-e5-small

A 47M-parameter sentence-embedding model for 13 Indian languages and English that understands Indic text **in its
native script and typed in the Latin alphabet** ("mujhe kal office jaana hai", "naan nalaikku varuven").
It is [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) with the vocabulary trimmed
to the scripts that matter here (250K → 64K tokens, 118M → 47M parameters), then distilled so that romanized,
casually spelled and code-mixed text lands where the original model puts the native-script text.

| model | params | MTEB(Indic) | Dakshina roman→native R@1 | native→roman R@1 | TanglishSTS |
|---|---|---|---|---|---|
| **indic-e5-small** | **47M** | **0.651** | **0.827** | **0.905** | 0.699 |
| multilingual-e5-small | 118M | 0.648 | 0.396 | 0.498 | 0.729 |
| L3Cube indic-sentence-similarity-sbert | 238M | 0.592 | 0.772 | 0.819 | 0.692 |
| bge-m3 | 568M | 0.686 | 0.534 | 0.586 | 0.758 |
| LaBSE | 471M | 0.617 | 0.478 | 0.436 | 0.750 |

MTEB(Indic, v1): mean over its 20 tasks (retrieval 0.780 vs 0.768 for multilingual-e5-small). Dakshina: 2,000 human-romanized sentences per language, 10 languages,
retrieval of the native sentence among 2,000. TanglishSTS: Spearman on 325 Tamil–English code-mixed pairs.
Neither Dakshina nor TanglishSTS was used for training or for choosing checkpoints. TanglishSTS is small: the
difference to multilingual-e5-small is -0.032, 95% paired-bootstrap CI [-0.081, +0.017], i.e. not significant.
English is unchanged (STSB 0.837 vs 0.841, SICK-R 0.773 vs 0.775).

## Usage

Same prefixes as e5: `query: ` for queries and symmetric tasks, `passage: ` for documents.
`encode_query` / `encode_document` add them for you (sentence-transformers 5+); with older versions, or with
plain `encode`, add them yourself.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("cidcodes/indic-e5-small")
q = model.encode_query(["mujhe kal office jana hai"])
d = model.encode_document(["मुझे कल ऑफिस जाना है", "আজ খুব বৃষ্টি হচ্ছে", "I have to go to the office tomorrow"])
print(model.similarity(q, d))
```

In the browser or Node with [transformers.js](https://github.com/huggingface/transformers.js) (`onnx/model_quantized.onnx`,
47 MB, per-channel int8; cosine ≥ 0.99 to the full model). Try it in the
[demo Space](https://huggingface.co/spaces/cidcodes/indic-e5-small-demo).

```js
import { pipeline } from "@huggingface/transformers";

const embed = await pipeline("feature-extraction", "cidcodes/indic-e5-small", { dtype: "q8" });
const out = await embed(["query: mujhe kal office jana hai", "passage: मुझे कल ऑफिस जाना है"],
                        { pooling: "mean", normalize: true });
```

## How it was made

1. **Vocabulary trimming.** Token frequencies from Wikipedia and Sangraha in 13 languages plus romanized text,
   balanced per language; the 64K most useful pieces (and every single character of the target scripts) are kept
   and the embedding rows gathered. MTEB(Indic) 0.6471 vs 0.6484 for the full model.
2. **Distillation** ([Reimers & Gurevych, 2020](https://arxiv.org/abs/2004.09813)). Teacher: the original
   multilingual-e5-small, frozen. On 256K aligned native/romanized paragraphs from Sangraha's synthetic split, the
   student is trained so that S(native), S(romanized), S(casually spelled romanized) and S(code-mixed romanized)
   all match T(native), and S(English) matches T(English). Code-mixing swaps words for English using a lexicon
   induced from the teacher's own embeddings (margin-based mutual nearest neighbours). Short, query-like inputs
   come from the same 2-8 word window of each word-aligned pair, with the teacher run on the fly. One epoch on a
   4 GB laptop GPU (RTX 3050), about 75 minutes.

## Limitations

- Romanization in the training data is synthetic; very informal spellings and heavy code-mixing are only
  approximated by rules and an induced lexicon.
- Tamil is the weakest language for romanized input (Dakshina R@1 0.668; Hindi 0.921).
- Training text is Wikipedia-style written language. Very colloquial spoken forms (e.g. spoken Tamil
  "enakku pasikkudhu" for written "enakku pasikkiradhu") and short chat messages are weaker.
- Max sequence length 512 as in e5-small, but training used 128 tokens.

## Data and licenses

Base model: multilingual-e5-small (MIT). Training text: [Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha)
(AI4Bharat, CC BY 4.0). Evaluation only: Dakshina (CC BY-SA 4.0), TanglishSTS (CC BY 4.0), MTEB.
