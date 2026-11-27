# indic-embed

Training and evaluation code for [**indic-e5-small**](https://huggingface.co/cidcodes/indic-e5-small), a 47M-parameter
sentence-embedding model for 13 Indian languages and English that handles Indic text both in its native script and
typed in the Latin alphabet ("mujhe kal office jana hai"). [Try it in the browser](https://huggingface.co/spaces/cidcodes/indic-e5-small-demo).

| model | params | MTEB(Indic) | Dakshina roman→native R@1 | native→roman R@1 | TanglishSTS |
|---|---|---|---|---|---|
| **indic-e5-small** | **47M** | **0.651** | **0.827** | **0.905** | 0.699 |
| multilingual-e5-small | 118M | 0.648 | 0.396 | 0.498 | 0.729 |
| L3Cube indic-sentence-similarity-sbert | 238M | 0.592 | 0.772 | 0.819 | 0.692 |
| bge-m3 | 568M | 0.686 | 0.534 | 0.586 | 0.758 |

Full comparison of 14 models: [results/baselines.md](results/baselines.md). Dakshina and TanglishSTS are used only
for evaluation; checkpoints are chosen on a held-out split of the training data.

## How it is made

1. **Baselines** (`run_mteb.py`, `eval_codemix.py`, `compare.py`): MTEB(Indic, v1) and a romanized/code-mixed suite
   (Dakshina cross-script retrieval in 10 languages, TanglishSTS) for candidate teachers and competitors.
2. **Vocabulary trimming** (`vocab_counts.py`, `vocab_counts_sangraha.py`, `trim_vocab.py`, `verify_trim.py`):
   token frequencies from Wikipedia and Sangraha, balanced per language; 250K → 64K tokens, keeping every single
   character of the target scripts, then gathering the embedding rows. 118M → 47M parameters at -0.2% MTEB(Indic).
3. **Distillation** (`build_data.py`, `build_lexicon.py`, `augment.py`, `train_phase_a.py`): the original
   multilingual-e5-small is the frozen teacher. On 256K word-aligned native/romanized paragraph pairs from
   Sangraha, the student learns S(romanized) ≈ T(native), plus
   - casual spelling variation (vowel length, aspiration, doubled consonants, Dravidian `zh`),
   - code-mixing: native→English word pairs induced from the teacher's own embeddings (margin-based mutual nearest
     neighbours) are swapped into the romanized text,
   - short query-like inputs: the same 2-8 word window of each aligned pair, with the teacher run on the fly,
   - S(native) ≈ T(native) and S(English) ≈ T(English) so nothing else moves.

   One epoch takes about 75 minutes on a 4 GB laptop GPU (RTX 3050).
4. **Release** (`prepare_release.py`, `check_release.py`, `export_onnx.py`): a model folder that loads with
   identical vectors on sentence-transformers 5.x/transformers 4.x and 6.x/5.x, ONNX fp32 and per-channel int8
   (47 MB, cosine 0.996), checked against transformers.js (`demo/node-check`).
5. **MTEB submission** (`mteb_submission/`): the ModelMeta and the runner used for the leaderboard results.

## Reproduce

```sh
python -m venv .venv && .venv/Scripts/pip install torch sentence-transformers mteb datasets wordfreq
python build_data.py                         # Sangraha pairs -> data/train/
python train_phase_a.py --precompute         # teacher embeddings
python build_lexicon.py                      # induced native->English lexicon
python train_phase_a.py --mix --crops --out models/indic-e5-small
python prepare_release.py models/indic-e5-small models/release/indic-e5-small
python run_mteb.py models/release/indic-e5-small --like intfloat/multilingual-e5-small
python eval_codemix.py models/release/indic-e5-small --like intfloat/multilingual-e5-small
```

The trimmed starting model comes from `vocab_counts*.py` and `trim_vocab.py --budget 65536 --out models/e5-small-indic-64k`.

## Data and licenses

Code: MIT. Base model: multilingual-e5-small (MIT). Training text: [Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha)
(AI4Bharat, CC BY 4.0). Evaluation only: Dakshina (CC BY-SA 4.0), TanglishSTS (CC BY 4.0), MTEB. No dataset is
redistributed here.
