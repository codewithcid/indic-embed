Adds [`cidcodes/indic-e5-small`](https://huggingface.co/cidcodes/indic-e5-small): `intfloat/multilingual-e5-small` with its vocabulary trimmed to Indic scripts, Latin and English (250K → 64K tokens, 118M → 47M parameters), then distilled from the original model on [Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) so that romanized and code-mixed Indic text is embedded like the native-script text.

It uses the multilingual-e5 prompts (`query: ` / `passage: `) and the existing `SentenceTransformerEncoderWrapper`. `training_datasets` is `ME5_TRAINING_DATA`, inherited from the base model; the distillation data (Sangraha) is not an MTEB dataset.

MTEB(Indic, v1), 20 tasks, at the pinned revision: Mean (Task) 65.06, Mean (TaskType) 63.67 (multilingual-e5-small: 64.84 / 63.45). Results: embeddings-benchmark/results#{{results_pr}}

- [x] I have filled out the ModelMeta object to the extent possible
- [x] I have ensured that my model can be loaded using
  - [x] `mteb.get_model(model_name, revision)` and
  - [x] `mteb.get_model_meta(model_name, revision)`
- [x] I have tested the implementation works on a representative set of tasks.
- [x] The model is public, i.e., is available either as an API or the weights are publicly available to download
- [ ] I reproduced results from the original paper (if applicable) on at least one benchmark, and I am including the results in the PR description. (not applicable: there is no paper; the MTEB(Indic, v1) scores are above)
