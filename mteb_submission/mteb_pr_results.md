Adds the 20 results of MTEB(Indic, v1) for [indic-e5-small](https://huggingface.co/cidcodes/indic-e5-small), revision `08d044b8792a0d345fc3d7911036edbe251e5425`. Mean (Task) 65.06, Mean (TaskType) 63.67, on a 0-100 scale (multilingual-e5-small: 64.84 / 63.45).

Scored from the pinned public revision with MTEB 2.27.23 and the standard Sentence Transformers wrapper with the multilingual-e5 prompts (torch 2.14.1, transformers 5.24.1, sentence-transformers 6.3.0, fp32, batch size 64); `run_settings.jsonl` covers every task.

Training disclosure: the model is a vocabulary-trimmed `intfloat/multilingual-e5-small`, distilled from that model on Sangraha (AI4Bharat). No split of any MTEB(Indic) dataset is used in the distillation. The base model's training data includes SQuAD, so XQuADRetrieval counts as trained on (it is in `ME5_TRAINING_DATA`). Sangraha's synthetic split is machine-translated English Wikipedia, and WikipediaRerankingMultilingual is also built from Wikipedia, so some article overlap with that task is possible.

### Checklist

- [x] My model has a model sheet, report, or similar
- [ ] My model has a reference implementation in [`mteb/models/model_implementations/`](https://github.com/embeddings-benchmark/mteb/tree/main/mteb/models/model_implementations), this can be as an API. Instruction on how to add a model can be found [here](https://embeddings-benchmark.github.io/mteb/contributing/adding_a_model/)
  - [x] No, but there is an existing PR embeddings-benchmark/mteb#{{model_pr}}
- [x] The results submitted are obtained using the reference implementation
- [x] My model is available, either as a publicly accessible API or publicly on e.g., Huggingface
- [x] I *solemnly swear* that for all results submitted I have not trained on the evaluation dataset including training splits. If I have, I have disclosed it clearly.
