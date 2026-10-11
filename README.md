# myPolar: Polarization Detection, Type Classification and Manifestation Identification

Group 3 submission for Assignment 7 (AI Engineering Fundamental, Batch 2). We merged the class's Assignment-2 myPolar annotations into one dataset and compared a traditional ML baseline with a Bi-LSTM, a Transformer trained from scratch, and a fine-tuned XLM-RoBERTa on the three [POLAR @ SemEval-2026 Task 9](https://polar-semeval.github.io/tasks.html) subtasks.

| Task | Question | Type | Labels |
|---|---|---|---|
| 1 | Is the sentence polarized? | binary | `polarization` |
| 2 | What kind of polarization? | multi-label | `political`, `racial/ethnic`, `religious`, `gender/sexual`, `other` |
| 3 | How is it expressed? | multi-label | `stereotype`, `vilification`, `dehumanization`, `extreme_language`, `lack_of_empathy`, `invalidation` |

## Folder contents

| Path | What it is |
|---|---|
| [final/raw-combined.csv](final/raw-combined.csv) | Stage 1: every contributor's file in one format, with `source` (resource link) and `annotator_file` |
| [final/final_dataset.csv](final/final_dataset.csv) | Merged dataset with all labels and `key_phrase` |
| [final/task1_polarization.csv](final/task1_polarization.csv), [task2_type.csv](final/task2_type.csv), [task3_manifestation.csv](final/task3_manifestation.csv) | Stage 2: de-duplicated model-ready files (`id`, `text`, labels) |
| [baseline_traditional_ml/](baseline_traditional_ml/) | TF-IDF + Logistic Regression / Linear SVM notebook and the train / val / test `splits/` used by every model |
| [deep_models/](deep_models/) | Bi-LSTM, Transformer and XLM-R training code, result files, logs, and the comparison notebook |
| `myPolar_dataset.zip`, `baseline_traditional_ml.zip` | Zipped copies of `final/` and `baseline_traditional_ml/` |

## Data

**Stage 1** (`raw-combined.csv`, 12,010 rows from 25 files) keeps provenance: `original_id`, `source`, `annotator_file`, `key_phrase` and all 12 labels.

**Stage 2** (task files) keeps only what a model needs.

| File | Rows | Positives |
|---|---|---|
| `task1_polarization.csv` | 10,710 | polarized 5,492 / not polarized 5,218 |
| `task2_type.csv` | 10,710 | political 2,655 · other 1,572 · racial/ethnic 904 · gender/sexual 651 · religious 389 |
| `task3_manifestation.csv` | 10,116 | vilification 3,095 · extreme_language 1,826 · invalidation 1,636 · stereotype 1,340 · lack_of_empathy 1,147 · dehumanization 976 |

Processing between the stages:

- **Duplicates removed.** 1,000 rows with identical text were dropped (11,710 to 10,710), so no sentence can appear in both train and test.
- **Labels made consistent.** Type and manifestation labels are cleared on rows marked not polarized (about 1,900 and 370 rows).
- **Task 3 is smaller.** Two contributor files have no manifestation labels, so their rows are left out of Task 3.
- **Splits.** 80 / 10 / 10 train / val / test, stratified, made separately for each task.

## Models

| Model | Input | Notes |
|---|---|---|
| TF-IDF + LogReg / LinearSVC (baseline) | word 1–2-grams, char 2–5-grams | C chosen on validation |
| Bi-LSTM | syllables | 2 layers, 128 hidden units per direction, mean + max pooling, trained from scratch |
| Transformer | syllables | 3 encoder layers, 4 heads, d = 192, trained from scratch |
| XLM-RoBERTa base | subwords, spaces between Myanmar characters removed | fine-tuned for 4 epochs at lr 2e-5; token embeddings frozen to fit in 16 GB |

The contributors segmented their text differently (word-segmented, phrase-segmented, or not at all), so the neural models ignore the original spacing: the Bi-LSTM and Transformer re-tokenise into syllables, and XLM-R reads unsegmented text.

All models share the same splits and protocol: the hyperparameter or epoch is chosen on validation macro F1, and the test set is scored once at that point. Task 1 macro F1 is over the two classes; Tasks 2 and 3 are macro over the labels. Positive classes are up-weighted in every model and the decision threshold is 0.5.

## Results

Test macro F1 on the shared random splits. Bi-LSTM and Transformer are the mean ± standard deviation of 3 seeds; the baseline and XLM-R are single runs.

| Model | Task 1 polarization | Task 2 type | Task 3 manifestation |
|---|---|---|---|
| TF-IDF + linear (baseline) | **0.790** | **0.572** | **0.509** |
| Bi-LSTM | 0.759 ± 0.001 | 0.522 ± 0.005 | 0.482 ± 0.013 |
| Transformer (from scratch) | 0.760 ± 0.012 | 0.473 ± 0.005 | 0.447 ± 0.003 |
| XLM-RoBERTa base (fine-tuned) | 0.770 | 0.560 | 0.481 |

![Test macro F1 by task and model](deep_models/results/model_comparison.png)

Task 1 with whole contributors held out (validation and test contributors never seen in training; one split, one seed):

| Model | Random split | Unseen contributors | Drop |
|---|---|---|---|
| TF-IDF + linear (char, LogReg) | 0.790 | 0.602 | 0.188 |
| Bi-LSTM | 0.759 | **0.662** | 0.097 |
| Transformer (from scratch) | 0.760 | 0.629 | 0.131 |
| XLM-RoBERTa base (fine-tuned) | 0.770 | 0.658 | 0.112 |

Per-label scores, validation scores and selected epochs are in [deep_models/results_comparison.ipynb](deep_models/results_comparison.ipynb).

### Findings

1. **The TF-IDF baseline was not beaten on the random splits.** It leads on all three tasks. XLM-R comes closest (0.02 behind on Task 1, 0.01 on Task 2); the Bi-LSTM is about 0.03 to 0.05 behind.
2. **A Transformer trained from scratch is the weakest model** on the two multi-label tasks. About 8,500 training sentences are too few for it without pre-training.
3. **Scores fall for every model on unseen contributors,** by 0.10 to 0.19, and the ranking reverses: the baseline drops the most (0.790 to 0.602) while the Bi-LSTM and XLM-R hold up best (about 0.66). The polarization rate ranges from 0% to 99% between contributors, so on a random split part of the score comes from recognising a contributor's source and writing style, and character n-grams are the best at that.
4. **Difficulty rises from Task 1 to Task 3.** Detecting polarization reaches about 0.79, naming its type about 0.57, and identifying how it is expressed about 0.51.
5. **Explicit labels are easier than implicit ones.** `political` and `vilification` reach 0.65 to 0.69 for the best models, while `stereotype`, `lack_of_empathy` and `invalidation` stay below 0.47 for every model. XLM-R is the best model on `gender/sexual` (0.64 vs 0.60) and `vilification` (0.67 vs 0.65).
6. **Small differences are within noise.** The test sets have about 1,000 sentences, and XLM-R's validation and test scores on Task 1 differ by 0.04 (0.811 vs 0.770), so gaps of 0.01 to 0.02 between models should not be read as real.

Caveats: XLM-R was run once per task with no hyperparameter search and with frozen token embeddings because of memory limits, so its numbers are a lower bound on what fine-tuning can do. The unseen-contributor XLM-R run used a 5-epoch schedule instead of 4.

## How to reproduce

```bash
pip install torch transformers scikit-learn pandas matplotlib

# 1. Baseline: run baseline_traditional_ml/baseline_traditional_ml.ipynb

# 2. Neural models (results are appended to deep_models/results/*.csv)
cd deep_models
python polar_models.py --model bilstm      --task all --seeds 42 43 44 --out results/bilstm.csv
python polar_models.py --model transformer --task all --seeds 42 43 44 --out results/transformer.csv
python polar_models.py --model xlmr        --task all --seeds 42       --out results/xlmr.csv

# 3. Unseen-contributor check for Task 1
python contributor_holdout.py
for m in bilstm transformer xlmr; do
  python polar_models.py --model $m --task task1 --splits splits_contributor --out results/contributor_holdout.csv
done

# 4. Tables and figure: run deep_models/results_comparison.ipynb
```

Trained on an Apple M4 (16 GB) with PyTorch MPS. The Bi-LSTM and Transformer take 2–8 minutes per run; XLM-R takes about 20–30 minutes per task.

## Known limitations

- **Provenance gaps.** Row `id`s were renumbered between `raw-combined.csv` and `final_dataset.csv`, so the two files must be joined on `original_id` or text, not `id`. `final_dataset.csv` contains 300 `mya_hanzawhtet` rows that are not in the raw file, the raw file contains 600 `mya_AUNGKOKOO` rows that are not in the final file, and two contributor files have no `original_id`.
- **Seven ids differ** between `task3_manifestation.csv` and the other two task files for the same sentence.
- **Conflicting duplicates.** 32 duplicated sentences had different polarization labels from different contributors; 24 rows in `task1_polarization.csv` therefore differ from `final_dataset.csv`.
- **Incomplete labels.** 342 polarized rows have no type and 568 have no manifestation.
- **Splits differ per task.** A sentence in the Task 3 test set can be in the Task 1 training set, so these splits should not be used for multi-task training.
- **Single annotation.** Almost every sentence was labelled by one person. The only double-annotated set (599 sentences, two contributors) agrees closely on polarization (Cohen's kappa 0.97).

---

## Assignment brief

- ဆရာတို့ [Assignment-2](https://github.com/ye-kyaw-thu/AIE-F-B2/tree/main/assignment/assignment-2) အနေနဲ့ ပြင်ဆင်ခဲ့ကြတဲ့ myPolar ဒေတာ အကြမ်းကိုပဲ သုံးပြီး Task-1, Task-2, Task-3 လုပ်ကြည့်ကြရအောင်။
- Task-1, Task-2, Task-3 နဲ့ ပတ်သက်ပြီး  ဒီလင့်ကို [https://polar-semeval.github.io/tasks.html](https://polar-semeval.github.io/tasks.html) refer လုပ်ပါ။
- အတန်းထဲမှာ ရှင်းပြခဲ့သလိုပဲ နည်းနည်း ခေါင်းစားမယ့် အပိုင်းက တစ်ယောက်ချင်းစီ ပြင်ဆင်ထားတဲ့ ဒေတာတွေကို အရင်ဆုံး format ညှိရမယ့် အပိုင်းပါ။ ဘယ်စာကြောင်းက ဘယ်သူက ပြင်ဆင်ထားတယ်ဆိုတဲ့ information လည်း ပါဝင်ပါစေ။ ဆရာ နားလည်တာက ဒေတာပြင်ဆင်တာကို အဆင့် နှစ်ဆင့် အနေနဲ့ ပြင်ဆင်ရလိမ့်မယ်။ ပထမအဆင့်မှာက ဘယ်ကနေ ယူထားတယ်ဆိုတဲ့ resource link, data annotator စတာတွေပါ အားလုံး ပါတဲ့ format အဖြစ်ပြောင်းလဲ ရလိမ့်မယ်။ ပြီးမှ မော်ဒယ်ဆောက်ဖို့အတွက် လိုအပ်တဲ့ အပိုင်းကိုပဲ နောက်ထပ် ဒုတိယ အဆင့်အနေနဲ့ ပြင်ဆင်ရလိမ့်မယ်။ အဲဒီ ဒုတိယ အဆင့် format မှာတော့ annotator name, resource link တွေ မပါလည်း ရပါတယ်။
- Group project အနေနဲ့ သွားကြရအောင်။
- Classification Model အပိုင်းကိုတော့ traditional ML method တွေကို baseline ထားလိုက်ပြီး experiment အနေနဲ့က Bi-LSTM, Transformer, Pretrained model finetuning စသည်ဖြင့် လုပ်တတ်သလို လုပ်ကြည့်ကြပါ။ ရလဒ်ကိုတော့ comparison လုပ်ကြည့်ပြီး ကိုယ့်အဖွဲ့အနေနဲ့ ရှာဖွေတွေ့ရှိတာကို report ရေးပါ။ Presentation လုပ်ပါ။

### Submission deadline

- ပြင်ဆင်ခဲ့တဲ့ ဒေတာနဲ့ ရေးခဲ့တဲ့ code, Jupyter notebook တွေအားလုံးကို zip လုပ်ပြီး သက်ဆိုင်ရာ group folder အောက်မှာ တင်ပါ။ အခက်အခဲရှိရင် အီးမေးလ်နဲ့ တင်ပါ။
- AI Engineering (Fundamental), Batch-2 အတန်း နောက်ဆုံးနေ့ (11 Oct 2026) မှာ presentation လုပ်ကြရအောင်။

## References

1. Assignment-2 Link: [https://github.com/ye-kyaw-thu/AIE-F-B2/tree/main/assignment/assignment-2](https://github.com/ye-kyaw-thu/AIE-F-B2/tree/main/assignment/assignment-2)
2. [POLAR: Detecting Multilingual, Multicultural and Multievent Online Polarization @ SemEval-2026 Task 9](https://polar-semeval.github.io/)
3. SemEval-2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization: [https://arxiv.org/html/2604.06817v1](https://arxiv.org/html/2604.06817v1)
4. POLAR: A Benchmark for Multilingual, Multicultural, and Multi-Event Online Polarization: [https://aclanthology.org/2026.findings-acl.1433.pdf](https://aclanthology.org/2026.findings-acl.1433.pdf)
5. Conneau et al. (2020), Unsupervised Cross-lingual Representation Learning at Scale (XLM-RoBERTa): [https://aclanthology.org/2020.acl-main.747/](https://aclanthology.org/2020.acl-main.747/)
6. Vaswani et al. (2017), Attention Is All You Need: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
7. Ye Kyaw Thu, sylbreak (Myanmar syllable segmentation rule used for the Bi-LSTM and Transformer input): [https://github.com/ye-kyaw-thu/sylbreak](https://github.com/ye-kyaw-thu/sylbreak)
8. XLM-RoBERTa base model: [https://huggingface.co/FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base)
9. Pedregosa et al. (2011), Scikit-learn: Machine Learning in Python: [https://jmlr.org/papers/v12/pedregosa11a.html](https://jmlr.org/papers/v12/pedregosa11a.html)
