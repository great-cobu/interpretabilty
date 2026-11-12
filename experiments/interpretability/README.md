# First language-model interpretability experiment

This experiment uses pretrained GPT-2 Small (124M parameters), without training
or changing its saved weights. It asks where the identity of a sentence's
recipient affects the next-token prediction.

## Run

From the repository root, create an isolated Python 3.12 environment:

```sh
uv venv experiments/interpretability/.venv --python 3.12
uv pip install --python experiments/interpretability/.venv/bin/python -r experiments/interpretability/requirements.txt
HF_HOME=experiments/interpretability/.cache/huggingface experiments/interpretability/.venv/bin/python experiments/interpretability/patching.py
```

The first run downloads approximately 500 MB of model weights. CPU is the default;
`--device mps` selects Apple Silicon acceleration, and `--device cuda` selects
CUDA where available. CPU was chosen for the initial reference experiment.
`--all-positions` adds every token position to the JSON scan (more forward passes).
The plotted summary always shows the same four named positions.

Outputs use stable paths under `results/`: `patching.json` and `patching.png`.
The JSON records the resolved model revision, package versions, all prompts,
individual scores, and aggregate effects. Reproduce the same weights with
`--revision <revision-from-patching.json>` rather than relying on `main`.

## Question and controls

Clean example: “When John and Mary went to the store, John gave a book to”

Contrasting example: “When John and Mary went to the store, Mary gave a book to”

The expected recipient changes from Mary to John. The score is the next-token
logit of the clean recipient minus that of the other name. It is a pairwise
comparison, not unrestricted next-token accuracy. Tokenization is checked:
name answers must be single tokens and the paired inputs must differ at exactly
one aligned position. Four name pairs appear in both name orders. A second
sentence template provides a small validation set; these 16 examples are not
a statistically comprehensive benchmark.

For each block output and selected position, the experiment performs:

* Restoration: insert the clean activation into the contrasting run.
* Disruption: insert the contrasting activation into the clean run.
* Identity control: patch the contrasting run with its own final-position
  activation, which must leave all next-token logits unchanged at every layer.

The heatmap reports raw changes in the logit difference. Positive effects support
the expected direction of the intervention. JSON also includes aggregate recovery
fractions; these are unstable when the baseline gap is small and can exceed one.
Always inspect the raw baselines, both patch directions, and individual examples.
All examples are retained, including failures; the report does not select only
prompts the model answers correctly.

A final-block final-token patch directly replaces the representation used by the
output head. Its recovery is an implementation check, not evidence locating an
upstream linguistic algorithm. Earlier-layer effects are intervention-specific
localization evidence, not proof of necessity or sufficiency across all contexts.
Patches can create unusual hidden states; redundant pathways can mask effects.

## Next research steps

1. Establish that the baseline behavior is consistent across a larger, held-out
   prompt set, including controls for name frequency and surface wording.
2. Use the layer/position results to choose a narrower attention-head or MLP
   experiment. Test explicit competing computational explanations.
3. Validate a proposed circuit with interventions on new templates and check
   unrelated behaviors for general damage.
4. Only then examine geometric structure: compare local downstream sensitivity
   across contexts or estimate Fisher-metric products along candidate directions.
   High sensitivity alone does not establish a semantic interpretation.

References:

* Model: https://huggingface.co/openai-community/gpt2
* Established task/demo: https://transformerlensorg.github.io/TransformerLens/generated/demos/Main_Demo.html
* Patching interpretation: https://arxiv.org/abs/2404.15255
* Patching metrics: https://arxiv.org/abs/2309.16042

The implementation uses ordinary PyTorch block-output hooks on a Hugging Face
GPT-2 model, rather than depending on TransformerLens-specific interfaces.

## Initial CPU run

Both templates achieved 8/8 correct pairwise choices on clean prompts and 8/8
on contrasting prompts. Mean clean-minus-alternative logit differences were
+2.659 (discovery) and +2.950 (validation); contrasting inputs gave -2.901 and
-2.902. All identity-patch controls had zero maximum logit error.

The repeated-subject position has strong effects at early block outputs; the
final position has strong effects later, with a transition around blocks 7–8
(zero indexed). This pattern motivates finer attention-head interventions. It
does not identify a complete circuit. In particular, early subject patches may
simply transfer lexical identity. The unchanged initial-name positions have
zero effects as expected from causal masking: their activations cannot depend
on the later subject substitution. They are useful negative controls, not
evidence that initial names are unimportant to recipient identification.

Run the independent, download-free hook test with:

```sh
experiments/interpretability/.venv/bin/python -m unittest discover -s experiments/interpretability -p 'test_*.py'
```
