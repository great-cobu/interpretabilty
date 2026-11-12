# Research and experiment guide

## Status and model

Proposed model: Qwen/Qwen3-8B. Lighter alternative: Qwen/Qwen3-4B.
Hardware: MacBook M4 Max with 48 GB unified memory.
No completed Qwen activation-patching study is reported here.

At 16-bit precision the weights alone need roughly 16 GB or 8 GB respectively.
Allow additional memory for activations, caches, runtime buffers, and macOS.
Start with short prompts, batch size one, and selective caching. Measure actual
peak memory and validate the hooks and numerical behavior before scaling up.

## Approach

1. Define a narrow behavior and establish its baseline.
2. Find candidate representations and computations.
3. Intervene and measure predicted consequences.
4. Connect components into a computational explanation.
5. Test beyond the discovery examples, with general-damage controls.

## Proposed task

Compare “When John and Mary went to the store, John gave a book to” with the
same sentence using Mary as the repeated subject. The expected recipients
are Mary and John respectively. Validate tokenization and model input format.
Single-token candidates permit logit-difference scoring; multi-token names
require appropriate sequence scoring or replacement candidates.

Keep name order fixed within each pair when calculating D, the logit of the
clean recipient minus the logit of the alternative. The restoration effect is
D_patched - D_contrasting. The disruption effect is D_clean - D_reverse_patched.
Recovery fractions are unstable when the baseline gap is small; report raw
scores and all examples. A self-patch must leave predictions unchanged.

The initial design proposes four name pairs in both orders and two sentence
templates: 16 prompt pairs, not a comprehensive benchmark. Expand validation
before claiming a general mechanism. Check new names, paraphrases, distractors,
and effects on unrelated behaviors. Causal patches can create unusual states.

## Possible geometry extension

After a causal hypothesis is established, examine local downstream sensitivity
and whether a position-dependent metric improves on a linear account. Geometry
is a tool for answering the behavioral question, not evidence by itself that a
direction has a particular linguistic meaning.
