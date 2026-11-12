"""A controlled GPT-2 residual-stream patching experiment. No training."""
import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import try_to_load_from_cache


def hidden(output):
    return output[0] if isinstance(output, tuple) else output


def replacement(output, value):
    return (value,) + output[1:] if isinstance(output, tuple) else value


def forward(model, tokens, intervention=None, capture=False):
    """Patch one block output; hooks are removed even if inference fails."""
    cache, handles = {}, []
    if capture:
        for layer, block in enumerate(model.transformer.h):
            def save(_module, _args, output, layer=layer):
                cache[layer] = hidden(output).detach().clone()
            handles.append(block.register_forward_hook(save))
    if intervention is not None:
        layer, positions, source = intervention
        def patch(_module, _args, output):
            value = hidden(output).clone()
            for batch, pos in enumerate(positions):
                value[batch, pos] = source[batch, pos]
            return replacement(output, value)
        handles.append(model.transformer.h[layer].register_forward_hook(patch))
    try:
        with torch.inference_mode():
            logits = model(input_ids=tokens, use_cache=False).logits[:, -1].detach()
        return logits, cache
    finally:
        for handle in handles:
            handle.remove()


def dataset(tokenizer):
    pairs = [('John', 'Mary'), ('Tom', 'James'), ('Sarah', 'David'), ('Alice', 'Bob')]
    templates = [
        'When {a} and {b} went to the store, {subject} gave a book to',
        'After {a} and {b} arrived at the park, {subject} handed a drink to',
    ]
    rows = []
    for t, template in enumerate(templates):
        for a, b in pairs:
            for a, b in [(a, b), (b, a)]:
                ids = [tokenizer.encode(' ' + name, add_special_tokens=False) for name in [a, b]]
                if any(len(x) != 1 for x in ids):
                    raise ValueError(f'Names must be single tokens: {a}, {b}')
                clean = template.format(a=a, b=b, subject=a)
                corrupt = template.format(a=a, b=b, subject=b)
                x, y = [tokenizer.encode(text, add_special_tokens=False) for text in [clean, corrupt]]
                changed = [i for i, (u, v) in enumerate(zip(x, y)) if u != v]
                if len(x) != len(y) or len(changed) != 1:
                    raise ValueError('Prompt pair must differ at exactly one aligned token')
                rows.append(dict(split='discovery' if t == 0 else 'validation',
                                 clean=clean, corrupt=corrupt, clean_ids=x, corrupt_ids=y,
                                 answer=ids[1][0], alternative=ids[0][0], subject=changed[0],
                                 first_name=x.index(ids[0][0]), second_name=x.index(ids[1][0])))
    return rows


def run_group(model, tokenizer, rows, all_positions):
    device = next(model.parameters()).device
    clean = torch.tensor([r['clean_ids'] for r in rows], device=device)
    corrupt = torch.tensor([r['corrupt_ids'] for r in rows], device=device)
    answers = torch.tensor([r['answer'] for r in rows], device=device)
    alternatives = torch.tensor([r['alternative'] for r in rows], device=device)
    batch = torch.arange(len(rows), device=device)
    def scores(logits):
        return (logits[batch, answers] - logits[batch, alternatives]).cpu()
    clean_logits, clean_cache = forward(model, clean, capture=True)
    corrupt_logits, corrupt_cache = forward(model, corrupt, capture=True)
    baseline_clean, baseline_corrupt = scores(clean_logits), scores(corrupt_logits)
    sites = {name: [r[name] for r in rows] for name in ['first_name', 'second_name', 'subject']}
    sites['last'] = [clean.shape[1] - 1] * len(rows)
    if all_positions:
        sites.update({f'position_{i}': [i] * len(rows) for i in range(clean.shape[1])})
    effects = []
    for layer in range(len(model.transformer.h)):
        # Restoring a run's own activation must leave predictions unchanged.
        control, _ = forward(model, corrupt, (layer, sites['last'], corrupt_cache[layer]))
        error = (control - corrupt_logits).abs().max().item()
        if error > 1e-4:
            raise AssertionError(f'Self-patch control failed at layer {layer}: {error}')
        for site, positions in sites.items():
            restored, _ = forward(model, corrupt, (layer, positions, clean_cache[layer]))
            disrupted, _ = forward(model, clean, (layer, positions, corrupt_cache[layer]))
            rs, ds = scores(restored), scores(disrupted)
            gap = (baseline_clean - baseline_corrupt).mean().item()
            effects.append(dict(layer=layer, site=site, self_patch_max_error=error,
                restored_scores=rs.tolist(), disrupted_scores=ds.tolist(),
                restoration_effect=(rs-baseline_corrupt).mean().item(),
                disruption_effect=(baseline_clean-ds).mean().item(),
                recovery_fraction=((rs-baseline_corrupt).mean().item()/gap if abs(gap)>1e-4 else None)))
        print(f"{rows[0]['split']}: finished layer {layer}", flush=True)
    return dict(split=rows[0]['split'], examples=rows,
                clean_scores=baseline_clean.tolist(), corrupt_scores=baseline_corrupt.tolist(),
                clean_pairwise_accuracy=(baseline_clean>0).float().mean().item(),
                corrupt_pairwise_accuracy=(baseline_corrupt<0).float().mean().item(),
                effects=effects)


def plot(groups, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sites = ['first_name', 'second_name', 'subject', 'last']
    fig, axes = plt.subplots(len(groups), 2, figsize=(10, 4*len(groups)), squeeze=False)
    for row, group in enumerate(groups):
        for col, field in enumerate(['restoration_effect', 'disruption_effect']):
            values = [[e[field] for e in group['effects'] if e['layer']==layer and e['site'] in sites]
                      for layer in sorted({e['layer'] for e in group['effects']})]
            limit = max(abs(x) for line in values for x in line) or 1
            im = axes[row,col].imshow(values, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
            axes[row,col].set(title=f"{group['split']}: {field.replace('_', ' ')}",
                              ylabel='Block output (zero indexed)', xticks=range(len(sites)), xticklabels=['first\nname', 'second\nname', 'subject', 'last'])
            fig.colorbar(im, ax=axes[row,col], label='Mean logit-difference change')
    fig.tight_layout(); fig.savefig(destination, dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='openai-community/gpt2')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--all-positions', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'results')
    args = parser.parse_args()
    torch.manual_seed(0); torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                    use_safetensors=True, attn_implementation='eager').to(args.device).eval()
    if model.config.model_type != 'gpt2':
        raise ValueError('This experiment supports GPT-2 architecture only')
    cached_config = try_to_load_from_cache(args.model, 'config.json', revision=args.revision)
    revision = (Path(cached_config).parent.name if isinstance(cached_config, str)
                else getattr(model.config, '_commit_hash', None) or args.revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    rows = dataset(tokenizer)
    # Separate groups avoid padding and keep both templates in the report.
    groups = [run_group(model, tokenizer, [r for r in rows if r['split']==split], args.all_positions)
              for split in ['discovery', 'validation']]
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(model=args.model, revision=revision, torch=torch.__version__,
                  transformers=transformers.__version__, device=args.device,
                  parameters=sum(p.numel() for p in model.parameters()), groups=groups)
    (args.output/'patching.json').write_text(json.dumps(report, indent=2)+'\n')
    plot(groups, args.output/'patching.png')
    for g in groups:
        print(g['split'], 'clean/corrupt pairwise accuracy:', g['clean_pairwise_accuracy'], g['corrupt_pairwise_accuracy'])
    print('Results:', args.output)


if __name__ == '__main__':
    main()
