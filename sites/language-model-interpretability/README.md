# Language-model interpretability research website

Static documentation for the completed GPT-2 Small experiment and proposed
Qwen3-8B study. A lighter future alternative is Qwen3-4B. The site distinguishes
completed evidence from research plans.

## Present locally

```sh
cd /Users/frankelliott/Documents/Projects/Python/interpetability
python3 -m http.server 8767 --bind 127.0.0.1 --directory sites/language-model-interpretability/dist
```

Open http://127.0.0.1:8767 and share the browser window in Zoom. Keep Terminal
running; press Control-C to stop. Use port 8768 in both places if necessary.

Edit dist/index.html and dist/style.css. The page has no runtime dependencies.
The downloads include the GPT-2 guide, code, environment, results, and a separate
Qwen study plan. No results are presented as completed Qwen research. Preserve the five-stage
research framework and the distinction between plans, observations, and claims.
The portable ZIP contains the same site. Hosting remains unpublished.
