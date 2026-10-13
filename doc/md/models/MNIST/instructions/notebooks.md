# Interactive Marimo demo

The notebook uses the installed `mnist_cnn` package to preview cached MNIST validation digits and checkpoint predictions on CPU. A run selector displays saved training metrics, learning curves, resolved configuration, and resume provenance. It also shows the CNN architecture. Run the [preliminary demos](demos.md) first to prepare the cache and create training artifacts.

## Install the optional notebook environment

```bash
cd /Users/frankelliott/Documents/Projects/Python/interpetability
uv sync --locked --no-editable --group notebooks --reinstall-package mnist-cnn
PLAYWRIGHT_BROWSERS_PATH=/Users/frankelliott/Documents/Projects/Python/interpetability/runs/.browser-cache uv run --locked --no-editable --group notebooks python -m playwright install chromium
```

## Open the interactive demo

Run this command in a terminal. Keep the process running and open the local URL it prints in your browser:

```bash
uv run --directory /Users/frankelliott/Documents/Projects/Python/interpetability --locked --no-editable --group notebooks marimo run /Users/frankelliott/Documents/Projects/Python/interpetability/notebooks/mnist_demo.py
```

Choose the **Training epochs** and **Training CPU threads**, then press **Train new demo run** to train a fresh CPU model. Epochs can be selected from 1 to 100 (default 2). This uses 20 training batches and five validation batches per epoch, saves a new run, and automatically selects it. Changing controls alone does not train; opening or exporting the notebook also does not train. Training begins only when you submit the form. Each submission creates a fresh run rather than resuming an existing model.

The validation digit and prediction plots appear first in the results, followed by saved metrics, learning curves, and the validation heat map.

Choose a saved run, then select the true validation digit (0–9) with the radio buttons to compare the true digit and predicted probabilities. Selecting 7, for example, always shows an example whose true label is 7. Press **Next example** to browse different handwriting for that digit; the counter wraps after the last matching example. Changing the digit or saved run resets browsing to the first matching example. The newest saved training run is selected initially. Reopen the notebook after generating additional runs to refresh the run list. Predictions use `checkpoints/best.ckpt`; validation examples use the checkpoint’s data configuration, including its split seed and normalization. Keep the checkpoint’s cache and any configured split-index file available. Without a checkpoint, the preview uses the project’s default cached dataset.

The annotated heat map covers all 5,000 validation examples and all ten digits, regardless of training or validation smoke-run batch limits. Rows represent true digits and columns predicted digits. Switch the heat map between counts and percentages within each true digit; its diagonal shows correct predictions. The displayed accuracy is measured across the full validation set, so it can differ from the saved demo metrics.

The CPU inference thread slider defaults to eight threads, capped by the available logical CPUs. Adjust it to suit your machine; more threads do not always make small batches faster. Changing the run or thread count recomputes the full validation matrix. Selecting a digit or pressing Next example or changing heat-map units reuses the existing counts. Restart the Marimo process after updating the package and notebook.

By default, the notebook reads local files and performs inference. Submitting the training form explicitly trains and writes a fresh run; ordinary browsing and HTML export do neither. It does not download MNIST, contact MLflow, or score the official test set. If the cache or training artifacts are absent, it shows preparation guidance and the model architecture.

To edit the notebook cells:

```bash
uv run --directory /Users/frankelliott/Documents/Projects/Python/interpetability --locked --no-editable --group notebooks marimo edit /Users/frankelliott/Documents/Projects/Python/interpetability/notebooks/mnist_demo.py
```

## Export the current results as PDF or PNG

Use **Export page** at the bottom of the notebook:

1. Choose **PDF** (paginated A4) or **PNG** (one full-height image).
2. Keep the default filename `mnist-demo`, or enter another name. A blank name also uses the default; the correct file extension is added automatically.
3. Select **Project exports**, **Desktop**, **Downloads**, or **Custom folder**. For a custom folder, enter its absolute path in the custom-folder field; `~` is also accepted. The default destination is `runs/exports/` in this project.
4. Press **Export page**. The notebook shows the saved filename and folder.

Exports contain the current run, chosen digit example and prediction, metrics, learning curves, heat map in its selected units, configuration, and model architecture. They use the shared notebook palette and a print-friendly layout without interactive controls. Exporting does not rerun training or inference. Existing files are preserved: repeated exports append `-1`, `-2`, and so on. Missing destination folders are created; unwritable destinations produce an error in the notebook.

The optional notebook environment includes Playwright, and the setup command above installs its Chromium renderer into the ignored project cache. Installation requires network access once; rendering then runs locally with embedded images and no external assets. Restart Marimo after updating the environment. The folder choices refer to the machine running Marimo.

## Export a snapshot

```bash
uv run --directory /Users/frankelliott/Documents/Projects/Python/interpetability --locked --no-editable --group notebooks marimo export html /Users/frankelliott/Documents/Projects/Python/interpetability/notebooks/mnist_demo.py -o /Users/frankelliott/Documents/Projects/Python/interpetability/runs/mnist_demo.html
```

The exported HTML contains the initial outputs as a static snapshot. Use `marimo run` for interactive controls. See Marimo's [script and export documentation](https://docs.marimo.io/guides/scripts/) and [app guide](https://docs.marimo.io/guides/apps/).

These paths match this checkout. On another machine, replace the absolute project prefix. Marimo remains optional; the CLI pipeline does not import it. Rich evaluation reports, MLflow integration, and the full baseline accuracy review remain separate planned work.

## Change the notebook colors

The palette is defined once, at the top of [theme.css](../../../../../notebooks/theme.css), as named CSS variables. Edit those values to change the notebook's colors without editing individual cells:

| Variable | Role |
| --- | --- |
| `--mnist-title` | Main title (`#` in Markdown), deep navy |
| `--mnist-heading` | Section headings (`##`), strong blue |
| `--mnist-subheading` | Subheadings (`###` and deeper), dark teal |
| `--mnist-supporting` | Supporting descriptions, slate gray |
| `--mnist-text` | Body text, dark charcoal |
| `--mnist-background` | Prose background, white |
| `--mnist-heading-background` | Background for all headings, pale blue |

All prose cells use the notebook's `notebook_md(text, supporting=False)` helper. Write ordinary Markdown headings instead of adding colors to individual headings. For short descriptive text, use `notebook_md(text, supporting=True)`; headings retain their level's color while paragraphs use the supporting color. The stylesheet also sets consistent heading sizes and weights.

The helper wraps prose in the notebook's own CSS classes, keeping the style rules independent of Marimo's internal class names. The notebook loads the stylesheet through `css_file="theme.css"`, relative to the notebook file, using [Marimo's custom CSS support](https://docs.marimo.io/guides/configuration/theming/). Styling applies in the editor, app, and HTML export. The palette is designed for white prose backgrounds; controls follow Marimo's theme and plots retain their existing colors. Restart or reload Marimo after editing the stylesheet, and regenerate exported HTML to update its embedded style.
