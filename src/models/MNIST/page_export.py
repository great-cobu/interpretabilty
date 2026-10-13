"""Export the current notebook results without rerunning training or inference.

Playwright is imported only by the rendering subprocess, keeping the runtime
CLI independent of the optional notebook dependencies.
"""

import base64
from html import escape
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def _image(image: bytes | None, description: str) -> str:
    if image is None:
        return ""
    encoded = base64.b64encode(image).decode("ascii")
    return f'<img alt="{escape(description)}" src="data:image/png;base64,{encoded}">'


def report_html(*, theme: str, run_name: str, summary: dict | None,
                config: str | None, digit: int, sample_index: int | None,
                threads: int, heatmap_mode: str, preview: bytes | None,
                heatmap: bytes | None, learning_curve: bytes | None,
                architecture: str, validation_accuracy: float | None = None) -> str:
    """Build a self-contained, printable snapshot of the selected UI results."""
    metrics = (summary or {}).get("metrics", {})
    rows = "".join(f"<tr><th>{escape(str(key))}</th><td>{float(value):.6g}</td></tr>"
                   for key, value in metrics.items())
    progress = (f"{summary['completed_epochs']} completed epochs; {summary['global_step']} optimizer steps; "
                f"smoke run: {summary['smoke_run']}" if summary else "No training results selected.")
    provenance = str((summary or {}).get("source_checkpoint") or "Initial training run")
    validation_score = f" Full-validation accuracy: {validation_accuracy:.2%}." if validation_accuracy is not None else ""
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>MNIST CNN demo</title><style>{theme}
body {{ margin: 0; background: white; font: 16px/1.6 system-ui, sans-serif; }}
main {{ max-width: 1000px; margin: auto; padding: 24px; }}
section {{ margin-bottom: 24px; }}
img {{ display: block; width: 100%; height: auto; break-inside: avoid; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ text-align: left; border-bottom: 1px solid #ddd; padding: 6px 10px; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font: 12px/1.5 monospace; }}
h1, h2, h3 {{ break-after: avoid; }}
@media print {{ main {{ padding: 0; }} section {{ break-inside: avoid; }}
pre {{ font-size: 11px; line-height: 1.35; }} }}
</style></head><body><main class="mnist-prose">
<h1>MNIST CNN interactive demo</h1>
<p>Run: <strong>{escape(run_name)}</strong>. CPU inference threads: {threads}.</p>
<section><h2>Validation digit and model prediction</h2>
<p>Selected true digit: <strong>{digit}</strong>; validation example index: {sample_index}.</p>
{_image(preview, 'Selected validation digit and predicted probabilities')}</section>
<section><h2>Saved results</h2><p>{escape(progress)}</p>
<table><tbody>{rows}</tbody></table><p>Resume provenance: {escape(provenance)}</p>
{_image(learning_curve, 'Training and validation learning curves')}</section>
<section><h2>Validation heat map</h2><p>All 5,000 validation examples; rows are true digits and columns predictions.
Display: {escape(heatmap_mode)}.{validation_score}</p>{_image(heatmap, 'Validation heat map for digits zero through nine')}</section>
<section class="config"><h3>Resolved configuration</h3><pre>{escape(config or 'Unavailable')}</pre></section>
<section class="config"><h2>CNN architecture</h2><pre>{escape(architecture)}</pre></section>
</main></body></html>'''


def export_page(html: str, folder: str, filename: str, file_format: str,
                browser_cache: Path) -> Path:
    """Save PDF or full-page PNG; preserve existing files by numbering repeats."""
    if file_format not in {"pdf", "png"}:
        raise ValueError("Choose PDF or PNG")
    name = filename.strip() or "mnist-demo"
    if name in {".", ".."} or any(char in name for char in '/\\\0'):
        raise ValueError("Use a filename without folder separators")
    if name.lower().endswith((".pdf", ".png")):
        name = name[:-4]
    if not name.strip():
        raise ValueError("Provide a filename before the extension")
    if not folder.strip():
        raise ValueError("Choose an export folder")
    destination = Path(folder).expanduser()
    if not destination.is_absolute():
        raise ValueError("Use an absolute destination folder path")
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="mnist-export-") as temporary:
        source = Path(temporary) / "page.html"
        rendered = Path(temporary) / f"page.{file_format}"
        source.write_text(html)
        env = dict(os.environ, PLAYWRIGHT_BROWSERS_PATH=str(browser_cache.resolve()))
        try:
            result = subprocess.run(
                [sys.executable, "-m", "models.MNIST.page_export", str(source), str(rendered), file_format],
                env=env, capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Page rendering timed out; retry the export") from exc
        if result.returncode:
            raise RuntimeError("Page rendering failed. Check that the notebook environment and Chromium are installed. "
                               + result.stderr[-1200:])
        content = rendered.read_bytes()
        suffix = 0
        while True:
            target = destination / f"{name}{f'-{suffix}' if suffix else ''}.{file_format}"
            try:
                with target.open("xb") as stream:
                    stream.write(content)
                return target
            except FileExistsError:
                suffix += 1


def _render(source: str, target: str, file_format: str) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1100, "height": 900}, device_scale_factor=2)
            # All assets are embedded; rendering makes no network requests.
            page.route("http://**/*", lambda route: route.abort())
            page.route("https://**/*", lambda route: route.abort())
            page.goto(Path(source).as_uri(), wait_until="load")
            page.evaluate("document.fonts.ready")
            if file_format == "pdf":
                page.pdf(path=target, format="A4", print_background=True,
                         margin={"top": "12mm", "bottom": "12mm", "left": "12mm", "right": "12mm"})
            else:
                page.screenshot(path=target, full_page=True)
        finally:
            browser.close()


if __name__ == "__main__":
    _render(*sys.argv[1:])
