from pathlib import Path
from types import SimpleNamespace

import pytest

from models.MNIST.page_export import export_page, report_html


def test_snapshot_preserves_selected_state_and_escapes_text():
    page = report_html(theme="", run_name="<script>run</script>", summary=None,
                       config="<custom>", digit=7, sample_index=321, threads=8,
                       heatmap_mode="Percent of true digit", preview=b"image",
                       heatmap=None, learning_curve=None, architecture="CNN")
    assert "<script>run</script>" not in page
    assert "&lt;script&gt;run&lt;/script&gt;" in page
    assert "<strong>7</strong>; validation example index: 321" in page
    assert "Percent of true digit" in page
    assert "data:image/png;base64," in page
    assert "&lt;custom&gt;" in page


def test_default_name_and_repeated_exports_preserve_files(tmp_path, monkeypatch):
    def render(args, **kwargs):
        Path(args[-2]).write_bytes(b"rendered snapshot")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr("mnist_cnn.page_export.subprocess.run", render)
    folder = tmp_path / "new folder"
    first = export_page("<html></html>", str(folder), "", "pdf", tmp_path)
    second = export_page("<html></html>", str(folder), "mnist-demo.pdf", "png", tmp_path)
    repeated = export_page("<html></html>", str(folder), "mnist-demo", "pdf", tmp_path)
    assert first.name == "mnist-demo.pdf"
    assert second.name == "mnist-demo.png"
    assert repeated.name == "mnist-demo-1.pdf"
    assert first.read_bytes() == b"rendered snapshot"


@pytest.mark.parametrize("name", ["../escape", "folder/name", "folder\\name", ".", "..", "\0", ".pdf"])
def test_invalid_filename_writes_nothing(tmp_path, name):
    with pytest.raises(ValueError):
        export_page("", str(tmp_path / "absent"), name, "pdf", tmp_path)
    assert not (tmp_path / "absent").exists()


def test_render_failure_does_not_create_output(tmp_path, monkeypatch):
    monkeypatch.setattr("mnist_cnn.page_export.subprocess.run", lambda *a, **k:
                        SimpleNamespace(returncode=1, stderr="Missing Chromium"))
    with pytest.raises(RuntimeError, match="Missing Chromium"):
        export_page("", str(tmp_path), "report", "pdf", tmp_path)
    assert not (tmp_path / "report.pdf").exists()
