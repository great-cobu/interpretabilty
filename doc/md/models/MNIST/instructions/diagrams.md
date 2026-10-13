# Graphviz diagrams

Graphviz is a system tool used only for documentation. It is not needed for dataset preparation or future model training. Install it through your operating system's package manager; on macOS:

```bash
brew install graphviz
```

Regenerate the committed SVG from its editable DOT source, from the project root:

```bash
dot -Tsvg doc/diagrams/data-flow.dot -o doc/diagrams/data-flow.svg
dot -Tsvg doc/diagrams/cnn-shapes.dot -o doc/diagrams/cnn-shapes.svg
dot -Tsvg doc/diagrams/training-checkpoints.dot -o doc/diagrams/training-checkpoints.svg
```

The [CNN shape diagram](../diagrams/cnn-shapes.svg) traces the network architecture and its loss/output branches. The [data-flow diagram](../diagrams/data-flow.svg) explains cache acquisition, splitting, transforms, and inspection. Update both files together and inspect the rendered labels and connections before committing. Graphviz 16.1.0 was used for the current rendering. No Python diagram dependency or notebook is needed.

The [training/checkpoint diagram](../diagrams/training-checkpoints.svg) shows validation selection, latest-state saving, resume into a new run, and standalone testing.
