# Contributing

Das Projekt ist ein Prototyp aus dem Studium, also gibt es bestimmt Dinge, die man besser machen kann. Jede Rückmeldung hilft, auch ein kurzes Issue.

## Issues

- **Fehler** im Code, in der Statistik oder in der Interpretation: Bitte schreib dazu, welches Experiment, welche Variante und welcher Seed betroffen sind und was du erwartet hättest.
- **Ideen für weitere Experimente:** Gerne erst als Issue, bevor du viel Zeit reinsteckst. Ein paar Ansätze stehen schon unter [Grenzen und Ideen](README.md#grenzen-und-ideen).

## Setup

Wie in der [README](README.md#deploy-it-yourself) beschrieben:

```bash
git clone https://github.com/dariandieck/ml-shuffling-regularization.git
cd ml-shuffling-regularization
uv sync
```

## Pull Requests

* Forken und einen Branch von `main` anlegen, zum Beispiel `feature/momentum` oder `fix/gap-metric`.
* In der Beschreibung kurz sagen, was du geändert hast und warum.

### Wo was hinkommt

| Vorhaben | Wo |
|---|---|
| neue Shuffling-Variante | Funktion mit der Signatur `(n, batch_size, gen) -> Tensor` in [`shuffling.py`](src/batch_shuffling_regularization/shuffling.py) und in `VARIANTS` eintragen. Für die Auswertung zusätzlich `SHUFFLE_VARIANTS` in [`analysis.py`](src/batch_shuffling_regularization/analysis.py) und `COLORS` in Notebook 04 ergänzen |
| neues Experiment (Datensatz, Batchgröße, Lernrate) | Eintrag in `EXPERIMENTS` in [`experiment.py`](src/batch_shuffling_regularization/experiment.py), neue Datensätze in `DATASETS` in [`data.py`](src/batch_shuffling_regularization/data.py) |
| neue Kennzahl | [`run_metrics`](src/batch_shuffling_regularization/analysis.py) |
| Plots und Auswertung | [`04_analysis.ipynb`](notebooks/04_analysis.ipynb) |

### Damit die Ergebnisse vergleichbar bleiben

- **Determinismus nicht anfassen.** Jeder Lauf muss bei gleichem Seed bit-identisch reproduzierbar bleiben. Neue Varianten ziehen ihre Zufallszahlen nur aus dem übergebenen `gen`, nie aus dem globalen RNG, sonst ist das gepaarte Design kaputt.
- **Gleiches Setup für alle Varianten.** Pro Experiment unterscheidet sich nur die Reihenfolge der Daten, alles andere bleibt gleich.
- **Ergebnisse nur bewusst ändern.** Wenn dein Change bestehende Läufe beeinflusst, das betroffene Experiment komplett neu rechnen (`results/<experiment>/runs/` löschen, dann `uv run shuffling run <experiment>`), die neuen CSVs mitcommitten und im PR erwähnen. Teilweise neu gerechnete Experimente bitte nicht mischen.
- **Plots neu erzeugen** mit `uv run shuffling analyse`, damit `figures/` und Notebook 04 zu den Daten passen.

### Code-Stil

- Python 3.12 mit Type Hints, kurze Funktionen.
- Code und Kommentare auf Englisch, Notebooks und README auf Deutsch (vorerst).
- Kommentare nur da, wo das *Warum* nicht offensichtlich ist.

## Lizenz

Das Projekt steht unter der [MIT-Lizenz](LICENSE). Mit einem Pull Request bist du einverstanden, dass dein Beitrag unter derselben Lizenz veröffentlicht wird.
