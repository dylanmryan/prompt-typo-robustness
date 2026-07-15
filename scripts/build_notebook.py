"""Build the analysis notebook programmatically so it stays in sync with the package."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Prompt Typo Robustness — Analysis\n"
        "Exploratory companion to the README. Loads the committed raw results and "
        "reproduces the headline statistics and figures."),
    nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "from typo_study.analysis import (accuracy_table, fit_logit, load_results,\n"
        "                                 fig_degradation, fig_typo_types, fig_heatmap)\n"
        "df = load_results(Path('../results/trials.jsonl'))\n"
        "df.head()"),
    nbf.v4.new_markdown_cell("## Accuracy by model and severity\n"
        "Wilson 95% intervals; `empty_rate` tracks refusal/format failures."),
    nbf.v4.new_code_cell("accuracy_table(df)"),
    nbf.v4.new_markdown_cell("## Is the degradation statistically significant?\n"
        "Logistic regression with cluster-robust standard errors grouped by item "
        "(each item appears under every severity and model, so trials are not independent)."),
    nbf.v4.new_code_cell("print(fit_logit(df))"),
    nbf.v4.new_markdown_cell("## Figures"),
    nbf.v4.new_code_cell(
        "import tempfile\n"
        "from IPython.display import Image, display\n"
        "tmp = Path(tempfile.mkdtemp())\n"
        "fig_degradation(df, tmp / 'deg.png')\n"
        "fig_typo_types(df, tmp / 'tt.png')\n"
        "fig_heatmap(df, tmp / 'hm.png')\n"
        "for p in ('deg.png', 'tt.png', 'hm.png'):\n"
        "    display(Image(str(tmp / p)))"),
    nbf.v4.new_markdown_cell(
        "## Response-length side note\n"
        "Do models get more or less verbose when prompts contain typos?"),
    nbf.v4.new_code_cell(
        "main = df[df.phase == 'main'].copy()\n"
        "main['resp_words'] = main.response.str.split().str.len()\n"
        "main.groupby(['model', 'severity']).resp_words.mean().unstack().round(1)"),
]
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
Path("notebooks").mkdir(exist_ok=True)
nbf.write(nb, "notebooks/analysis.ipynb")
print("wrote notebooks/analysis.ipynb")
