import pandas as pd
from dmml_project import PAPER_TABLES
from dmml_project.model_selection.load_results import model_name_from_index
import numpy as np
from pandas.io.formats.style import Styler

def dataframe_to_latex(df: pd.DataFrame, caption: str, label: str, columns_format: str | None = None, formatters: dict = {}, file: str | None = None) -> str:
    styler = Styler(df, escape="latex")
    styler \
        .hide(axis="index")
    for column, formatter in formatters.items():
        styler.format(formatter, subset=column)

    table = styler.to_latex(
        position="H", 
        column_format=columns_format, 
        position_float="centering", 
        hrules=True,
    )
    
    table = table.replace("\\centering", "\\centering\n\\capstart")
    table = table.replace("\\begin{tabular}", "\\begin{tabularx}{0.48\\textwidth}")
    table = table.replace("\\toprule","\\hline")
    table = table.replace("\\midrule","\\hline")
    table = table.replace("\\bottomrule","\\hline")
    
    table = table.replace("\\end{tabular}", 
        f"\\end{{tabularx}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
    )
    
    if file is not None:
        with open(file, "w") as f:
            f.write(table)
    return table
    
def single_hyper_to_latex(hypers: dict[str, list], caption: str, label: str) -> str:
    table = pd.DataFrame(columns=["Hyperparameter", "Search Space"])
    for key, values in hypers.items():
        table.loc[len(table)] = [key, values]
    return dataframe_to_latex(table, caption, label, columns_format="|l|X|", formatters=
    {
        "Hyperparameter": lambda x: f"\\texttt{{{x.replace('_','\\_')}}}",
        "Search Space": lambda x: f"\\{{{", ".join([str(xi).replace("_","\\_") for xi in x])}\\}}"
    })

def hyper_range_to_latex(hypers: list[dict[str, dict[str, list]]], captions: list[dict[str, str]]) -> str:
    ret = ""
    for gen, hyper_gen in enumerate(hypers):
        for model_kind, hypers in hyper_gen.items():
            ret += single_hyper_to_latex(hypers, captions[gen][model_kind], f"tab:hyperparameters_{model_kind}_{gen}")
            ret += "\n\n"
    return ret
    
def best_models_to_latex(best_model_index, best_accuracy, results_indices, similar_models, search_results):
    # generate table for docs
    with open(f"{PAPER_TABLES}/best_models.tex", "w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("    \\centering\n")
        f.write("    \\begin{tabular}{|l|r|}\n")
        f.write("        \\hline\n")
        f.write("        \\textbf{Model} & \\textbf{Accuracy} \\\\\n")
        f.write("        \\hline\n")
        f.write(f"        \\texttt{{{model_name_from_index(results_indices[best_model_index]).replace("_","\\_")}}} & {best_accuracy*100:.2f}\\% \\\\\n")
        for index in similar_models:
            gen, model_kind, idx = index
            accuracy = np.mean(search_results[gen][model_kind][idx][1])
            f.write(f"        \\texttt{{{model_name_from_index(index).replace("_","\\_")}}} & {accuracy*100:.2f}\\% \\\\\n")
        f.write("        \\hline\n")
        f.write("    \\end{tabular}\n")
        f.write("    \\caption{Best model (first row) and models not significantly different from the best model.}\n")
        f.write("    \\label{tab:best_models}\n")
        f.write("\\end{table}\n")