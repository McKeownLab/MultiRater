import numpy as np
import pandas as pd
import pingouin as pg


def icc_c1_per_task(
    long_df: pd.DataFrame,
    raters: list[str],
    task_col: str = "task",
    score_col: str = "score",
    subject_col: str = "subject_id",
    rater_col: str = "rater",
) -> pd.DataFrame:
    """Consistency ICC(C,1) (pingouin "ICC(C,1)", = ICC3) per task item.

    Complete cases only: a subject enters a task only if every rater in `raters` scored it
    (46-52 subjects per item for the five raters). Sessions of the same subject are averaged.
    """
    rows = []
    for task, task_df in long_df.groupby(task_col):
        wide = task_df.pivot_table(
            index=subject_col,
            columns=rater_col,
            values=score_col,
            aggfunc="mean",
        )
        wide = wide.reindex(columns=raters).dropna(how="any")
        if len(wide) < 2:
            rows.append(
                {
                    "task": task,
                    "icc": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "n_subjects": len(wide),
                }
            )
            continue

        melted = wide.reset_index().melt(
            id_vars=subject_col, var_name=rater_col, value_name=score_col
        )
        icc_table = pg.intraclass_corr(
            data=melted,
            targets=subject_col,
            raters=rater_col,
            ratings=score_col,
        )
        icc_row = icc_table[icc_table["Type"] == "ICC(C,1)"].iloc[0]
        ci95 = np.asarray(icc_row["CI95"], dtype=float)
        rows.append(
            {
                "task": task,
                "icc": float(icc_row["ICC"]),
                "ci95_low": float(ci95[0]),
                "ci95_high": float(ci95[1]),
                "n_subjects": len(wide),
            }
        )
    return pd.DataFrame(rows)


# Deprecated name kept for old scripts; the statistic is ICC(C,1), not ICC(2,1).
icc2_1_per_task = icc_c1_per_task
