"""Validate, clean and summarise creforge credit datasets.

    from pipeline import run
    results = run("dirty", "output")
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPD_ORDER = ["0", "1-29", "30-59", "60-89", "90-119", "120+"]
STATUSES = ["current", "delinquent", "restructured", "written_off", "closed"]
PRODUCTS = ["credit_card", "personal_loan", "mortgage", "auto_loan", "overdraft", "bnpl"]
CLOSE_REASONS = ["paid_off", "written_off", "closed_by_customer", "refinanced"]
TERMINAL = ["written_off", "closed"]

REQUIRED = {
    "account": ["account_id", "subject_id", "lender_id", "product_type", "open_date", "interest_rate"],
    "account_month": ["account_id", "as_of_month", "balance", "amount_due", "amount_paid",
                      "dpd_bucket", "months_in_arrears", "status"],
}

SCALE_LIMIT = 20  # x median before a value is treated as a scale error


# --- load -------------------------------------------------------------------

def load_table(folder, table):
    files = sorted(glob.glob(os.path.join(folder, table, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {os.path.join(folder, table)}")
    # Read as text so bad numbers and dates can be reported rather than silently coerced
    parts = [pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[""]) for f in files]
    return pd.concat(parts, ignore_index=True)


def load(folder):
    tables = {t: load_table(folder, t) for t in ["subject", "account", "account_month"]}
    for name, df in tables.items():
        print(f"  {name:14s} {len(df):>10,} rows")
    return tables


def to_number(col):
    return pd.to_numeric(col, errors="coerce")


def to_date(col):
    return pd.to_datetime(col, format="%Y-%m-%d", errors="coerce")


# --- validate ---------------------------------------------------------------

def month_key(df):
    return df["account_id"].fillna("") + "|" + df["as_of_month"].fillna("")


def flag(mask, df, table, check, key):
    return pd.DataFrame({"table": table, "row_key": key(df[mask]), "check": check})


def check_subjects(subj):
    key = lambda d: d["subject_id"].fillna("")
    return pd.concat([
        flag(subj["subject_id"].duplicated(keep=False), subj, "subject", "duplicate subject_id", key),
        flag(subj.isna().any(axis=1), subj, "subject", "missing field", key),
    ])


def check_accounts(acc, months):
    key = lambda d: d["account_id"].fillna("")
    out = [flag(acc["account_id"].duplicated(keep=False), acc, "account", "duplicate account_id", key)]

    for col in REQUIRED["account"]:
        out.append(flag(acc[col].isna(), acc, "account", f"missing {col}", key))

    for col, allowed in [("product_type", PRODUCTS), ("close_reason", CLOSE_REASONS)]:
        bad = acc[col].notna() & ~acc[col].isin(allowed)
        out.append(flag(bad, acc, "account", f"invalid {col}", key))

    open_date = to_date(acc["open_date"])
    window_end = to_date(months["as_of_month"]).max() + pd.offsets.MonthEnd(0)
    out.append(flag(acc["open_date"].notna() & open_date.isna(), acc, "account", "open_date not a valid date", key))
    out.append(flag(open_date > window_end, acc, "account", "open_date in the future", key))

    rate = to_number(acc["interest_rate"])
    out.append(flag(acc["interest_rate"].notna() & rate.isna(), acc, "account", "interest_rate not a number", key))

    for col in ["credit_limit", "principal"]:
        amount = to_number(acc[col])
        median = amount.groupby(acc["product_type"]).transform("median")
        out.append(flag(amount > SCALE_LIMIT * median, acc, "account", f"{col} far above product median", key))

    final = months.assign(m=to_date(months["as_of_month"])).sort_values("m").groupby("account_id")["status"].last()
    final = acc["account_id"].map(final)
    mismatch = ((final == "written_off") & (acc["close_reason"] != "written_off")) | \
               ((acc["close_reason"] == "written_off") & final.notna() & (final != "written_off"))
    out.append(flag(mismatch, acc, "account", "close_reason contradicts final status", key))

    return pd.concat(out)


def check_months(months, acc):
    key = month_key
    t = "account_month"
    out = [
        flag(months.duplicated(["account_id", "as_of_month"], keep=False), months, t, "duplicate account-month", key),
        flag(~months["account_id"].isin(acc["account_id"]), months, t, "orphan row", key),
        flag(months["dpd_bucket"].notna() & ~months["dpd_bucket"].isin(DPD_ORDER), months, t, "invalid dpd_bucket", key),
        flag(months["status"].notna() & ~months["status"].isin(STATUSES), months, t, "invalid status", key),
    ]
    for col in REQUIRED[t]:
        out.append(flag(months[col].isna(), months, t, f"missing {col}", key))
    for col in ["balance", "amount_due", "amount_paid"]:
        value = to_number(months[col])
        out.append(flag(months[col].notna() & value.isna(), months, t, f"{col} not a number", key))
        out.append(flag(value < 0, months, t, f"negative {col}", key))

    balance = to_number(months["balance"])
    median = balance.groupby(months["account_id"]).transform("median")
    out.append(flag((balance > SCALE_LIMIT * median) & (median > 0), months, t,
                    "balance far above account median", key))

    h = months.drop_duplicates(["account_id", "as_of_month"]).copy()
    h["month"] = to_date(h["as_of_month"])
    h["bucket"] = h["dpd_bucket"].map({b: i for i, b in enumerate(DPD_ORDER)})
    h = h.sort_values(["account_id", "month"])
    prev = h.groupby("account_id")[["bucket", "month"]].shift(1)
    # only compare against the immediately preceding month, not across a gap
    roll = (h["bucket"] - prev["bucket"]).where(prev["month"] == h["month"] - pd.DateOffset(months=1))

    out.append(flag(roll > 1, h, t, "DPD skipped a bucket", key))
    out.append(flag((roll == 1) & (to_number(h["amount_paid"]) > 0), h, t, "payment while rolling worse", key))

    closed_at = h["month"].where(h["status"].isin(TERMINAL)).groupby(h["account_id"]).transform("min")
    out.append(flag(h["month"] > closed_at, h, t, "activity after closure", key))

    # gaps: months numbered year*12 + month should step by exactly 1
    g = h.dropna(subset=["month"]).copy()
    g["n"] = g["month"].dt.year * 12 + g["month"].dt.month
    g["step"] = g.groupby("account_id")["n"].diff()
    gaps = []
    for acc_id, n, step in g.loc[g["step"] > 1, ["account_id", "n", "step"]].itertuples(index=False):
        for missing in range(int(n - step) + 1, int(n)):
            year, month = divmod(missing - 1, 12)
            gaps.append((t, f"{acc_id}|{year:04d}-{month + 1:02d}-01", "month missing from history"))
    out.append(pd.DataFrame(gaps, columns=["table", "row_key", "check"]))

    return pd.concat(out)


def validate(tables):
    acc, months = tables["account"], tables["account_month"]
    findings = pd.concat([
        check_subjects(tables["subject"]),
        check_accounts(acc, months),
        check_months(months, acc),
    ], ignore_index=True).drop_duplicates()

    summary = (findings.groupby(["table", "check"]).size().rename("rows flagged")
               .reset_index().sort_values("rows flagged", ascending=False))
    print(f"  {len(findings):,} findings across {findings['row_key'].nunique():,} rows")
    return findings, summary


# --- clean ------------------------------------------------------------------

def clean(tables):
    """Fix formatting, drop rows that fail checks, log what was removed.
    Values are never repaired by guessing (no dividing suspect balances by 100)."""
    acc = tables["account"].copy()
    months = tables["account_month"].copy()
    log = []

    def keep(df, mask, step, table):
        log.append({"step": step, "table": table, "rows removed": int((~mask).sum())})
        return df[mask]

    for df, cols in [(acc, ["product_type", "close_reason"]), (months, ["status", "dpd_bucket"])]:
        for col in cols:
            df[col] = df[col].str.strip().str.lower()

    # some open dates arrive as DD/MM/YYYY; unambiguous, so convert rather than drop
    uk = pd.to_datetime(acc["open_date"], format="%d/%m/%Y", errors="coerce")
    acc["open_date"] = to_date(acc["open_date"]).fillna(uk)
    acc["close_date"] = to_date(acc["close_date"])
    for col in ["credit_limit", "principal", "interest_rate", "tenor_months"]:
        acc[col] = to_number(acc[col])
    for col in ["balance", "amount_due", "amount_paid", "months_in_arrears"]:
        months[col] = to_number(months[col])
    months["as_of_month"] = to_date(months["as_of_month"])
    window_end = months["as_of_month"].max() + pd.offsets.MonthEnd(0)

    acc = acc.drop_duplicates()
    acc = keep(acc, ~acc["account_id"].duplicated(), "duplicate account_id (kept first)", "account")
    acc = keep(acc, acc[REQUIRED["account"]].notna().all(axis=1), "missing required field", "account")
    acc = keep(acc, acc["product_type"].isin(PRODUCTS), "invalid product_type", "account")
    acc = keep(acc, acc["open_date"] <= window_end, "open_date in the future", "account")
    for col in ["credit_limit", "principal"]:
        median = acc[col].groupby(acc["product_type"]).transform("median")
        acc = keep(acc, ~(acc[col] > SCALE_LIMIT * median), f"{col} scale error", "account")

    t = "account_month"
    months = keep(months, ~months.duplicated(["account_id", "as_of_month"]), "duplicate account-month (kept first)", t)
    months = keep(months, months["account_id"].isin(acc["account_id"]), "orphan or excluded account", t)
    months = keep(months, months[REQUIRED[t]].notna().all(axis=1), "missing required field", t)
    months = keep(months, months["dpd_bucket"].isin(DPD_ORDER) & months["status"].isin(STATUSES), "invalid code", t)
    months = keep(months, (months[["balance", "amount_due", "amount_paid"]] >= 0).all(axis=1), "negative amount", t)
    median = months.groupby("account_id")["balance"].transform("median")
    months = keep(months, ~((months["balance"] > SCALE_LIMIT * median) & (median > 0)), "balance scale error", t)

    months = months.sort_values(["account_id", "as_of_month"])
    bucket = months["dpd_bucket"].map({b: i for i, b in enumerate(DPD_ORDER)})
    prev = months.groupby("account_id")["as_of_month"].shift(1)
    jumped = (prev == months["as_of_month"] - pd.DateOffset(months=1)) & \
             (bucket - bucket.groupby(months["account_id"]).shift(1) > 1)
    months = keep(months, ~jumped, "DPD skipped a bucket", t)
    closed_at = months["as_of_month"].where(months["status"].isin(TERMINAL)).groupby(months["account_id"]).transform("min")
    months = keep(months, ~(months["as_of_month"] > closed_at), "activity after closure", t)

    acc = acc.merge(tables["subject"][["subject_id", "risk_grade"]], on="subject_id", how="left")

    log = pd.DataFrame(log)
    log = log[log["rows removed"] > 0].reset_index(drop=True)
    print(f"  {len(acc):,} accounts, {len(months):,} account-months kept")
    return {"account": acc, "account_month": months, "cleaning_log": log}


# --- summarise --------------------------------------------------------------

def with_account_fields(months, acc):
    cols = ["account_id", "product_type", "risk_grade", "open_date", "interest_rate", "secured"]
    return months.merge(acc[cols], on="account_id", how="inner")


def live_rows(m):
    return m[~m["status"].isin(TERMINAL)].copy()


def portfolio_summary(m):
    live = live_rows(m[m["as_of_month"] == m["as_of_month"].max()])
    live["dpd30"] = live["dpd_bucket"].isin(DPD_ORDER[2:]) * live["balance"]
    live["dpd90"] = live["dpd_bucket"].isin(DPD_ORDER[4:]) * live["balance"]
    live["rate_x_bal"] = live["interest_rate"] * live["balance"]

    g = live.groupby("product_type").agg(
        accounts=("account_id", "count"), balance=("balance", "sum"),
        dpd30=("dpd30", "sum"), dpd90=("dpd90", "sum"), rate_x_bal=("rate_x_bal", "sum"))
    g.loc["Total"] = g.sum()

    return pd.DataFrame({
        "Accounts": g["accounts"].astype(int),
        "Balance": g["balance"].round(0).astype("int64"),
        "% of balance": (g["balance"] / g.loc["Total", "balance"] * 100).round(1),
        "30+ DPD %": (g["dpd30"] / g["balance"] * 100).round(2),
        "90+ DPD %": (g["dpd90"] / g["balance"] * 100).round(2),
        "WA rate %": (g["rate_x_bal"] / g["balance"] * 100).round(2),
    }).rename_axis("product")


def writeoff_rates(m):
    """Last 12 months: amount written off / average live balance."""
    last12 = m[m["as_of_month"] > m["as_of_month"].max() - pd.DateOffset(months=12)]
    written = last12[last12["status"] == "written_off"].groupby("product_type")["balance"].sum()
    avg_bal = (live_rows(last12).groupby(["product_type", "as_of_month"])["balance"].sum()
               .groupby("product_type").mean())
    out = pd.DataFrame({"Written off (12m)": written, "Avg balance (12m)": avg_bal}).fillna(0).round(0)
    out["Write-off rate %"] = (out["Written off (12m)"] / out["Avg balance (12m)"] * 100).round(2)
    return out.rename_axis("product")


def risk_by_grade(m, acc):
    bad = m["dpd_bucket"].isin(DPD_ORDER[4:]) | (m["status"] == "written_off")
    ever_bad = bad.groupby(m["account_id"]).any()
    a = acc.assign(bad=acc["account_id"].map(ever_bad).fillna(False).astype(bool))
    out = a.groupby("risk_grade").agg(accounts=("account_id", "count"), bad=("bad", "mean"))
    out["Ever 90+ or written off %"] = (out.pop("bad") * 100).round(2)
    return out


def arrears_trend(m):
    live = live_rows(m)
    live["dpd30"] = live["dpd_bucket"].isin(DPD_ORDER[2:]) * live["balance"]
    g = live.groupby(["as_of_month", "product_type"])[["dpd30", "balance"]].sum()
    return (g["dpd30"] / g["balance"] * 100).unstack("product_type").round(2)


def roll_rates(m):
    h = m.sort_values(["account_id", "as_of_month"])[["account_id", "as_of_month", "dpd_bucket"]].copy()
    nxt = h.groupby("account_id")[["dpd_bucket", "as_of_month"]].shift(-1)
    h = h[nxt["as_of_month"] == h["as_of_month"] + pd.DateOffset(months=1)]
    matrix = pd.crosstab(h["dpd_bucket"], nxt.loc[h.index, "dpd_bucket"], normalize="index") * 100
    return matrix.reindex(index=DPD_ORDER, columns=DPD_ORDER).fillna(0).round(2)


def vintage_curves(m, product="auto_loan"):
    """Cumulative % ever 90+ DPD or written off, by months on book and half-year of
    origination. Accounts opened before the window are excluded so every vintage
    is observed from month 0."""
    p = m[(m["product_type"] == product) & (m["open_date"] >= m["as_of_month"].min())].copy()
    p["mob"] = ((p["as_of_month"].dt.year - p["open_date"].dt.year) * 12
                + p["as_of_month"].dt.month - p["open_date"].dt.month)
    p["vintage"] = p["open_date"].dt.year.astype(str) + "-H" + ((p["open_date"].dt.month - 1) // 6 + 1).astype(str)
    p["bad"] = p["dpd_bucket"].isin(DPD_ORDER[4:]) | (p["status"] == "written_off")

    accounts = p.groupby("account_id").agg(vintage=("vintage", "first"))
    accounts["first_bad"] = p[p["bad"]].groupby("account_id")["mob"].min()
    seen = p.groupby("vintage")["mob"].max()

    curves = {}
    for vintage, grp in accounts.groupby("vintage"):
        curves[vintage] = [(grp["first_bad"] <= k).mean() * 100 if k <= seen[vintage] else np.nan
                           for k in range(int(p["mob"].max()) + 1)]
    curves = pd.DataFrame(curves).round(2).rename_axis("months on book")
    return curves, accounts["vintage"].value_counts().sort_index()


def summarise(cleaned, product="auto_loan"):
    m = with_account_fields(cleaned["account_month"], cleaned["account"])
    curves, sizes = vintage_curves(m, product)
    return {
        "portfolio_summary": portfolio_summary(m),
        "writeoff_rates": writeoff_rates(m),
        "risk_by_grade": risk_by_grade(m, cleaned["account"]),
        "arrears_trend": arrears_trend(m),
        "roll_rates": roll_rates(m),
        f"vintage_{product}": curves,
        "vintage_sizes": sizes.rename("accounts").to_frame(),
    }


# --- report -----------------------------------------------------------------

BLUE, RED = "#2f5d8a", "#c0392b"


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, folder, name):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def make_charts(summary, exceptions, folder, product="auto_loan"):
    paths = []

    fig, ax = plt.subplots(figsize=(8, 5))
    top = exceptions.head(15).iloc[::-1]
    ax.barh(top["check"], top["rows flagged"], color=BLUE)
    _style(ax, "Data-quality findings by check (top 15)", "Rows flagged")
    paths.append(_save(fig, folder, "1_findings.png"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    share = summary["portfolio_summary"]["% of balance"].drop("Total")
    trend = summary["arrears_trend"]
    trend[[p for p in trend.columns if share.get(p, 0) >= 1]].plot(ax=ax)
    _style(ax, "30+ DPD, % of balance (products over 1% of the book)", ylabel="% of balance")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    paths.append(_save(fig, folder, "2_arrears_trend.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    r = summary["risk_by_grade"]["Ever 90+ or written off %"]
    ax.bar(r.index.astype(str), r.values, color=BLUE)
    _style(ax, "Bad rate by risk grade", "Risk grade", "Ever 90+ DPD or written off (%)")
    paths.append(_save(fig, folder, "3_risk_by_grade.png"))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    rr = summary["roll_rates"]
    ax.imshow(rr.values, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(rr.columns)), rr.columns)
    ax.set_yticks(range(len(rr.index)), rr.index)
    for i, j in np.ndindex(rr.shape):
        v = rr.iat[i, j]
        if v >= 0.05:
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 60 else "black")
    _style(ax, "Roll rates (% of accounts, month to month)", "Next month", "This month")
    paths.append(_save(fig, folder, "4_roll_rates.png"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    summary[f"vintage_{product}"].plot(ax=ax)
    _style(ax, f"{product} vintages: cumulative % 90+ DPD or written off", "Months on book", "%")
    ax.legend(title="Opened", frameon=False, fontsize=8)
    paths.append(_save(fig, folder, "5_vintage_curves.png"))

    return paths


def compare_scenarios(base, stressed, product="auto_loan", out_folder="output"):
    both = pd.DataFrame({"Baseline": base["arrears_trend"][product],
                         "Stressed": stressed["arrears_trend"][product]})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    both.plot(ax=ax, color=[BLUE, RED])
    _style(ax, f"{product}: 30+ DPD % of balance, baseline vs stressed", ylabel="% of balance")
    ax.legend(frameon=False)
    return both, _save(fig, os.path.join(out_folder, "charts"), "6_baseline_vs_stressed.png")


def run(data_folder, out_folder="output", product="auto_loan"):
    os.makedirs(out_folder, exist_ok=True)

    print("load")
    tables = load(data_folder)
    print("validate")
    findings, exceptions = validate(tables)
    print("clean")
    cleaned = clean(tables)
    print("summarise")
    summary = summarise(cleaned, product)

    findings.to_csv(os.path.join(out_folder, "findings.csv"), index=False)
    with pd.ExcelWriter(os.path.join(out_folder, "report.xlsx")) as xl:
        sheets = {"exceptions": exceptions.set_index(["table", "check"]),
                  "cleaning_log": cleaned["cleaning_log"].set_index("step"), **summary}
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name[:31])
    charts = make_charts(summary, exceptions, os.path.join(out_folder, "charts"), product)
    print(f"saved to {out_folder}/")

    return {"findings": findings, "exceptions": exceptions, "cleaned": cleaned,
            "summary": summary, "charts": charts}
