"""Render the README tables directly from ``results/*.csv``.

Every number quoted in the README is generated here, so the README cannot drift from the
artifacts.  Marker blocks in README.md::

    <!-- TABLE:mqar -->
    ...generated...
    <!-- /TABLE:mqar -->

    python tools/render_tables.py            # rewrite README.md
    python tools/render_tables.py --check    # exit 1 if README is stale (used by CI)

Tables sourced from ``results/notebook/*.csv`` come from ``src/notebook.py``; the rest
come from the ``experiments/`` drivers.  Both are labelled in the README.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

README = ROOT / "README.md"


def _load(name: str) -> pd.DataFrame:
    p = RESULTS / name
    if not p.exists():
        raise SystemExit(f"missing {p} — run the experiments first")
    return pd.read_csv(p)


def _f(v, nd: int = 3) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x != x:
        return "**NaN**"
    return f"{x:.{nd}f}"


def _pm(r, col: str, nd: int = 3) -> str:
    return f"{_f(getattr(r, col + '_mean'), nd)} ± {_f(getattr(r, col + '_std'), nd)}"


def _acc_cols(df: pd.DataFrame) -> list[int]:
    return sorted(int(c[len("acc_N"):-len("_mean")]) for c in df.columns
                  if c.startswith("acc_N") and c.endswith("_mean"))


def table_mqar() -> str:
    df = _load("mqar.csv")
    ns = _acc_cols(df)
    head = "| variant | final loss | " + " | ".join(f"N={n}" for n in ns) + " |"
    sep = "|---|---|" + "---|" * len(ns)
    rows = []
    for r in df.itertuples():
        cells = [f"`{r.variant}`", _pm(r, "final_loss")]
        for n in ns:
            cells.append(f"**{_f(getattr(r, f'acc_N{n}_mean'))}** ± "
                         f"{_f(getattr(r, f'acc_N{n}_std'))}")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows, "",
                      f"Mean ± std over **{int(df.n_seeds.max())} seeds**; random baseline "
                      f"= 1/64 = 0.016."])


def table_hybrid_ratio() -> str:
    df = _load("hybrid_ratio.csv").sort_values("n_attn")
    ns = _acc_cols(df)
    head = ("| attention layers | linear layers | final loss | "
            + " | ".join(f"N={n}" for n in ns) + " |")
    sep = "|---|---|---|" + "---|" * len(ns)
    rows = []
    for r in df.itertuples():
        cells = [f"`{r.config}`", f"{r.n_attn}/{int(r.layers)}",
                 f"{int(r.layers) - r.n_attn}/{int(r.layers)}", _pm(r, "final_loss")]
        for n in ns:
            cells.append(f"**{_f(getattr(r, f'acc_N{n}_mean'))}** ± "
                         f"{_f(getattr(r, f'acc_N{n}_std'))}")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows, "",
                      f"Mean ± std over **{int(df.n_seeds.max())} seeds**."])


def table_char_lm() -> str:
    df = _load("char_lm.csv")
    head = "| variant | valid loss | perplexity | 4-gram diversity |"
    sep = "|---|---|---|---|"
    rows = [f"| `{r.variant}` | **{_f(r.val_loss_mean, 4)}** ± {_f(r.val_loss_std, 4)} | "
            f"{_f(r.perplexity_mean, 2)} | {_f(r.ngram4_diversity_mean)} |"
            for r in df.itertuples()]
    return "\n".join([head, sep, *rows, "",
                      f"Mean ± std over **{int(df.n_seeds.max())} seeds**. Validation is a "
                      "held-out 10 % split of tiny shakespeare, not a random training batch."])


def table_decode() -> str:
    df = _load("decode_bench.csv")
    df = df[~df["oom"].fillna(False).astype(bool)]
    order = list(dict.fromkeys(df["variant"]))
    out = []
    for b in sorted(df["batch"].unique()):
        sub = df[df.batch == b]
        seqs = sorted(sub.seq_len.unique())
        out += [f"**batch = {b}** — tok/s", "",
                "| model | " + " | ".join(f"{s // 1024}K" for s in seqs) + " |",
                "|---|" + "---|" * len(seqs)]
        for v in order:
            cells = []
            for s in seqs:
                row = sub[(sub.variant == v) & (sub.seq_len == s)]
                cells.append("—" if row.empty else f"**{row.tok_per_s.iloc[0]:,.0f}**")
            out.append(f"| `{v}` | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out).rstrip()


def table_decode_state() -> str:
    df = _load("decode_bench.csv")
    df = df[(df.batch == df.batch.min()) & (~df["oom"].fillna(False).astype(bool))]
    last = df.seq_len.max()
    head = f"| model | KV cache @ {last // 1024}K | fixed state @ {last // 1024}K |"
    sep = "|---|---|---|"
    rows = []
    for r in df[df.seq_len == last].itertuples():
        kv = f"{r.kv_MB:,.0f} MB" if r.kv_MB else "0 (no attention layer)"
        rows.append(f"| `{r.variant}` | {kv} | **{r.state_KB:,.1f} KB** |")
    return "\n".join([head, sep, *rows, "", "batch = 1."])


def table_featmap() -> str:
    df = _load("notebook/featmap_kernel.csv")
    head = "| feature map | M | κ(self) | κ(off-diagonal) |"
    sep = "|---|---|---|---|"
    rows = [f"| `{r.featmap}` | {r.M:,} | {_f(r.kernel_self)} | "
            f"**{_f(r.kernel_off_mean, 4)}** |" for r in df.itertuples()]
    return "\n".join([head, sep, *rows])


def table_featmap_fidelity() -> str:
    df = _load("notebook/featmap_kernel.csv")
    ns = sorted(int(c[len("fid_N"):]) for c in df.columns if c.startswith("fid_N"))
    head = "| feature map | M | " + " | ".join(f"N={n}" for n in ns) + " |"
    sep = "|---|---|" + "---|" * len(ns)
    rows = []
    for r in df.itertuples():
        rows.append(f"| `{r.featmap}` | {r.M:,} | "
                    + " | ".join(_f(getattr(r, f"fid_N{n}")) for n in ns) + " |")
    return "\n".join([head, sep, *rows])


def table_capacity() -> str:
    df = _load("notebook/capacity_summary.csv").sort_values(["kind", "M"])
    head = "| state kind | dimension `M` | state elements | capacity @ 0.9 |"
    sep = "|---|---|---|---|"
    rows = [f"| `{r.kind}` | {r.M:,} | {r.state_elems:,} | **{r.capacity}** |"
            for r in df.itertuples()]
    return "\n".join([head, sep, *rows])


def table_delta_stability() -> str:
    df = _load("mqar_l4.csv")
    df = df[df.variant.str.startswith("delta")]
    head = "| variant | final loss | N=8 |"
    sep = "|---|---|---|"
    rows = [f"| `{r.variant}` | {_pm(r, 'final_loss')} | **{_f(r.acc_N8_mean)}** ± "
            f"{_f(r.acc_N8_std)} |" for r in df.itertuples()]
    return "\n".join([head, sep, *rows, "",
                      f"{int(df.n_seeds.max())} seeds. `delta_rule_nonorm` diverges to NaN "
                      "without key normalisation."])


TABLES = {
    "mqar": table_mqar,
    "hybrid_ratio": table_hybrid_ratio,
    "char_lm": table_char_lm,
    "decode": table_decode,
    "decode_state": table_decode_state,
    "featmap": table_featmap,
    "featmap_fidelity": table_featmap_fidelity,
    "capacity": table_capacity,
    "delta_stability": table_delta_stability,
}


def render(readme: str) -> str:
    for name, fn in TABLES.items():
        begin, end = f"<!-- TABLE:{name} -->", f"<!-- /TABLE:{name} -->"
        i, j = readme.find(begin), readme.find(end)
        if i < 0 or j < 0:
            raise SystemExit(f"missing markers for table '{name}' in README.md")
        readme = readme[:i + len(begin)] + "\n" + fn() + "\n" + readme[j:]
    return readme


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    current = README.read_text(encoding="utf-8")
    updated = render(current)
    if args.check:
        if updated != current:
            print("README.md is out of date with results/*.csv — "
                  "run: python tools/render_tables.py", file=sys.stderr)
            sys.exit(1)
        print("README tables are in sync with results/*.csv")
        return
    README.write_text(updated, encoding="utf-8", newline="\n")
    print(f"rewrote {README}")


if __name__ == "__main__":
    main()
