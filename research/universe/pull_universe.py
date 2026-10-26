"""Databento pulls for the cross-commodity spread study (energy, ags, softs, FX, ICE venues).

Run from the repo folder with PYTHONPATH=src and DATABENTO_API_KEY set (cmd shown):

    set PYTHONPATH=src
    python research\\universe\\pull_universe.py --group all --cost-only          # free: estimate only
    python research\\universe\\pull_universe.py --group ice_us ice_eu endex --discover   # free: which ICE roots resolve
    python research\\universe\\pull_universe.py --group energy --only CL --schemas ohlcv-1d --start 2025-01-01

SAFETY (after the 2026-10-26 overage): the pull only starts if
  1. data/databento/budget.json says how much may be spent ({"remaining_usd": 50}),
  2. the estimate printed in THIS run is within that budget and within --max-cost,
  3. you type the exact estimated dollar figure back at the prompt,
  4. no other pull is running (lock file).
Every request is written to data/databento/spend_ledger.csv, and the run stops once it has
spent 80% of the budget. There is no auto-confirm flag.

Each group can run in its own window at the same time (they write different files).
Every (root, schema, year) lands in its own parquet file and a file that already exists is
never pulled again, so it is safe to stop with Ctrl+C and rerun.

What is pulled for every root
  ohlcv-1d    daily bars, all contracts under the parent symbol
  statistics  settlement price (stat 3) and open interest (stat 9), last message per
              contract per trading day

Where it goes
  CME (GLBX.MDP3)  data/databento/<ROOT>_ohlcv-1d_<year>.parquet and <ROOT>_settle_<year>.parquet
                   (same layout as the GC/SI and metals cache)
  ICE datasets     data/databento/<DATASET>/<NAME>_ohlcv-1d_<year>.parquet and _settle_
  The ICE parent codes are not documented in one place, so for each ICE product the script
  tries a few candidate codes with the free symbology endpoint and uses the first that
  resolves. --discover prints that mapping without pulling anything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from gcsi import data

SETTLE, OPEN_INTEREST = 3, 9

# group -> (dataset, earliest date, {file name: [candidate parent codes]})
GROUPS = {
    "energy": ("GLBX.MDP3", "2010-06-06", {
        "CL": ["CL"], "RB": ["RB"], "HO": ["HO"],          # crude, RBOB gasoline, ULSD
        "NG": ["NG"], "HH": ["HH"],                         # Henry Hub physical vs last-day financial
        "BZ": ["BZ"],                                       # NYMEX Brent last-day financial
        "CU": ["CU"],                                       # Chicago ethanol (Platts)
    }),
    "ags": ("GLBX.MDP3", "2010-06-06", {
        "ZS": ["ZS"], "ZM": ["ZM"], "ZL": ["ZL"],          # soybeans, meal, oil
        "ZC": ["ZC"], "ZW": ["ZW"], "KE": ["KE"],          # corn, Chicago wheat, KC wheat
    }),
    "fx": ("GLBX.MDP3", "2010-06-06", {
        "6B": ["6B"], "6E": ["6E"], "6L": ["6L"], "6C": ["6C"],   # GBP, EUR, BRL, CAD
    }),
    "ice_us": ("IFUS.IMPACT", "2018-12-23", {
        "CC": ["CC"], "KC": ["KC"], "SB": ["SB"], "CT": ["CT"], "RS": ["RS"],  # cocoa coffee sugar cotton canola
    }),
    "ice_eu": ("IFEU.IMPACT", "2018-12-23", {
        "BRN": ["BRN", "B"],            # ICE Brent
        "GO": ["G", "GAS", "GO"],       # ICE low sulphur gasoil
        "WTI": ["T", "WBS", "WTI"],     # ICE WTI
        "LCC": ["C", "LCC"],            # London cocoa
        "RC": ["RC"],                   # robusta coffee
        "WS": ["W", "WS"],              # white sugar
    }),
    "endex": ("NDEX.IMPACT", "2018-12-23", {
        "TTF": ["TFM", "TTF", "TFO"],   # Dutch TTF gas
    }),
}
SCHEMAS = ("ohlcv-1d", "statistics")


def folder(dataset: str) -> Path:
    base = data.DEFAULT_CACHE_DIR
    return base if dataset == data.DATASET else base / dataset.split(".")[0]


def path_for(dataset: str, name: str, schema: str, year: int) -> Path:
    tag = "settle" if schema == "statistics" else schema
    return folder(dataset) / f"{name}_{tag}_{year}.parquet"


def year_chunks(start: pd.Timestamp, end: pd.Timestamp):
    for y in range(start.year, end.year + 1):
        s = max(pd.Timestamp(year=y, month=1, day=1, tz="UTC"), start)
        e = min(pd.Timestamp(year=y + 1, month=1, day=1, tz="UTC"), end)
        if s < e:
            yield y, s, e


def dataset_end(client, dataset: str, requested: pd.Timestamp) -> pd.Timestamp:
    try:
        return min(requested, data._utc(client.metadata.get_dataset_range(dataset=dataset)["end"]))
    except Exception:
        return requested


def resolve_code(client, dataset: str, candidates: list[str], day: str) -> str | None:
    """First candidate parent code that resolves on `dataset` (free symbology call)."""
    for code in candidates:
        try:
            r = client.symbology.resolve(dataset=dataset, symbols=[f"{code}.FUT"], stype_in="parent",
                                         stype_out="instrument_id", start_date=day)
        except Exception:
            continue
        if r.get("result") and f"{code}.FUT" not in (r.get("not_found") or []):
            return code
    return None


def stat_code(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        s = str(v).upper()
        return SETTLE if "SETTLEMENT_PRICE" in s else OPEN_INTEREST if "OPEN_INTEREST" in s else -1


def pull(client, dataset: str, code: str, schema: str, s, e) -> pd.DataFrame:
    store = client.timeseries.get_range(dataset=dataset, symbols=[f"{code}.FUT"], stype_in="parent",
                                        schema=schema, start=s.strftime("%Y-%m-%d"),
                                        end=e.strftime("%Y-%m-%d"))
    df = store.to_df().reset_index()
    if schema == "ohlcv-1d":
        cols = ["ts_event", "instrument_id", "symbol", "open", "high", "low", "close", "volume"]
        return df[[c for c in cols if c in df.columns]] if not df.empty else pd.DataFrame(columns=cols)
    cols = ["trade_date", "instrument_id", "symbol", "stat", "value"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["stat"] = df["stat_type"].map(stat_code)
    df = df[df["stat"].isin([SETTLE, OPEN_INTEREST])].copy()
    df["value"] = df["price"].where(df["stat"] == SETTLE, df["quantity"])
    ref = "ts_ref" if "ts_ref" in df.columns else "ts_event"
    df["trade_date"] = pd.to_datetime(df[ref], utc=True).dt.normalize()
    df = df.sort_values("ts_recv").drop_duplicates(["trade_date", "instrument_id", "stat"], keep="last")
    return df[cols].reset_index(drop=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--group", nargs="+", default=["all"], help=f"any of {list(GROUPS)} or all")
    p.add_argument("--only", nargs="+", help="restrict to these file names, e.g. CL RB HO")
    p.add_argument("--schemas", nargs="+", default=list(SCHEMAS))
    p.add_argument("--start", help="override the earliest date for every group")
    p.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p.add_argument("--discover", action="store_true", help="only print which parent codes resolve")
    p.add_argument("--cost-only", action="store_true")
    p.add_argument("--yes", action="store_true", help="ignored: confirmation is always typed")
    p.add_argument("--max-cost", type=float, default=0.0)
    a = p.parse_args(argv)

    groups = list(GROUPS) if "all" in a.group else a.group
    client = data.get_client()
    todo, total = [], 0.0
    for g in groups:
        dataset, first, products = GROUPS[g]
        start = data._utc(a.start or first)
        end = dataset_end(client, dataset, data._utc(a.end))
        probe = (end - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        for name, cands in products.items():
            if a.only and name not in a.only:
                continue
            code = resolve_code(client, dataset, cands, probe)
            if code is None:
                print(f"[{g}] {name}: none of {cands} resolved on {dataset}, skipped", flush=True)
                continue
            if a.discover:
                print(f"[{g}] {name}: uses {dataset} parent {code}.FUT", flush=True)
                continue
            for schema in a.schemas:
                for y, s, e in year_chunks(start, end):
                    path = path_for(dataset, name, schema, y)
                    meta = path.with_suffix(".json")
                    if path.exists() and (not meta.exists() or pd.Timestamp(json.loads(meta.read_text())["end"]) >= e):
                        continue
                    try:
                        c = float(client.metadata.get_cost(dataset=dataset, symbols=[f"{code}.FUT"],
                                                           stype_in="parent", schema=schema,
                                                           start=s.strftime("%Y-%m-%d"),
                                                           end=e.strftime("%Y-%m-%d")))
                    except Exception as exc:
                        print(f"[{g}] {name} {schema} {y}: cost query failed ({exc}), skipped", flush=True)
                        continue
                    total += c
                    todo.append((g, dataset, name, code, schema, y, s, e, path, c))
    if a.discover:
        return
    by = pd.DataFrame([(t[0], t[2], t[4], t[9]) for t in todo], columns=["group", "name", "schema", "usd"])
    if not by.empty:
        print(by.groupby(["group", "name", "schema"])["usd"].sum().round(2).unstack("schema").fillna(0).to_string())
    print(f"\n{len(todo)} uncached requests, estimated ${total:,.2f}", flush=True)
    if a.cost_only or not todo:
        return
    budget_file = data.DEFAULT_CACHE_DIR / "budget.json"
    budget = json.loads(budget_file.read_text())["remaining_usd"] if budget_file.exists() else 0.0
    if total > a.max_cost or total > budget:
        sys.exit(f"STOP: estimate ${total:,.2f} is above --max-cost ${a.max_cost:,.2f} or the budget "
                 f"${budget:,.2f} in {budget_file}. Narrow the pull (--only, --schemas, --start) and rerun.")
    typed = input(f"This will spend about ${total:,.2f}. Type {total:.2f} to proceed: ").strip()
    if typed != f"{total:.2f}":
        sys.exit("not confirmed, nothing pulled")
    lock = data.DEFAULT_CACHE_DIR / ".pull.lock"
    try:
        lock.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        sys.exit(f"another pull is running (or crashed): {lock}. Only one paid pull at a time.")
    ledger = data.DEFAULT_CACHE_DIR / "spend_ledger.csv"
    spent = 0.0
    try:
        _run(client, todo, ledger, budget, spent)
    finally:
        lock.unlink(missing_ok=True)


def _run(client, todo, ledger, budget, spent):
    for g, dataset, name, code, schema, y, s, e, path, c in todo:
        if path.exists():  # another window got there first
            continue
        t0 = time.time()
        for attempt in range(3):
            try:
                df = pull(client, dataset, code, schema, s, e)
                break
            except Exception as exc:  # network hiccup or an unsupported schema: retry, then move on
                print(f"  {name} {schema} {y} attempt {attempt + 1} failed: {exc}", flush=True)
                time.sleep(10)
        else:
            continue
        spent += c
        with open(ledger, "a") as fh:
            fh.write(f"{pd.Timestamp.now(tz='UTC').isoformat()},{dataset},{name},{schema},{y},{c:.4f},{spent:.4f}\n")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        path.with_suffix(".json").write_text(json.dumps({"end": e.isoformat(), "dataset": dataset,
                                                          "parent": f"{code}.FUT"}))
        print(f"[{g}] {name} {schema} {y}: {len(df):,} rows, ${c:.2f} ({time.time() - t0:.0f}s)", flush=True)
        if spent >= 0.8 * budget:
            print(f"STOP: spent ${spent:,.2f}, 80% of the ${budget:,.2f} budget. Update budget.json to continue.")
            return
    print("done")


if __name__ == "__main__":
    main()
