# -*- coding: utf-8 -*-
"""
Real-data WNV Early-Warning (City of Chicago + Meteostat) — NO SYNTHETIC FALLBACK

Requirements:
    pip install sodapy meteostat pandas numpy scikit-learn matplotlib

Outputs:
  - outputs/tables/*.csv
  - outputs/figs/*.png
  - wnv_ml_outputs_real.zip
"""

import os, json, zipfile
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sodapy import Socrata
from meteostat import Point, Daily

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve, confusion_matrix, classification_report
)
from sklearn.neighbors import NearestNeighbors

# --------------------------- CONFIG ---------------------------
SOCRATA_DOMAIN = "data.cityofchicago.org"
DATASET_ID = "jqe8-8r6s"  # West Nile Virus (WNV) Mosquito Test Results)

# <<< Your Socrata API Key ID (acts as SODA App Token) >>>
APP_TOKEN = "provide token here"
# NOTE: API Key Secret is not required for public reads via SODA and is not used here.

# To limit download volume initially, you can narrow the window; then widen later:
DATE_MIN = "2010-01-01"
DATE_MAX = date.today().isoformat()

OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figs")
TAB_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

RNG_SEED = 42
np.random.seed(RNG_SEED)

TARGET_COL = "wnv_positive"

# ------------------------- UTILITIES --------------------------
def require_cols(df: pd.DataFrame, cols, msg=""):
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"Missing columns {missing}. {msg}"

def week_start_from_ts(ts: pd.Timestamp) -> pd.Timestamp:
    return ts - pd.Timedelta(days=ts.weekday())

def add_lags(df, group_cols, time_col, cols, lags=(1,2,3)):
    df = df.sort_values(group_cols + [time_col]).copy()
    for c in cols:
        for L in lags:
            df[f"{c}_lag{L}"] = df.groupby(group_cols, sort=False)[c].shift(L)
    return df

def build_knn_graph(coords, k=5):
    k = min(max(2, k), max(2, len(coords)-1))
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    n = coords.shape[0]
    W = np.zeros((n,n), dtype=float)
    for i in range(n):
        for j in idx[i,1:]:
            W[i,j]=1.0; W[j,i]=1.0
    rs = W.sum(axis=1, keepdims=True); rs[rs==0]=1.0
    return W/rs

def laplacian_smooth_probs(p, W, eta=0.5):
    eps=1e-6
    logits = np.log(np.clip(p,eps,1-eps)/np.clip(1-p,eps,1-eps))
    n = W.shape[0]; I = np.eye(n); L = I - W
    u = np.linalg.solve(I + eta*L, logits)
    return 1.0/(1.0 + np.exp(-u))

# ---------------------- DATA DOWNLOAD -------------------------
def fetch_chicago_wnv(date_min: str, date_max: str) -> pd.DataFrame:
    """
    Pull WNV pool test table from Socrata (jqe8-8r6s) with explicit date filter.
    Returns tidy fields:
      ['pool_id','collection_date','trap_id','species','wnv_positive',
       'num_mosquitoes','lat','lon']
    """
    if not APP_TOKEN or APP_TOKEN.strip() == "":
        raise RuntimeError("APP_TOKEN missing. Provide your Socrata API Key ID.")

    client = Socrata(SOCRATA_DOMAIN, APP_TOKEN, timeout=60)
    # Filter by date using SoQL WHERE; handle both 'collection_date' and 'date_of_collection'
    where = (
        f"(collection_date between '{date_min}T00:00:00' and '{date_max}T23:59:59') "
        f"OR (date_of_collection between '{date_min}T00:00:00' and '{date_max}T23:59:59')"
    )
    limit = 50000
    offset = 0
    rows = []
    while True:
        batch = client.get(DATASET_ID, where=where, limit=limit, offset=offset)
        if not batch:
            break
        rows.extend(batch)
        offset += limit
        if len(batch) < limit:
            break
    client.close()

    df = pd.DataFrame.from_records(rows)
    if df.empty:
        raise RuntimeError(
            "Socrata returned 0 rows for the requested window.\n"
            "Tip: widen DATE_MIN/DATE_MAX or confirm dataset availability. "
            "Your APP_TOKEN looks fine syntactically."
        )

    # ---- normalize fields ----
    # collection date
    date_col = None
    for cand in ["collection_date","date_of_collection","sample_date","date","trap_collection_date"]:
        if cand in df.columns:
            date_col = cand; break
    if not date_col:
        raise RuntimeError("Could not find a collection date column in the dataset schema.")
    df["collection_date"] = pd.to_datetime(df[date_col], errors="coerce")

    # trap id/name
    trap_col = None
    for cand in ["trap","trap_id","trap_name"]:
        if cand in df.columns:
            trap_col = cand; break
    if not trap_col:
        raise RuntimeError("Could not locate a trap id/name column.")
    df["trap_id"] = df[trap_col].astype(str)

    # species
    species_col = None
    for cand in ["species","mosquito_species"]:
        if cand in df.columns:
            species_col = cand; break
    df["species"] = df[species_col].astype(str) if species_col else ""

    # result → binary
    result_col = None
    for cand in ["results","result","wnv_result","wnvpositive","wnv_positive","test_result"]:
        if cand in df.columns:
            result_col = cand; break
    if not result_col:
        raise RuntimeError("Could not find a WNV result column in the dataset.")

    res = df[result_col].astype(str).str.lower().str.strip()
    y = np.where(res.isin(["positive","true","1","yes","y"]), 1,
        np.where(res.isin(["negative","false","0","no","n"]), 0, np.nan))
    if np.isnan(y).mean() > 0.2:
        yn = pd.to_numeric(df[result_col], errors="coerce")
        y = np.where(~yn.isna(), yn, y)
    df[TARGET_COL] = pd.to_numeric(y, errors="coerce")

    # number of mosquitoes
    n_col = None
    for cand in ["number_of_mosquitoes","num_mosquitoes","mosquito_count","mosquitos_count","number_of_mosquitos"]:
        if cand in df.columns:
            n_col = cand; break
    df["num_mosquitoes"] = pd.to_numeric(df[n_col], errors="coerce") if n_col else np.nan

    # location → lat/lon
    lat, lon = None, None
    if "location" in df.columns:
        def to_lat(x):
            if isinstance(x, dict):
                coords = x.get("coordinates")
                if isinstance(coords, (list,tuple)) and len(coords)==2:
                    return coords[1]
            return np.nan
        def to_lon(x):
            if isinstance(x, dict):
                coords = x.get("coordinates")
                if isinstance(coords, (list,tuple)) and len(coords)==2:
                    return coords[0]
            return np.nan
        lat = df["location"].apply(to_lat)
        lon = df["location"].apply(to_lon)
    else:
        lat_cand = [c for c in df.columns if "lat" in c.lower()]
        lon_cand = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower() or "long" in c.lower()]
        lat = pd.to_numeric(df[lat_cand[0]], errors="coerce") if lat_cand else np.nan
        lon = pd.to_numeric(df[lon_cand[0]], errors="coerce") if lon_cand else np.nan

    out = pd.DataFrame({
        "pool_id": df.get(":id", df.index).astype(str),
        "collection_date": df["collection_date"],
        "trap_id": df["trap_id"],
        "species": df["species"],
        TARGET_COL: pd.to_numeric(df[TARGET_COL], errors="coerce"),
        "num_mosquitoes": df["num_mosquitoes"],
        "lat": lat,
        "lon": lon
    })
    out = out.dropna(subset=["collection_date","trap_id","lat","lon", TARGET_COL]).reset_index(drop=True)
    if out.empty:
        raise RuntimeError(
            "After cleaning, no rows remained with valid date/trap/lat/lon/result. "
            "Inspect raw columns and update the parsers."
        )
    return out

def build_trap_week_panel(pools: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse pools to trap-week. Target = 1 if any positive pool in that trap-week.
    """
    pools["week_start"] = pools["collection_date"].apply(week_start_from_ts)
    agg = pools.groupby(["trap_id","week_start"], as_index=False).agg({
        TARGET_COL: "max",
        "lat": "first",
        "lon": "first",
        "num_mosquitoes": "sum"
    })
    return agg

def fetch_weather_for_traps(panel: pd.DataFrame) -> pd.DataFrame:
    """
    For each trap, get Meteostat Daily from [min(week)-14d, max(week)+7d], aggregate weekly.
    Produces: ['trap_id','week_start','tavg_mean','tavg_max','prcp_sum','rh_mean','rh_min','sin_woy','cos_woy']
    """
    start = (panel["week_start"].min() - pd.Timedelta(days=14)).date()
    end   = (panel["week_start"].max() + pd.Timedelta(days=7)).date()

    frames = []
    for tid, g in panel.groupby("trap_id"):
        lat = float(g["lat"].iloc[0]); lon = float(g["lon"].iloc[0])
        p = Point(lat, lon)
        d = Daily(p, start, end).fetch().reset_index().rename(columns={"time":"date"})
        for c in ["tavg","tmax","prcp","rhum"]:
            if c not in d.columns: d[c] = np.nan
        d["week_start"] = d["date"].apply(week_start_from_ts)
        w = d.groupby("week_start", as_index=False).agg(
            tavg_mean=("tavg","mean"),
            tavg_max=("tmax","max"),
            prcp_sum=("prcp","sum"),
            rh_mean=("rhum","mean"),
        )
        w["trap_id"] = tid
        frames.append(w)

    if not frames:
        raise RuntimeError("No Meteostat frames assembled; check panel date range & internet connectivity.")

    wx = pd.concat(frames, ignore_index=True)
    wx["rh_min"] = wx["rh_mean"]  # proxy if daily RH min not available
    woy = wx["week_start"].dt.isocalendar().week.astype(int)
    wx["sin_woy"] = np.sin(2*np.pi*woy/52.0)
    wx["cos_woy"] = np.cos(2*np.pi*woy/52.0)
    return wx

# ---------------------- MODELING PIPELINE ----------------------
def main():
    print(f"[1/8] Downloading WNV pools (real data) {DATE_MIN} → {DATE_MAX} ...")
    df_pools = fetch_chicago_wnv(DATE_MIN, DATE_MAX)
    print(f"   pools: {len(df_pools):,} rows; span {df_pools['collection_date'].min().date()} → {df_pools['collection_date'].max().date()}")

    print("[2/8] Aggregating to trap-week target ...")
    panel = build_trap_week_panel(df_pools)
    prev = panel[TARGET_COL].mean()
    print(f"   trap-weeks: {len(panel):,}, prevalence={prev:.3f}")
    if panel.empty:
        raise RuntimeError("Trap-week panel is empty after aggregation; cannot proceed.")

    print("[3/8] Fetching Meteostat daily weather and aggregating weekly ...")
    wx = fetch_weather_for_traps(panel)
    print(f"   weather rows: {len(wx):,}")

    print("[4/8] Merging panel + weather & engineering lags ...")
    df = panel.merge(wx, on=["trap_id","week_start"], how="inner")
    feature_cols_base = ["tavg_mean","tavg_max","prcp_sum","rh_mean"]
    df = add_lags(df, ["trap_id"], "week_start", cols=feature_cols_base, lags=(1,2,3))
    df["rh_min"] = df["rh_min"].fillna(df["rh_mean"])
    df = df.dropna().reset_index(drop=True)

    # Save previews
    df.head(25).to_csv(os.path.join(TAB_DIR, "trap_week_head.csv"), index=False)
    df.to_csv(os.path.join(TAB_DIR, "trap_week_panel.csv"), index=False)

    # Prevalence over time
    plt.figure()
    pr = df.groupby(df["week_start"].dt.to_period("W").dt.start_time)[TARGET_COL].mean().reset_index()
    plt.plot(pr["week_start"], pr[TARGET_COL])
    plt.title("Trap-week WNV Positivity Rate Over Time (Real Data)")
    plt.xlabel("Week"); plt.ylabel("Positivity Rate")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "positivity_over_time.png"), dpi=150); plt.show()

    # ---------------- Splits ----------------
    df = df.sort_values(["week_start","trap_id"]).reset_index(drop=True)
    weeks = np.sort(df["week_start"].unique())
    if len(weeks) < 20:
        raise RuntimeError("Insufficient number of weeks for robust train/val/test splits.")
    train_cut = weeks[int(0.6*len(weeks))]
    val_cut   = weeks[int(0.8*len(weeks))]

    feature_cols = [
        "tavg_mean","tavg_max","prcp_sum","rh_mean","rh_min",
        "tavg_mean_lag1","tavg_mean_lag2","tavg_mean_lag3",
        "prcp_sum_lag1","prcp_sum_lag2","prcp_sum_lag3",
        "sin_woy","cos_woy","lat","lon","num_mosquitoes"
    ]
    require_cols(df, feature_cols + ["trap_id","week_start",TARGET_COL], "post-merge")

    df_train = df[df["week_start"]<=train_cut].copy()
    df_val   = df[(df["week_start"]>train_cut)&(df["week_start"]<=val_cut)].copy()
    df_test  = df[df["week_start"]>val_cut].copy()
    if df_train.empty or df_val.empty or df_test.empty:
        raise RuntimeError("One or more splits are empty. Adjust DATE_MIN/DATE_MAX to broaden the window.")

    # ---------------- Matrices ----------------
    Xtr, ytr = df_train[feature_cols].values, df_train[TARGET_COL].values
    Xva, yva = df_val[feature_cols].values,   df_val[TARGET_COL].values
    Xte, yte = df_test[feature_cols].values,  df_test[TARGET_COL].values

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    # ---------------- Models ----------------
    logit = LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0,
        max_iter=3000, class_weight="balanced", random_state=RNG_SEED
    ).fit(Xtr_s, ytr)

    gb = GradientBoostingClassifier(
        n_estimators=180, learning_rate=0.06, max_depth=3, subsample=0.9, random_state=RNG_SEED
    ).fit(Xtr, ytr)

    # ---------------- Predict week-wise ----------------
    def predict_weekwise(model, frame, scale=None, smooth=False, eta=0.5):
        outs = []
        for wk, g in frame.groupby("week_start", sort=True):
            X = g[feature_cols].values
            Xs = scale.transform(X) if scale is not None else X
            p = model.predict_proba(Xs)[:, 1]
            if smooth and len(g)>=3:
                W = build_knn_graph(g[["lat","lon"]].values, k=5)
                p = laplacian_smooth_probs(p, W, eta=eta)
            outs.append(pd.DataFrame({
                "trap_id": g["trap_id"].values,
                "week_start": g["week_start"].values,
                "p": p
            }))
        return pd.concat(outs, ignore_index=True)

    pred_val_logit = predict_weekwise(logit, df_val,  scale=scaler, smooth=False)
    pred_val_gb    = predict_weekwise(gb,    df_val,  scale=None,   smooth=True, eta=0.5)
    pred_test_logit= predict_weekwise(logit, df_test, scale=scaler, smooth=False)
    pred_test_gb   = predict_weekwise(gb,    df_test, scale=None,   smooth=True, eta=0.5)

    # ---------------- Evaluation ----------------
    def evaluate(true_part, pred_df, name):
        m = true_part[["trap_id","week_start",TARGET_COL]].merge(pred_df, on=["trap_id","week_start"], how="inner")
        y, p = m[TARGET_COL].values, m["p"].values
        return {
            "model": name,
            "AUROC": roc_auc_score(y,p),
            "AUPRC": average_precision_score(y,p),
            "Brier": brier_score_loss(y,p)
        }, precision_recall_curve(y,p), roc_curve(y,p), m

    evv_l, pr_v_l, roc_v_l, _ = evaluate(df_val,  pred_val_logit, "LogitEN")
    evv_g, pr_v_g, roc_v_g, _ = evaluate(df_val,  pred_val_gb,    "GB+Graph")
    ev_tl, pr_tl, roc_tl, _   = evaluate(df_test, pred_test_logit,"LogitEN")
    ev_tg, pr_tg, roc_tg, mtest = evaluate(df_test, pred_test_gb,   "GB+Graph")

    eval_val  = pd.DataFrame([evv_l, evv_g]); eval_test = pd.DataFrame([ev_tl, ev_tg])
    eval_val.to_csv(os.path.join(TAB_DIR, "evaluation_validation.csv"), index=False)
    eval_test.to_csv(os.path.join(TAB_DIR, "evaluation_test.csv"), index=False)
    print("\nValidation metrics:\n", eval_val)
    print("\nTest metrics:\n", eval_test)

    # PR (Test)
    plt.figure()
    for (prec, rec, thr), name in [(pr_tl,"LogitEN"), (pr_tg,"GB+Graph")]:
        plt.plot(rec, prec, label=name)
    plt.title("Precision-Recall (Test, Real Data)")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "pr_curve_test.png"), dpi=150); plt.show()

    # ROC (Test)
    plt.figure()
    for (fpr, tpr, thr), name in [(roc_tl,"LogitEN"), (roc_tg,"GB+Graph")]:
        plt.plot(fpr, tpr, label=name)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.title("ROC (Test, Real Data)")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "roc_curve_test.png"), dpi=150); plt.show()

    # Calibration (GB+Graph)
    y_cal, p_cal = mtest[TARGET_COL].values, mtest["p"].values
    bins = np.quantile(p_cal, np.linspace(0,1,11)); bins[0]=0.0; bins[-1]=1.0
    bid = np.digitize(p_cal, bins) - 1
    calib = pd.DataFrame({"bin": bid, "p_hat": p_cal, "y": y_cal}).groupby("bin", as_index=False).agg(
        pred_mean=("p_hat","mean"), obs_rate=("y","mean"), count=("y","size")
    )
    calib.to_csv(os.path.join(TAB_DIR, "calibration_gb_test.csv"), index=False)

    plt.figure()
    plt.plot(calib["pred_mean"], calib["obs_rate"], marker="o")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.title("Calibration (GB+Graph, Test, Real Data)")
    plt.xlabel("Predicted probability (bin mean)"); plt.ylabel("Observed frequency")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "calibration_gb_test.png"), dpi=150); plt.show()

    # Importances
    gb_imp = pd.DataFrame({"feature": feature_cols, "importance": gb.feature_importances_}).sort_values("importance", ascending=False)
    gb_imp.to_csv(os.path.join(TAB_DIR, "gb_importance.csv"), index=False)

    plt.figure()
    plt.bar(gb_imp["feature"], gb_imp["importance"])
    plt.xticks(rotation=90)
    plt.title("Gradient Boosting Feature Importance (Real Data)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "gb_importance.png"), dpi=150); plt.show()

    # Threshold selection (GB+Graph) on validation (F1)
    mv = df_val[["trap_id","week_start",TARGET_COL]].merge(pred_val_gb, on=["trap_id","week_start"], how="inner")
    yv, pv = mv[TARGET_COL].values, mv["p"].values
    prec, rec, thr = precision_recall_curve(yv, pv)
    f1s = []
    for i in range(len(thr)):
        t = thr[i]
        yhat = (pv >= t).astype(int)
        tp = ((yhat==1)&(yv==1)).sum(); fp = ((yhat==1)&(yv==0)).sum(); fn = ((yhat==0)&(yv==1)).sum()
        pi = tp/(tp+fp) if (tp+fp)>0 else 0.0
        ri = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1s.append(2*pi*ri/(pi+ri) if (pi+ri)>0 else 0.0)
    best_idx = int(np.argmax(f1s)) if len(f1s)>0 else 0
    best_thr = thr[best_idx] if len(thr)>0 else 0.5
    with open(os.path.join(TAB_DIR, "chosen_threshold.json"), "w") as f:
        json.dump({"best_threshold_validation_GBGraph": float(best_thr)}, f, indent=2)

    # Apply to test
    yhat = (p_cal >= best_thr).astype(int)
    rep = classification_report(y_cal, yhat, output_dict=True)
    pd.DataFrame(rep).T.to_csv(os.path.join(TAB_DIR, "classification_report_gb_test.csv"))
    cm = confusion_matrix(y_cal, yhat)

    plt.figure()
    plt.imshow(cm, cmap=None)
    plt.title("Confusion Matrix (GB+Graph, Test, Real Data)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks([0,1],["Neg","Pos"]); plt.yticks([0,1],["Neg","Pos"])
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, int(v), ha="center", va="center")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "confusion_matrix_gb_test.png"), dpi=150); plt.show()

    # Save predictions
    mtest_out = mtest.copy(); mtest_out["y_pred"] = yhat
    mtest_out.to_csv(os.path.join(TAB_DIR, "predictions_gb_test.csv"), index=False)

    # Spatial snapshot: week with highest median risk
    wk_stats = mtest.groupby("week_start")["p"].median().sort_values(ascending=False)
    if len(wk_stats) > 0:
        top_week = wk_stats.index[0]
        snap = mtest[mtest["week_start"]==top_week].merge(
            df_test[["trap_id","lat","lon","week_start"]], on=["trap_id","week_start"], how="left"
        )
        plt.figure()
        plt.scatter(snap["lon"], snap["lat"], s=snap["p"]*800+10)
        plt.title(f"Spatial Risk Snapshot (GB+Graph) — Week starting {top_week.date()}")
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "spatial_snapshot_topweek.png"), dpi=150); plt.show()

    # ZIP everything
    zip_path = "wnv_ml_outputs_real.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUT_DIR):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, arcname=os.path.join("wnv_outputs_real", os.path.relpath(full, OUT_DIR)))
    print(f"[OK] All outputs saved. ZIP: {zip_path}")

if __name__ == "__main__":
    main()
