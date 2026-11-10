"""SafeSpend: reusable functions for scoring and explaining a bank statement."""
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

FEATURES = ["log_amount", "hour", "is_night", "amount_vs_usual",
            "merchant_is_rare", "city_is_rare", "rapid_repeat"]

# Which plain-language reason each feature belongs to
REASON_GROUP = {
    "log_amount": "amount", "amount_vs_usual": "amount",
    "hour": "time", "is_night": "time",
    "merchant_is_rare": "merchant",
    "city_is_rare": "city",
    "rapid_repeat": "repeat",
}


def build_features(statement):
    """Turn a raw statement (datetime, merchant, amount, city) into model features."""
    st = statement.copy()
    st["datetime"] = pd.to_datetime(st["datetime"])
    st = st.sort_values("datetime").reset_index(drop=True)

    st["hour"] = st["datetime"].dt.hour
    st["is_night"] = st["hour"].between(0, 5).astype(int)
    st["log_amount"] = np.log1p(st["amount"])
    st["amount_vs_usual"] = st["amount"] / st["amount"].median()
    st["merchant_is_rare"] = (st["merchant"].map(st["merchant"].value_counts()) <= 2).astype(int)
    st["city_is_rare"] = (st["city"].map(st["city"].value_counts()) <= 2).astype(int)
    st["mins_since_prev"] = st["datetime"].diff().dt.total_seconds().div(60).fillna(1e4)
    st["rapid_repeat"] = (st["mins_since_prev"] < 10).astype(int)
    return st


def score_statement(st):
    """Fit Isolation Forest on this person's statement and add a 0-100 risk score and level."""
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    iso.fit(st[FEATURES])

    raw = -iso.score_samples(st[FEATURES])
    st["risk_score"] = ((raw - raw.min()) / (raw.max() - raw.min()) * 100).round()
    st["level"] = pd.cut(st["risk_score"], bins=[-1, 59, 79, 100],
                         labels=["🟢 Normal", "🟡 Check", "🔴 High risk"])
    return st, iso


def user_profile(st):
    """What is 'normal' for this person."""
    return {
        "median_amount": st["amount"].median(),
        "usual_hours": (int(st["hour"].quantile(0.10)), int(st["hour"].quantile(0.90))),
        "home_city": st["city"].mode()[0],
    }


def reason_text(group, row, profile):
    """Turn one reason group into a plain-language sentence, or None if it isn't unusual."""
    if group == "amount" and row["amount_vs_usual"] >= 3:
        return (f"₹{row['amount']:,.0f} is {row['amount_vs_usual']:.0f}× your usual "
                f"spend of ₹{profile['median_amount']:,.0f}")
    if group == "time":
        lo, hi = profile["usual_hours"]
        if row["is_night"] or not (lo <= row["hour"] <= hi):
            return (f"Made at {row['datetime']:%I:%M %p}; you usually pay between "
                    f"{lo}:00 and {hi}:00")
    if group == "merchant" and row["merchant_is_rare"]:
        return f"You have rarely or never paid '{row['merchant']}' before"
    if group == "city" and row["city_is_rare"]:
        return f"Made in {row['city']}; you usually transact in {profile['home_city']}"
    if group == "repeat" and row["rapid_repeat"]:
        return f"Only {row['mins_since_prev']:.0f} min after your previous payment"
    return None


def explain_statement(st, iso, min_score=60):
    """Add a 'reasons' column for risky transactions, ordered by SHAP importance."""
    profile = user_profile(st)
    explainer = shap.TreeExplainer(iso)
    risky = st[st["risk_score"] >= min_score]
    shap_values = explainer.shap_values(risky[FEATURES])

    reasons = {}
    for i, (idx, row) in enumerate(risky.iterrows()):
        # Negative SHAP pushes a transaction towards "anomaly", so flip the sign
        push = pd.Series(-shap_values[i], index=FEATURES)
        group_push = push.groupby(REASON_GROUP).sum().sort_values(ascending=False)

        texts = [reason_text(g, row, profile) for g in group_push.index if group_push[g] > 0]
        texts = [t for t in texts if t]
        reasons[idx] = texts or ["Unusual combination of amount, time and place for you"]

    st["reasons"] = pd.Series(reasons)
    return st