"""SafeSpend web app. Run from the project folder with:  streamlit run app/app.py"""
import json
import sys
from pathlib import Path

import altair as alt
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from safespend import build_features, score_statement, explain_statement, user_profile  # noqa: E402

APP_DIR = ROOT / "app"
MODELS = ROOT / "models"
REQUIRED = ["datetime", "merchant", "amount", "city"]

st.set_page_config(page_title="SafeSpend", page_icon="🛡️", layout="wide")


# ---------- Loading (cached so it only happens once) ----------
@st.cache_resource
def load_card_model():
    model = joblib.load(MODELS / "best_model.pkl")
    cfg = json.loads((MODELS / "threshold.json").read_text())
    high = cfg.get("high_risk", 0.5)
    check = cfg.get("check", cfg.get("threshold", 0.5))
    return model, shap.TreeExplainer(model), high, check


@st.cache_data
def load_demo_transactions():
    return pd.read_csv(APP_DIR / "demo_transactions.csv")


@st.cache_data
def load_sample_statement():
    return pd.read_csv(APP_DIR / "sample_statement.csv")


# ---------- Header ----------
st.title("🛡️ SafeSpend")
st.caption("Check your bank statement for suspicious transactions — with a plain-language reason for every alert.")

tab_statement, tab_card, tab_about = st.tabs(
    ["📄 Check my statement", "💳 Card fraud model demo", "ℹ️ How it works"])


# ---------- Tab 1: personal statement checker ----------
with tab_statement:
    st.write("Upload a CSV with the columns **datetime, merchant, amount, city** "
             "(one row per transaction), or try the sample statement.")
    st.info("Your file is processed in memory and never stored.", icon="🔒")

    col_up, col_sample = st.columns([3, 1])
    uploaded = col_up.file_uploader("Bank statement (CSV)", type="csv")
    use_sample = col_sample.toggle("Use sample statement", value=uploaded is None)

    raw = None
    if uploaded is not None and not use_sample:
        raw = pd.read_csv(uploaded)
    elif use_sample:
        raw = load_sample_statement()

    if raw is None:
        st.stop()

    raw.columns = [c.strip().lower() for c in raw.columns]
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        st.error(f"Your file is missing these columns: {', '.join(missing)}")
        st.stop()
    raw["amount"] = pd.to_numeric(raw["amount"], errors="coerce")
    raw = raw.dropna(subset=REQUIRED)
    if len(raw) < 30:
        st.warning("SafeSpend needs at least 30 transactions to learn what's normal for you.")
        st.stop()

    with st.spinner("Learning your normal spending pattern..."):
        stmt = build_features(raw[REQUIRED])
        stmt, iso = score_statement(stmt)
        stmt = explain_statement(stmt, iso)
        profile = user_profile(stmt)

    high = stmt[stmt["level"] == "🔴 High risk"]
    check = stmt[stmt["level"] == "🟡 Check"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions checked", f"{len(stmt):,}")
    m2.metric("🔴 High risk", len(high))
    m3.metric("🟡 Worth checking", len(check))
    m4.metric("Your usual spend", f"₹{profile['median_amount']:,.0f}")

    st.subheader("Alerts")
    alerts = stmt[stmt["risk_score"] >= 70].sort_values("risk_score", ascending=False)
    if alerts.empty:
        st.success("No suspicious transactions found. Everything matches your usual pattern. ✅")
    for _, r in alerts.iterrows():
        box = st.error if r["level"] == "🔴 High risk" else st.warning
        reasons = "\n".join(f"- {reason}" for reason in r["reasons"])
        box(f"**{r['level']} · risk {r['risk_score']:.0f}/100**  \n"
            f"**₹{r['amount']:,.0f}** at **{r['merchant']}** · {r['city']} · "
            f"{r['datetime']:%d %b %Y, %I:%M %p}\n\n{reasons}")

    if not high.empty:
        st.markdown(
            "**If you don't recognise a 🔴 transaction:** block your card in your bank app, "
            "call your bank's helpline, and report it to the National Cyber Crime Helpline "
            "**1930** or at **cybercrime.gov.in** as soon as possible.")

    st.subheader("Your spending over time")
    chart = (alt.Chart(stmt).mark_circle(size=60, opacity=0.75)
             .encode(x=alt.X("datetime:T", title="Date"),
                     y=alt.Y("amount:Q", title="Amount (₹, log scale)", scale=alt.Scale(type="log")),
                     color=alt.Color("level:N", title="Risk level",
                                     scale=alt.Scale(domain=["🟢 Normal", "🟡 Check", "🔴 High risk"],
                                                     range=["#2e9e5b", "#e0a800", "#d64545"])),
                     tooltip=["datetime:T", "merchant", "amount", "city", "risk_score"])
             .properties(height=350))
    st.altair_chart(chart, width="stretch")

    with st.expander("See all transactions"):
        show = stmt[REQUIRED + ["risk_score", "level"]].sort_values("datetime", ascending=False)
        st.dataframe(show, width="stretch", hide_index=True)
        st.download_button("Download results as CSV", show.to_csv(index=False),
                           file_name="safespend_results.csv", mime="text/csv")


# ---------- Tab 2: XGBoost card fraud model demo ----------
with tab_card:
    st.write("This is the **XGBoost model** trained on 284,807 real card transactions. "
             "Its inputs (V1–V28) are anonymised by the bank, so here you can test it on "
             "transactions from the held-out test set.")
    model, explainer, high_t, check_t = load_card_model()
    demo = load_demo_transactions()

    c1, c2 = st.columns(2)
    if c1.button("🎲 Random transaction", width="stretch"):
        st.session_state["pick"] = int(demo.sample(1).index[0])
    if c2.button("🎯 Random real fraud", width="stretch"):
        st.session_state["pick"] = int(demo[demo["Class"] == 1].sample(1).index[0])

    if "pick" in st.session_state:
        row = demo.loc[[st.session_state["pick"]]]
        X = row.drop(columns=["Class"])
        prob = float(model.predict_proba(X)[:, 1][0])
        actual = "Fraud" if int(row["Class"].iloc[0]) == 1 else "Genuine"

        if prob >= high_t:
            st.error(f"🔴 **High risk** — fraud probability {prob:.1%}")
        elif prob >= check_t:
            st.warning(f"🟡 **Check this** — fraud probability {prob:.1%}")
        else:
            st.success(f"🟢 **Looks normal** — fraud probability {prob:.2%}")
        st.write(f"Actual label in the dataset: **{actual}**")

        st.caption("Why the model decided this (SHAP): red bars push towards fraud, blue towards genuine.")
        sv = explainer(X)
        plt.figure()
        shap.plots.waterfall(sv[0], max_display=10, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
    else:
        st.write("Press a button to test the model.")


# ---------- Tab 3: about ----------
with tab_about:
    st.markdown("""
**SafeSpend uses two models:**

1. **Personal anomaly detection (your statement).** An Isolation Forest learns *your* normal
   pattern — typical amount, usual hours, usual merchants and cities — and flags transactions that
   don't fit. It needs no fraud labels. SHAP ranks which reasons mattered most, and each reason is
   written from your own numbers.

2. **Card fraud classifier (demo tab).** XGBoost trained on the Kaggle credit card fraud dataset
   (0.17% fraud). On a held-out test set: PR-AUC 0.83, 93.5% precision at the 0.5 threshold.

**Limitations:** SafeSpend looks at one transaction at a time, so fraud that matches your usual
amount, time and place won't be caught. It is a learning project, not a replacement for your
bank's own fraud checks.
""")