import os
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client, Client
from streamlit_cookies_manager import EncryptedCookieManager


# -----------------------------
# App config / UI
# -----------------------------
st.set_page_config(
    page_title="Weight Tracker App",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)
cookies = EncryptedCookieManager(
    prefix="wtapp/",
    password=get_env("COOKIE_PASSWORD", "change-me-to-a-long-random-string"),
)

if not cookies.ready():
    st.stop()
def persist_login(session_dict: dict):
    # Store refresh_token so we can restore login after refresh
    rt = session_dict.get("refresh_token")
    if rt:
        cookies["refresh_token"] = rt
        cookies.save()


def clear_persisted_login():
    if "refresh_token" in cookies:
        del cookies["refresh_token"]
        cookies.save()

# Minimal UI polish
st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      div[data-testid="stMetricValue"] {font-size: 1.6rem;}
      div[data-testid="stMetricLabel"] {font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Env helpers
# -----------------------------
def get_env(name: str, default: str | None = None) -> str:
    # Works with Streamlit secrets OR environment variables
    if name in st.secrets:
        return str(st.secrets[name])
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required config: {name}")
    return v


SUPABASE_URL = get_env("SUPABASE_URL")
SUPABASE_ANON_KEY = get_env("SUPABASE_ANON_KEY")

GOAL_WEIGHT = 142.0
GOAL_DATE = date(2026, 5, 1)

# -----------------------------
# Supabase helpers
# -----------------------------
def supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def authed_postgrest(client: Client, access_token: str):
    """
    Create an authed PostgREST client so RLS policies apply with the user's JWT.
    supabase-py v2 supports: client.postgrest.auth(token)
    """
    return client.postgrest.auth(access_token)


def sign_up(email: str, password: str):
    client = supabase_client()
    return client.auth.sign_up({"email": email, "password": password})


def sign_in(email: str, password: str):
    client = supabase_client()
    return client.auth.sign_in_with_password({"email": email, "password": password})


def sign_out():
    clear_persisted_login()
    st.session_state.pop("sb_session", None)
    st.session_state.pop("sb_user", None)


def ensure_session_fresh():
    """
    Refresh session if close to expiry.
    We store the supabase session dict in st.session_state["sb_session"].
    """
    sess = st.session_state.get("sb_session")
    if not sess:
        return

    # supabase session object typically has expires_at (unix seconds) and refresh_token
    expires_at = sess.get("expires_at")
    refresh_token = sess.get("refresh_token")
    if not expires_at or not refresh_token:
        return

    now_ts = int(datetime.now(tz=timezone.utc).timestamp())
    # refresh if expiring within 2 minutes
    if expires_at - now_ts <= 120:
        client = supabase_client()
        refreshed = client.auth.refresh_session(refresh_token)
        # refreshed can be a Session object; normalize to dict-like
        new_session = getattr(refreshed, "model_dump", None)
        if callable(new_session):
            refreshed = refreshed.model_dump()
        st.session_state["sb_session"] = refreshed
        st.session_state["sb_user"] = refreshed.get("user")


# -----------------------------
# Data helpers
# -----------------------------
def fetch_entries(access_token: str) -> pd.DataFrame:
    client = supabase_client()
    pg = authed_postgrest(client, access_token)

    resp = (
        pg.from_("weight_entries")
        .select("entry_date, weight_lbs, waist_in, notes, updated_at")
        .order("entry_date", desc=False)
        .execute()
    )

    data = resp.data or []
    if not data:
        return pd.DataFrame(columns=["entry_date", "weight_lbs", "waist_in", "notes", "updated_at"])

    df = pd.DataFrame(data)
    df["entry_date"] = pd.to_datetime(df["entry_date"]).dt.date
    df["weight_lbs"] = pd.to_numeric(df["weight_lbs"], errors="coerce")
    df["waist_in"] = pd.to_numeric(df["waist_in"], errors="coerce")
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
    return df


def upsert_entry(access_token: str, user_id: str, entry_date: date, weight_lbs: float, waist_in: float | None, notes: str | None):
    client = supabase_client()
    pg = authed_postgrest(client, access_token)

    payload = {
        "user_id": user_id,
        "entry_date": entry_date.isoformat(),
        "weight_lbs": float(weight_lbs),
        "waist_in": (None if waist_in is None else float(waist_in)),
        "notes": (None if not notes else notes.strip()),
    }

    # Upsert on the unique constraint (user_id, entry_date)
    # supabase PostgREST expects on_conflict columns
    pg.from_("weight_entries").upsert(payload, on_conflict="user_id,entry_date").execute()


def delete_entry(access_token: str, entry_date: date):
    client = supabase_client()
    pg = authed_postgrest(client, access_token)
    pg.from_("weight_entries").delete().eq("entry_date", entry_date.isoformat()).execute()


def compute_metrics(df: pd.DataFrame) -> dict:
    metrics = {
        "latest_weight": None,
        "avg_7": None,
        "avg_14": None,
        "trend_30_lbs_per_week": None,
        "weekly_change_7avg": None,
        "volatility_7_range": None,
        "streak_current": 0,
        "streak_best": 0,
        "days_left": (GOAL_DATE - date.today()).days,
        "required_lbs_per_week": None,
        "on_pace": None,
        "proj_goal_date": None,
        "estimated_lbs_per_week": None,
        "weekly_trend_up": None,
    }

    if df.empty:
        return metrics

    df = df.sort_values("entry_date").copy()
    df["entry_date_dt"] = pd.to_datetime(df["entry_date"])
    df.set_index("entry_date_dt", inplace=True)

    # Latest weight
    metrics["latest_weight"] = float(df["weight_lbs"].dropna().iloc[-1])

    # Rolling averages on daily series (by date index)
    df["avg_7"] = df["weight_lbs"].rolling(window=7, min_periods=4).mean()
    df["avg_14"] = df["weight_lbs"].rolling(window=14, min_periods=7).mean()

    # 7-day / 14-day latest values
    if not df["avg_7"].dropna().empty:
        metrics["avg_7"] = float(df["avg_7"].dropna().iloc[-1])
    if not df["avg_14"].dropna().empty:
        metrics["avg_14"] = float(df["avg_14"].dropna().iloc[-1])

    # Volatility: last 7 days range (use raw weights)
    last7 = df["weight_lbs"].dropna().tail(7)
    if len(last7) >= 2:
        metrics["volatility_7_range"] = float(last7.max() - last7.min())

    # Weekly rate of change based on 7-day averages:
    # compare last available 7d avg vs the one 7 days earlier (nearest)
    s7 = df["avg_7"].dropna()
    if len(s7) >= 10:
        latest_date = s7.index[-1]
        prior_target = latest_date - pd.Timedelta(days=7)
        # pick the closest prior point at/earlier than target
        prior = s7[s7.index <= prior_target]
        if not prior.empty:
            weekly_change = float(s7.iloc[-1] - prior.iloc[-1])  # + means gaining
            metrics["weekly_change_7avg"] = weekly_change
            metrics["weekly_trend_up"] = weekly_change > 0

    # 30-day trend line (linear regression)
    last30 = df["weight_lbs"].dropna().tail(30)
    if len(last30) >= 10:
        x = np.arange(len(last30), dtype=float)
        y = last30.values.astype(float)
        slope, intercept = np.polyfit(x, y, 1)  # lbs per day-step
        metrics["trend_30_lbs_per_week"] = float(slope * 7.0)

    # Estimated lbs/week loss using 7-day average trend (more stable)
    # Fit a line to last ~21 points of avg_7
    s7_for_fit = df["avg_7"].dropna().tail(21)
    if len(s7_for_fit) >= 10:
        x = np.arange(len(s7_for_fit), dtype=float)
        y = s7_for_fit.values.astype(float)
        slope, _ = np.polyfit(x, y, 1)
        metrics["estimated_lbs_per_week"] = float(slope * 7.0)  # +gain, -loss

    # Goal pacing
    days_left = metrics["days_left"]
    if metrics["avg_7"] is not None and days_left is not None:
        weeks_left = max(days_left / 7.0, 0.0)
        if weeks_left > 0:
            metrics["required_lbs_per_week"] = float((metrics["avg_7"] - GOAL_WEIGHT) / weeks_left)
        else:
            metrics["required_lbs_per_week"] = None

    # On pace? (need both required and estimated)
    if metrics["required_lbs_per_week"] is not None and metrics["estimated_lbs_per_week"] is not None:
        # required is positive if you need to lose weight (avg_7 > goal)
        # estimated is negative for losing
        required_loss = metrics["required_lbs_per_week"]
        estimated = metrics["estimated_lbs_per_week"]
        # Convert required to "change per week" direction:
        # if you need to lose, you want estimated <= -required_loss
        if required_loss > 0:
            metrics["on_pace"] = (estimated <= -required_loss + 1e-9)
        else:
            # already at/below goal: on pace by default
            metrics["on_pace"] = True

    # Projected goal date from estimated lbs/week
    if metrics["avg_7"] is not None and metrics["estimated_lbs_per_week"] is not None:
        est = metrics["estimated_lbs_per_week"]
        if est < -1e-6:  # losing
            weeks_to_goal = (metrics["avg_7"] - GOAL_WEIGHT) / (-est)
            if weeks_to_goal >= 0:
                metrics["proj_goal_date"] = (date.today() + timedelta(days=int(round(weeks_to_goal * 7))))

    # Streaks (based on entry dates present, not weight values)
    dates = sorted(set(pd.to_datetime(df.index).date))
    if dates:
        # current streak ending today or yesterday (so you don't get punished by time of day)
        today = date.today()
        date_set = set(dates)

        def streak_ending_at(end: date) -> int:
            s = 0
            d = end
            while d in date_set:
                s += 1
                d = d - timedelta(days=1)
            return s

        # If today's not logged yet, allow streak ending yesterday
        cur = streak_ending_at(today)
        if cur == 0:
            cur = streak_ending_at(today - timedelta(days=1))
        metrics["streak_current"] = cur

        # best streak overall
        best = 0
        for d in dates:
            # compute streak starting at d going forward quickly by checking end streaks
            pass
        # efficient best streak: walk sorted dates
        best = 1
        run = 1
        for i in range(1, len(dates)):
            if dates[i] == dates[i - 1] + timedelta(days=1):
                run += 1
                best = max(best, run)
            else:
                run = 1
        metrics["streak_best"] = best if dates else 0

    return metrics


def build_chart_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    d = df.sort_values("entry_date").copy()
    d["entry_date"] = pd.to_datetime(d["entry_date"])
    d.set_index("entry_date", inplace=True)
    d["7d_avg"] = d["weight_lbs"].rolling(7, min_periods=4).mean()
    return d[["weight_lbs", "7d_avg"]]


# -----------------------------
# Auth UI
# -----------------------------
def auth_screen():
    st.title("⚖️ Weight Tracker App")
    st.caption("Log daily weight and see trends without obsessing over single-day noise.")

    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", use_container_width=True)
        if submitted:
            try:
                res = sign_in(email.strip(), password)
                sess = res.session
                user = res.user
                # normalize to dict
                if hasattr(sess, "model_dump"):
                    sess = sess.model_dump()
                if hasattr(user, "model_dump"):
                    user = user.model_dump()
                st.session_state["sb_session"] = sess
                st.session_state["sb_user"] = user
                persist_login(sess)
                st.success("Logged in.")
                st.rerun()
            except Exception as e:
                st.error("Login failed. Check email/password and try again.")
                st.caption(str(e))

    with tab_signup:
        with st.form("signup_form", clear_on_submit=False):
            email = st.text_input("Email", placeholder="you@example.com", key="su_email")
            password = st.text_input("Password (min 6 chars)", type="password", key="su_pw")
            submitted = st.form_submit_button("Create account", use_container_width=True)
        if submitted:
            try:
                res = sign_up(email.strip(), password)
                st.success("Account created. You can log in now.")
                st.info("If you enabled email confirmation in Supabase, confirm your email before logging in.")
            except Exception as e:
                st.error("Signup failed.")
                st.caption(str(e))


# -----------------------------
# Main App UI
# -----------------------------
def app_screen():
    ensure_session_fresh()
    sess = st.session_state.get("sb_session") or {}
    user = st.session_state.get("sb_user") or {}

    access_token = sess.get("access_token")
    user_id = user.get("id")

    if not access_token or not user_id:
        sign_out()
        st.rerun()

    top = st.columns([1, 1])
    with top[0]:
        st.title("⚖️ Weight Tracker")
        st.caption(f"Goal: {GOAL_WEIGHT:.0f} lbs by {GOAL_DATE.strftime('%b %d, %Y')}")
    with top[1]:
        if st.button("Log out", use_container_width=True):
            sign_out()
            st.rerun()

    # Fetch data
    df = fetch_entries(access_token)

    # Logging form
    st.subheader("Log today")
    default_date = date.today()
    existing_today = None
    if not df.empty:
        match = df[df["entry_date"] == default_date]
        if not match.empty:
            existing_today = match.iloc[-1]

    with st.form("log_form", clear_on_submit=False):
        c1, c2 = st.columns([1, 1])
        with c1:
            entry_date = st.date_input("Date", value=default_date)
        with c2:
            weight = st.number_input("Weight (lbs)", min_value=0.0, max_value=999.0, value=float(existing_today["weight_lbs"]) if existing_today is not None else 0.0, step=0.1)

        waist_val = None
        if existing_today is not None and not pd.isna(existing_today.get("waist_in")):
            waist_val = float(existing_today["waist_in"])

        waist = st.number_input("Waist (in) — optional", min_value=0.0, max_value=200.0, value=waist_val if waist_val is not None else 0.0, step=0.1)
        notes = st.text_area("Notes — optional", value=str(existing_today["notes"]) if existing_today is not None and existing_today.get("notes") else "")

        submitted = st.form_submit_button("Save entry", use_container_width=True)

    # Interpret optional waist: treat 0 as None for convenience
    waist_clean = None if waist == 0.0 else float(waist)

    if submitted:
        if weight <= 0:
            st.error("Please enter a valid weight.")
        else:
            try:
                upsert_entry(access_token, user_id, entry_date, float(weight), waist_clean, notes)
                st.success("Saved.")
                st.rerun()
            except Exception as e:
                st.error("Could not save entry.")
                st.caption(str(e))

    # Metrics + insights
    metrics = compute_metrics(df)

    st.divider()
    st.subheader("Trends")

    m1, m2, m3 = st.columns(3)
    m1.metric("Latest", "—" if metrics["latest_weight"] is None else f'{metrics["latest_weight"]:.1f} lb')
    m2.metric("7-day avg", "—" if metrics["avg_7"] is None else f'{metrics["avg_7"]:.1f} lb')
    m3.metric("14-day avg", "—" if metrics["avg_14"] is None else f'{metrics["avg_14"]:.1f} lb')

    m4, m5, m6 = st.columns(3)
    est = metrics["estimated_lbs_per_week"]
    m4.metric("Est. lbs/week", "—" if est is None else f"{est:+.2f}")
    wc = metrics["weekly_change_7avg"]
    m5.metric("Weekly change (7d avg)", "—" if wc is None else f"{wc:+.2f}")
    vol = metrics["volatility_7_range"]
    m6.metric("7-day range", "—" if vol is None else f"{vol:.1f} lb")

    # Goal pacing block
    st.divider()
    st.subheader("Goal pacing")

    days_left = metrics["days_left"]
    req = metrics["required_lbs_per_week"]
    on_pace = metrics["on_pace"]
    proj = metrics["proj_goal_date"]

    g1, g2, g3 = st.columns(3)
    g1.metric("Days left", f"{days_left}" if days_left is not None else "—")
    g2.metric("Required lbs/week", "—" if req is None else f"{req:.2f}")
    if on_pace is None:
        pace_txt = "—"
    else:
        pace_txt = "On pace ✅" if on_pace else "Behind ⚠️"
    g3.metric("Status", pace_txt)

    if proj:
        st.caption(f"Projected goal date (based on recent trend): **{proj.strftime('%b %d, %Y')}**")

    # Behavior insights (gentle, smoothed)
    st.divider()
    st.subheader("Insights")
    insights = []

    if metrics["weekly_trend_up"] is True:
        insights.append("Your **weekly trend is up** (based on 7-day averages). Consider tightening consistency for a week before judging outcomes.")
    elif metrics["weekly_trend_up"] is False and metrics["weekly_change_7avg"] is not None:
        insights.append("Your **weekly trend is down** (based on 7-day averages). Keep doing what’s working.")

    if est is not None:
        if abs(est) < 0.2:
            insights.append("Your recent trend is **pretty flat**. That’s normal—give changes ~2 weeks before making big adjustments.")
        elif est < 0:
            insights.append(f"Estimated pace: **{abs(est):.2f} lb/week loss** (smoothed).")
        else:
            insights.append(f"Estimated pace: **{abs(est):.2f} lb/week gain** (smoothed).")

    if vol is not None:
        insights.append(f"Scale volatility (last 7 days): **{vol:.1f} lb range**. Don’t overreact to a single-day spike.")

    if metrics["streak_current"] is not None:
        insights.append(f"Logging streak: **{metrics['streak_current']} days** (best: {metrics['streak_best']} days).")

    if not insights:
        insights.append("Log at least ~10 days to unlock meaningful averages and projections.")

    for i in insights:
        st.write("• " + i)

    # Chart
    st.divider()
    st.subheader("Chart")
    chart_df = build_chart_df(df)
    if chart_df.empty:
        st.info("No entries yet. Add your first log above.")
    else:
        st.line_chart(chart_df, use_container_width=True)

    # Recent logs + delete (optional)
    with st.expander("Recent logs"):
        if df.empty:
            st.write("No data.")
        else:
            view = df.sort_values("entry_date", ascending=False).head(14).copy()
            view["entry_date"] = view["entry_date"].astype(str)
            st.dataframe(view[["entry_date", "weight_lbs", "waist_in", "notes"]], use_container_width=True, hide_index=True)

            cdel1, cdel2 = st.columns([1, 1])
            with cdel1:
                del_date = st.date_input("Delete entry date", value=date.today(), key="del_date")
            with cdel2:
                if st.button("Delete that date", use_container_width=True):
                    try:
                        delete_entry(access_token, del_date)
                        st.success("Deleted.")
                        st.rerun()
                    except Exception as e:
                        st.error("Could not delete.")
                        st.caption(str(e))


# -----------------------------
# Router
# -----------------------------
def main():
    if "sb_session" not in st.session_state or "sb_user" not in st.session_state:
        # Attempt restore from cookie if session_state is empty
if ("sb_session" not in st.session_state or "sb_user" not in st.session_state) and "refresh_token" in cookies:
    try:
        client = supabase_client()
        refreshed = client.auth.refresh_session(cookies["refresh_token"])
        if hasattr(refreshed, "model_dump"):
            refreshed = refreshed.model_dump()
        st.session_state["sb_session"] = refreshed
        st.session_state["sb_user"] = refreshed.get("user")
    except Exception:
        # token expired/invalid — clear it and show login
        clear_persisted_login()
        auth_screen()
        return
    app_screen()


if __name__ == "__main__":
    main()
