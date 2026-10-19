
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timezone

st.set_page_config(
    page_title="SentinelAI | Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #07111f 0%, #101c31 100%);
        color: #e8eef9;
    }
    [data-testid="stSidebar"] {
        background-color: #0b1525;
    }
    [data-testid="stMetric"] {
        background: #14243a;
        padding: 18px;
        border: 1px solid #263b56;
        border-radius: 12px;
    }
    h1, h2, h3 {
        color: #e8eef9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "events" not in st.session_state:
    rng = np.random.default_rng(42)
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    st.session_state.events = pd.DataFrame({
        "timestamp": pd.date_range(end=now, periods=100, freq="min"),
        "source": rng.choice(
            ["Web Server", "API Gateway", "Database", "User Service"],
            size=100,
        ),
        "requests": rng.poisson(35, 100),
        "response_ms": rng.normal(180, 45, 100).clip(30, 1200).round(1),
        "risk_score": rng.integers(1, 101, 100),
        "status": rng.choice(
            ["Normal", "Warning", "Critical"],
            size=100,
            p=[0.78, 0.17, 0.05],
        ),
    })

st.title("🛡️ SentinelAI")
st.caption("Integrated Data Intelligence & Security Analytics")
st.divider()

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select module",
        [
            "Overview",
            "Data Explorer",
            "AI Anomaly Detection",
            "Security Center",
            "About",
        ],
    )

    st.divider()
    st.caption("SentinelAI Hackathon Edition")
    st.caption("Local development environment")

if page == "Overview":
    st.subheader("Platform Overview")
    st.write(
        "Monitor sample system events, inspect risk indicators, "
        "and explore the integrated analytics platform."
    )

    events = st.session_state.events
    total = len(events)
    critical = int((events["status"] == "Critical").sum())
    high_risk = int((events["risk_score"] >= 80).sum())
    avg_response = float(events["response_ms"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events Analyzed", f"{total:,}")
    c2.metric("Critical Events", critical)
    c3.metric("High-Risk Records", high_risk)
    c4.metric("Avg Response", f"{avg_response:.1f} ms")

    left, right = st.columns(2)

    with left:
        st.subheader("System Activity")
        activity = (
            events.set_index("timestamp")["requests"]
            .resample("10min")
            .sum()
            .reset_index()
        )
        fig = px.line(
            activity,
            x="timestamp",
            y="requests",
            markers=True,
            title="Requests per 10 Minutes",
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Event Distribution")
        distribution = events["status"].value_counts().reset_index()
        distribution.columns = ["status", "count"]
        fig = px.pie(
            distribution,
            names="status",
            values="count",
            hole=0.55,
            title="Sample Event Status",
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Events")
    st.dataframe(events.tail(10), use_container_width=True)

elif page == "Data Explorer":
    st.subheader("Data Explorer")
    st.write("Upload a CSV file to inspect and analyze its contents.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            if uploaded.size > 5 * 1024 * 1024:
                st.error("File exceeds the 5 MB upload limit.")
            else:
                data = pd.read_csv(uploaded, nrows=10000)

                st.success("CSV loaded successfully.")
                a, b, c = st.columns(3)
                a.metric("Rows Loaded", len(data))
                b.metric("Columns", len(data.columns))
                c.metric(
                    "Missing Cells",
                    int(data.isna().sum().sum()),
                )

                st.subheader("Data Preview")
                st.dataframe(data.head(100), use_container_width=True)

                st.subheader("Column Summary")
                st.dataframe(
                    data.describe(include="all").transpose(),
                    use_container_width=True,
                )

                numeric = data.select_dtypes(include="number").columns.tolist()
                if numeric:
                    selected = st.selectbox(
                        "Choose a numeric column",
                        numeric,
                    )
                    fig = px.histogram(
                        data,
                        x=selected,
                        title=f"Distribution of {selected}",
                    )
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

        except (ValueError, UnicodeDecodeError, pd.errors.ParserError):
            st.error("Unable to read this CSV. Check its format and encoding.")

elif page == "AI Anomaly Detection":
    st.subheader("AI Anomaly Detection")
    st.write(
        "Isolation Forest identifies unusual numerical records in the "
        "sample event dataset. Results are model-generated indicators, "
        "not proof of a cyberattack."
    )

    from sklearn.ensemble import IsolationForest

    events = st.session_state.events.copy()
    features = ["requests", "response_ms", "risk_score"]

    contamination = st.slider(
        "Expected proportion of unusual records",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
    )

    if st.button("Run anomaly detection", type="primary"):
        model = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=42,
        )
        events["anomaly"] = model.fit_predict(events[features])
        events["AI result"] = np.where(
            events["anomaly"] == -1,
            "Unusual",
            "Typical",
        )

        unusual = events[events["anomaly"] == -1]
        st.metric("Unusual Records Detected", len(unusual))
        st.dataframe(unusual, use_container_width=True)

        st.download_button(
            "Download Analysis CSV",
            data=events.to_csv(index=False).encode("utf-8"),
            file_name="sentinelai_anomaly_results.csv",
            mime="text/csv",
        )

elif page == "Security Center":
    st.subheader("Security Center")
    st.warning(
        "This is a demonstration dashboard using synthetic data. "
        "It does not monitor your computer, network, or live attacks."
    )

    events = st.session_state.events
    critical_events = events[events["status"] == "Critical"]

    st.metric("Sample Critical Events", len(critical_events))
    st.subheader("Risk Review")

    minimum_risk = st.slider(
        "Minimum risk score",
        min_value=1,
        max_value=100,
        value=70,
    )

    flagged = events[events["risk_score"] >= minimum_risk]
    st.dataframe(flagged, use_container_width=True)

    st.download_button(
        "Export Risk Review",
        data=flagged.to_csv(index=False).encode("utf-8"),
        file_name="sentinelai_risk_review.csv",
        mime="text/csv",
    )

elif page == "About":
    st.subheader("About SentinelAI")
    st.markdown(
        """
        **Mission:** Bring data analytics, machine learning, and
        security-oriented review into one application.

        **Integrated disciplines**
        - Data Science: descriptive statistics and visual analytics.
        - Artificial Intelligence: anomaly detection.
        - Cybersecurity: risk review and safer file handling.
        - Software Engineering: modular, testable application design.
        - Cloud Engineering: version control and deployment.

        **Current prototype:** Uses synthetic event data by default.
        Uploaded CSV files are analyzed in the running application.
        This prototype is not a certified security product.
        """
    )

st.divider()
st.caption("SentinelAI | Hackathon Prototype | Built with Python")
