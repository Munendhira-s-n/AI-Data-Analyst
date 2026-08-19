import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import json
import math
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

from analysis_engine import analyze_dataset
from dataset_profiler import profile_dataset
from visual_analysis import generate_visual_analysis


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SESSION STATE
# =========================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if "last_questions" not in st.session_state:
    st.session_state.last_questions = []

if "last_verified_results" not in st.session_state:
    st.session_state.last_verified_results = []

if "last_ai_answer" not in st.session_state:
    st.session_state.last_ai_answer = None

if "ai_insights" not in st.session_state:
    st.session_state.ai_insights = None

if "ai_insights_file" not in st.session_state:
    st.session_state.ai_insights_file = None


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 20px;
    }

    .insight-box {
        padding: 18px;
        border-radius: 12px;
        background: #172b42;
        border: 1px solid #23415f;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .success-box {
        padding: 18px;
        border-radius: 12px;
        background: #123524;
        border: 1px solid #1d6242;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .warning-box {
        padding: 18px;
        border-radius: 12px;
        background: #3b2d0b;
        border: 1px solid #765b14;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# GENERAL HELPERS
# =========================================================

def split_questions(text):
    if not text:
        return []

    text = text.replace("\r\n", "\n").strip()

    if not text:
        return []

    questions = []

    parts = re.split(r"\?", text)

    for part in parts:
        part = part.strip()

        if not part:
            continue

        part = re.sub(r"^\s*\d+\s*[\.\)\-:]\s*", "", part).strip()

        if part:
            questions.append(part + "?")

    if len(questions) <= 1:
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        if len(lines) > 1:
            questions = []

            for line in lines:
                line = re.sub(r"^\s*\d+\s*[\.\)\-:]\s*", "", line).strip()

                if not line:
                    continue

                if not line.endswith("?"):
                    line += "?"

                questions.append(line)

    return questions


def get_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df):
    categorical_types = ["object", "category", "bool", "string"]

    return df.select_dtypes(include=categorical_types).columns.tolist()


def get_date_columns(df):
    date_columns = []

    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            date_columns.append(column)

        elif (
            pd.api.types.is_object_dtype(df[column])
            or
            pd.api.types.is_string_dtype(df[column])
        ):
            converted = pd.to_datetime(df[column], errors="coerce")

            if (
                converted.notna().mean() >= 0.70
                and converted.notna().sum() > 0
            ):
                date_columns.append(column)

    return date_columns


def safe_numeric_series(df, column):
    if column not in df.columns:
        return pd.Series(dtype=float)

    return (
        pd.to_numeric(df[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def format_number(value):
    if value is None:
        return "N/A"

    try:
        value = float(value)

        if math.isnan(value):
            return "N/A"

        if value.is_integer():
            return f"{int(value):,}"

        return f"{value:,.2f}"

    except Exception:
        return str(value)


def clean_dataframe(df):
    cleaned = df.copy()

    cleaned = cleaned.dropna(how="all")

    cleaned = cleaned.dropna(axis=1, how="all")

    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    seen = {}

    new_columns = []

    for column in cleaned.columns:
        if column not in seen:
            seen[column] = 0
            new_columns.append(column)

        else:
            seen[column] += 1

            new_columns.append(f"{column}_{seen[column]}")

    cleaned.columns = new_columns

    return cleaned


# =========================================================
# COLUMN MATCHING
# =========================================================

def normalize_text(text):
    return (
        str(text)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def find_column(df, candidates):
    normalized = {normalize_text(column): column for column in df.columns}

    for candidate in candidates:
        candidate_key = normalize_text(candidate)

        if candidate_key in normalized:
            return normalized[candidate_key]

    for column in df.columns:
        column_key = normalize_text(column)

        for candidate in candidates:
            candidate_key = normalize_text(candidate)

            if candidate_key in column_key:
                return column

    return None


# =========================================================
# BUSINESS COLUMN DETECTION
# =========================================================

def detect_sales_column(df):
    return find_column(
        df,
        [
            "sales",
            "revenue",
            "total sales",
            "total revenue",
            "sales amount",
            "revenue amount",
            "amount"
        ]
    )


def detect_salary_column(df):
    return find_column(
        df,
        [
            "salary",
            "salaries",
            "annual salary",
            "monthly salary",
            "pay",
            "income",
            "compensation"
        ]
    )


def detect_quantity_column(df):
    return find_column(
        df,
        [
            "quantity",
            "qty",
            "units",
            "units sold",
            "quantity sold"
        ]
    )


def detect_product_column(df):
    return find_column(df, ["product", "product name", "item", "item name"])


def detect_region_column(df):
    return find_column(df, ["region", "area", "territory", "zone"])


def detect_department_column(df):
    return find_column(
        df,
        ["department", "dept", "division", "business unit"]
    )


def detect_segment_column(df):
    return find_column(df, ["customer segment", "segment", "customer type"])


def detect_employee_column(df):
    return find_column(
        df,
        ["employee", "employee name", "name", "employee id"]
    )


# =========================================================
# FIND QUESTION METRIC
# =========================================================

def detect_question_metric(df, question):
    q = normalize_text(question)

    if "salary" in q or "salaries" in q:
        column = detect_salary_column(df)

        if column:
            return column

    if "sales" in q or "revenue" in q:
        column = detect_sales_column(df)

        if column:
            return column

    if "quantity" in q or "qty" in q or "units" in q:
        column = detect_quantity_column(df)

        if column:
            return column

    for column in df.columns:
        column_words = normalize_text(column)

        if column_words in q:
            if column in get_numeric_columns(df):
                return column

    for column in df.columns:
        column_words = normalize_text(column)

        if (
            len(column_words) >= 3
            and column_words in q
            and column in get_numeric_columns(df)
        ):
            return column

    numeric_columns = get_numeric_columns(df)

    if len(numeric_columns) == 1:
        return numeric_columns[0]

    preferred = [
        "salary",
        "sales",
        "revenue",
        "amount",
        "income",
        "score",
        "performance",
        "quantity"
    ]

    for keyword in preferred:
        for column in numeric_columns:
            if keyword in normalize_text(column):
                return column

    return None


# =========================================================
# FIND QUESTION CATEGORY
# =========================================================

def detect_question_category(df, question):
    q = normalize_text(question)

    category_columns = get_categorical_columns(df)

    keywords = {
        "region": ["region", "area", "territory", "zone"],
        "department": ["department", "dept", "division"],
        "product": ["product", "item"],
        "segment": ["segment", "customer type"],
        "employee": ["employee", "employee name", "name"]
    }

    for _, words in keywords.items():
        for word in words:
            if word in q:
                for column in category_columns:
                    if word in normalize_text(column):
                        return column

    for column in category_columns:
        column_name = normalize_text(column)

        if column_name in q:
            return column

    return None


# =========================================================
# QUESTION TYPE
# =========================================================

def detect_question_type(question):
    q = normalize_text(question)

    if re.search(r"\b(top|highest|maximum|most|best)\b", q):
        return "highest"

    if re.search(r"\b(lowest|minimum|least|bottom)\b", q):
        return "lowest"

    if re.search(r"\b(average|mean)\b", q):
        return "average"

    if any(
        phrase in q
        for phrase in ["compare", "comparison", "versus", " vs ", "difference"]
    ):
        return "comparison"

    if any(
        phrase in q
        for phrase in ["trend", "over time", "monthly", "daily", "weekly", "yearly"]
    ):
        return "trend"

    if any(phrase in q for phrase in ["distribution", "spread"]):
        return "distribution"

    if any(phrase in q for phrase in ["correlation", "relationship", "related"]):
        return "correlation"

    if any(phrase in q for phrase in ["how many", "count", "number of"]):
        return "count"

    if any(phrase in q for phrase in ["total", "sum"]):
        return "total"

    if any(
        phrase in q
        for phrase in [
            "above",
            "greater than",
            "more than",
            "over",
            "below",
            "less than",
            "under"
        ]
    ):
        return "filter"

    return "general"


# =========================================================
# TOP N DETECTION
# =========================================================

def detect_top_n(question):
    match = re.search(r"\btop\s+(\d+)", question.lower())

    if match:
        return int(match.group(1))

    return None


# =========================================================
# QUESTION VISUALIZATION
# =========================================================

def create_question_visualization(df, question):
    q = normalize_text(question)

    question_type = detect_question_type(question)

    metric_column = detect_question_metric(df, question)

    category_column = detect_question_category(df, question)

    numeric_columns = get_numeric_columns(df)

    categorical_columns = get_categorical_columns(df)

    date_columns = get_date_columns(df)

    # =====================================================
    # CORRELATION
    # =====================================================

    if question_type == "correlation":
        if len(numeric_columns) < 2:
            return False

        correlation = (
            df[numeric_columns]
            .apply(pd.to_numeric, errors="coerce")
            .corr()
            .round(2)
        )

        st.dataframe(correlation, width="stretch")

        return True

    # =====================================================
    # TREND
    # =====================================================

    if question_type == "trend":
        if not date_columns or not numeric_columns:
            return False

        if metric_column is None:
            metric_column = numeric_columns[0]

        date_column = date_columns[0]

        trend_df = df.copy()

        trend_df[date_column] = pd.to_datetime(
            trend_df[date_column],
            errors="coerce"
        )

        trend_df[metric_column] = pd.to_numeric(
            trend_df[metric_column],
            errors="coerce"
        )

        trend_df = trend_df.dropna(subset=[date_column, metric_column])

        if trend_df.empty:
            return False

        trend_data = (
            trend_df
            .groupby(date_column)[metric_column]
            .sum()
            .sort_index()
        )

        st.line_chart(trend_data, width="stretch")

        return True

    # =====================================================
    # NUMERIC METRIC WITHOUT CATEGORY
    # =====================================================

    if metric_column and question_type in ["average", "total", "general", "distribution"]:
        series = safe_numeric_series(df, metric_column)

        if series.empty:
            return False

        if question_type == "average":
            value = series.mean()

            chart_data = pd.DataFrame({"Average": [value]})

            st.bar_chart(chart_data, width="stretch")

            st.success(f"Average {metric_column}: {format_number(value)}")

            return True

        if question_type == "total":
            value = series.sum()

            chart_data = pd.DataFrame({"Total": [value]})

            st.bar_chart(chart_data, width="stretch")

            st.success(f"Total {metric_column}: {format_number(value)}")

            return True

        if question_type == "distribution":
            if len(series) < 2:
                return False

            histogram = pd.cut(
                series,
                bins=10,
                duplicates="drop"
            ).value_counts(sort=False)

            histogram.index = histogram.index.astype(str)

            histogram_df = pd.DataFrame(
                {"Frequency": histogram.values},
                index=histogram.index
            )

            st.bar_chart(histogram_df, width="stretch")

            return True

    # =====================================================
    # CATEGORY + NUMERIC METRIC
    # =====================================================

    if metric_column and category_column:
        if category_column not in df.columns:
            return False

        data = df.copy()

        data[metric_column] = pd.to_numeric(data[metric_column], errors="coerce")

        data = data.dropna(subset=[category_column, metric_column])

        if data.empty:
            return False

        if question_type == "average":
            grouped = (
                data
                .groupby(category_column)[metric_column]
                .mean()
                .sort_values(ascending=False)
            )

        elif question_type == "total":
            grouped = (
                data
                .groupby(category_column)[metric_column]
                .sum()
                .sort_values(ascending=False)
            )

        elif question_type == "highest":
            grouped = (
                data
                .groupby(category_column)[metric_column]
                .sum()
                .sort_values(ascending=False)
            )

            n = detect_top_n(question)

            if n is None:
                n = 1

            grouped = grouped.head(min(n, len(grouped)))

        elif question_type == "lowest":
            grouped = (
                data
                .groupby(category_column)[metric_column]
                .sum()
                .sort_values(ascending=True)
            )

            n = detect_top_n(question)

            if n is None:
                n = 1

            grouped = grouped.head(min(n, len(grouped)))

        elif question_type == "comparison":
            grouped = (
                data
                .groupby(category_column)[metric_column]
                .sum()
                .sort_values(ascending=False)
            )

        else:
            grouped = (
                data
                .groupby(category_column)[metric_column]
                .sum()
                .sort_values(ascending=False)
            )

        if grouped.empty:
            return False

        display_data = grouped.head(20)

        chart_df = pd.DataFrame(
            {metric_column: display_data.values},
            index=display_data.index.astype(str)
        )

        st.bar_chart(chart_df, width="stretch")

        result_table = pd.DataFrame(
            {
                str(category_column): display_data.index.astype(str),
                str(metric_column): display_data.values
            }
        )

        st.dataframe(result_table, width="stretch", hide_index=True)

        if question_type == "highest":
            first_category = display_data.index[0]
            first_value = display_data.iloc[0]

            st.success(
                f"{first_category} has the highest "
                f"{metric_column} at "
                f"{format_number(first_value)}."
            )

        elif question_type == "lowest":
            first_category = display_data.index[0]
            first_value = display_data.iloc[0]

            st.success(
                f"{first_category} has the lowest "
                f"{metric_column} at "
                f"{format_number(first_value)}."
            )

        return True

    # =====================================================
    # NUMERIC OVERVIEW FALLBACK
    # =====================================================

    if metric_column and question_type in ["highest", "lowest"]:
        series = safe_numeric_series(df, metric_column)

        if series.empty:
            return False

        if question_type == "highest":
            value = series.max()
        else:
            value = series.min()

        chart_df = pd.DataFrame({metric_column: [value]})

        st.bar_chart(chart_df, width="stretch")

        st.success(f"{question_type.title()} {metric_column}: {format_number(value)}")

        return True

    # =====================================================
    # GENERIC NUMERIC FALLBACK
    # =====================================================

    if len(numeric_columns) >= 1 and question_type == "general":
        if metric_column:
            series = safe_numeric_series(df, metric_column)

            if not series.empty:
                st.bar_chart(series.head(20), width="stretch")

                return True

    return False


# =========================================================
# BUSINESS FACTS
# =========================================================

def build_business_facts(df):
    facts = {}

    numeric_columns = get_numeric_columns(df)

    categorical_columns = get_categorical_columns(df)

    facts["rows"] = len(df)

    facts["columns"] = len(df.columns)

    facts["missing_values"] = int(df.isna().sum().sum())

    facts["numeric_metrics"] = {}

    for column in numeric_columns:
        series = safe_numeric_series(df, column)

        if series.empty:
            continue

        facts["numeric_metrics"][column] = {
            "total": float(series.sum()),
            "average": float(series.mean()),
            "maximum": float(series.max()),
            "minimum": float(series.min()),
            "count": int(len(series))
        }

    facts["category_summaries"] = {}

    for category in categorical_columns:
        if category not in df.columns:
            continue

        if df[category].nunique(dropna=True) > 100:
            continue

        category_data = {}

        for metric in numeric_columns:
            temp = df.copy()

            temp[metric] = pd.to_numeric(temp[metric], errors="coerce")

            grouped = (
                temp
                .groupby(category)[metric]
                .sum()
                .sort_values(ascending=False)
            )

            if not grouped.empty:
                category_data[metric] = {
                    str(k): float(v)
                    for k, v in grouped.head(20).items()
                }

        if category_data:
            facts["category_summaries"][category] = category_data

    return facts


def call_gemini(prompt, timeout=180):
    import time

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        return None, "GEMINI_API_KEY was not found."

    client = genai.Client(api_key=api_key)

    last_error = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            answer = str(response.text or "").strip()

            if not answer:
                return None, "Gemini returned an empty response."

            return answer, None

        except Exception as e:
            last_error = e

            if "503" in str(e) or "UNAVAILABLE" in str(e):
                if attempt < 2:
                    time.sleep(5)
                    continue

            return None, f"Gemini request failed: {e}"

    return None, f"Gemini request failed after 3 attempts: {last_error}"
# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📊 AI Data Analyst</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Upload your dataset, explore business metrics,
    ask questions, generate visualizations and
    receive AI-powered insights.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("⚙️ Controls")

    st.write("Upload a CSV or Excel dataset to begin.")

    if st.session_state.chat_history:
        st.divider()

        st.subheader("🧠 Session")

        st.write(f"Questions answered: {len(st.session_state.chat_history)}")

    st.divider()

    st.caption(
        "Calculations are performed from the "
        "uploaded dataset before AI explanation."
    )


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📁 Upload a CSV or Excel file",
    type=["csv", "xlsx"],
    help="Supported formats: CSV and XLSX"
)


# =========================================================
# NO FILE
# =========================================================

if uploaded_file is None:
    st.info("👆 Upload a dataset to start the analysis.")

    st.stop()


# =========================================================
# DATASET CHANGE DETECTION
# =========================================================

current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"

if st.session_state.last_uploaded_file != current_file_id:
    st.session_state.last_uploaded_file = current_file_id

    st.session_state.ai_insights = None
    st.session_state.ai_insights_file = None

    st.session_state.last_questions = []
    st.session_state.last_verified_results = []
    st.session_state.last_ai_answer = None


# =========================================================
# READ DATASET
# =========================================================

try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)

except Exception as e:
    st.error(f"❌ Could not read the uploaded file: {e}")

    st.stop()


# =========================================================
# CLEAN DATASET
# =========================================================

df = clean_dataframe(df)

if df.empty:
    st.error("The uploaded dataset is empty.")

    st.stop()


# =========================================================
# FILE STATUS
# =========================================================

st.success(
    f"✅ Loaded **{uploaded_file.name}** — "
    f"{len(df):,} rows × "
    f"{len(df.columns):,} columns"
)


# =========================================================
# COLUMN DETECTION
# =========================================================

numeric_columns = get_numeric_columns(df)

categorical_columns = get_categorical_columns(df)

date_columns = get_date_columns(df)

sales_column = detect_sales_column(df)

salary_column = detect_salary_column(df)

quantity_column = detect_quantity_column(df)

product_column = detect_product_column(df)

region_column = detect_region_column(df)

department_column = detect_department_column(df)

segment_column = detect_segment_column(df)

employee_column = detect_employee_column(df)


# =========================================================
# DATASET PROFILE
# =========================================================

try:
    dataset_profile = profile_dataset(df)

except Exception:
    dataset_profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_details": []
    }


st.divider()

st.header("🔍 Dataset Profile")

profile_col1, profile_col2, profile_col3, profile_col4 = st.columns(4)

with profile_col1:
    st.metric("Rows", f"{len(df):,}")

with profile_col2:
    st.metric("Columns", f"{len(df.columns):,}")

with profile_col3:
    st.metric("Numeric Columns", f"{len(numeric_columns):,}")

with profile_col4:
    st.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")


# =========================================================
# DETECTED BUSINESS COLUMNS
# =========================================================

with st.expander("🎯 Detected Business Columns"):
    detected_columns = pd.DataFrame(
        {
            "Business Field": [
                "Sales",
                "Salary",
                "Quantity",
                "Product",
                "Region",
                "Department",
                "Customer Segment",
                "Employee"
            ],
            "Detected Column": [
                sales_column or "Not detected",
                salary_column or "Not detected",
                quantity_column or "Not detected",
                product_column or "Not detected",
                region_column or "Not detected",
                department_column or "Not detected",
                segment_column or "Not detected",
                employee_column or "Not detected"
            ]
        }
    )

    st.dataframe(detected_columns, width="stretch", hide_index=True)


# =========================================================
# PROFILE DETAILS
# =========================================================

if dataset_profile.get("column_details"):
    with st.expander("📋 Detailed Column Profile"):
        profile_table = pd.DataFrame(dataset_profile["column_details"])

        st.dataframe(profile_table, width="stretch", hide_index=True)


# =========================================================
# BUSINESS DASHBOARD
# =========================================================

st.divider()

st.header("📊 Business Dashboard")

st.caption("Automatically generated metrics based on the uploaded dataset.")


# =========================================================
# KPI CARDS
# =========================================================

kpi_values = []

kpi_values.append(("Total Records", f"{len(df):,}"))


if salary_column:
    salary_series = safe_numeric_series(df, salary_column)

    if not salary_series.empty:
        kpi_values.extend(
            [
                ("Total Salary", format_number(salary_series.sum())),
                ("Average Salary", format_number(salary_series.mean())),
                ("Maximum Salary", format_number(salary_series.max()))
            ]
        )


elif sales_column:
    sales_series = safe_numeric_series(df, sales_column)

    if not sales_series.empty:
        kpi_values.extend(
            [
                ("Total Sales", format_number(sales_series.sum())),
                ("Average Sales", format_number(sales_series.mean())),
                ("Maximum Sale", format_number(sales_series.max()))
            ]
        )


elif numeric_columns:
    primary_metric = numeric_columns[0]

    series = safe_numeric_series(df, primary_metric)

    if not series.empty:
        kpi_values.extend(
            [
                (f"Total {primary_metric}", format_number(series.sum())),
                (f"Average {primary_metric}", format_number(series.mean())),
                (f"Maximum {primary_metric}", format_number(series.max()))
            ]
        )


if quantity_column:
    quantity_series = safe_numeric_series(df, quantity_column)

    if not quantity_series.empty:
        kpi_values.append(("Total Quantity", format_number(quantity_series.sum())))


missing_values = int(df.isna().sum().sum())

kpi_values.append(("Missing Values", f"{missing_values:,}"))


kpi_values = kpi_values[:6]

kpi_columns = st.columns(len(kpi_values))

for index, (label, value) in enumerate(kpi_values):
    with kpi_columns[index]:
        st.metric(label, value)


# =========================================================
# AUTOMATIC CATEGORY SUMMARY
# =========================================================

if numeric_columns and categorical_columns:
    primary_metric = None

    if salary_column:
        primary_metric = salary_column

    elif sales_column:
        primary_metric = sales_column

    else:
        primary_metric = numeric_columns[0]

    primary_category = None

    if department_column:
        primary_category = department_column

    elif region_column:
        primary_category = region_column

    elif product_column:
        primary_category = product_column

    elif segment_column:
        primary_category = segment_column

    else:
        for column in categorical_columns:
            if df[column].nunique(dropna=True) <= 20:
                primary_category = column
                break

    if primary_category:
        dashboard_data = df.copy()

        dashboard_data[primary_metric] = pd.to_numeric(
            dashboard_data[primary_metric],
            errors="coerce"
        )

        dashboard_data = dashboard_data.dropna(
            subset=[primary_category, primary_metric]
        )

        grouped_dashboard = (
            dashboard_data
            .groupby(primary_category)[primary_metric]
            .sum()
            .sort_values(ascending=False)
        )

        if not grouped_dashboard.empty:
            st.subheader(f"📊 {primary_metric} by {primary_category}")

            st.bar_chart(grouped_dashboard.head(20), width="stretch")


# =========================================================
# DATA PREVIEW
# =========================================================

with st.expander("👀 View Data Preview"):
    st.dataframe(df.head(20), width="stretch", hide_index=True)


# =========================================================
# COLUMN INFORMATION
# =========================================================

with st.expander("🔤 Column Information"):
    column_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isna().sum().values,
            "Unique Values": [
                df[column].nunique(dropna=True) for column in df.columns
            ]
        }
    )

    st.dataframe(column_info, width="stretch", hide_index=True)


# =========================================================
# STATISTICAL SUMMARY
# =========================================================

with st.expander("📈 Statistical Summary"):
    try:
        st.dataframe(df.describe(include="all").T, width="stretch")

    except Exception:
        st.info("Statistical summary is not available for this dataset.")


# =========================================================
# AUTOMATIC VISUAL ANALYSIS
# =========================================================

st.divider()

st.header("📈 Automatic Visual Analysis")


# =========================================================
# NUMERIC DISTRIBUTION
# =========================================================

if numeric_columns:
    st.subheader("📊 Numeric Distribution")

    selected_numeric = st.selectbox(
        "Select numeric metric",
        numeric_columns,
        key="dashboard_distribution_metric"
    )

    numeric_series = safe_numeric_series(df, selected_numeric)

    if len(numeric_series) >= 2:
        try:
            histogram = pd.cut(
                numeric_series,
                bins=10,
                duplicates="drop"
            ).value_counts(sort=False)

            histogram.index = histogram.index.astype(str)

            histogram_df = pd.DataFrame(
                {"Frequency": histogram.values},
                index=histogram.index
            )

            st.bar_chart(histogram_df, width="stretch")

        except Exception:
            st.info("Could not create the distribution chart.")


# =========================================================
# CATEGORY PERFORMANCE
# =========================================================

if categorical_columns and numeric_columns:
    st.subheader("📊 Category Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_dimension = st.selectbox(
            "Category",
            categorical_columns,
            key="dashboard_dimension"
        )

    with col2:
        selected_metric = st.selectbox(
            "Metric",
            numeric_columns,
            key="dashboard_metric"
        )

    with col3:
        selected_aggregation = st.selectbox(
            "Aggregation",
            ["Sum", "Average", "Maximum", "Minimum", "Count"],
            key="dashboard_aggregation"
        )

    visual_df = df.copy()

    visual_df[selected_metric] = pd.to_numeric(
        visual_df[selected_metric],
        errors="coerce"
    )

    if selected_aggregation == "Sum":
        visual_data = (
            visual_df
            .groupby(selected_dimension)[selected_metric]
            .sum()
            .sort_values(ascending=False)
        )

    elif selected_aggregation == "Average":
        visual_data = (
            visual_df
            .groupby(selected_dimension)[selected_metric]
            .mean()
            .sort_values(ascending=False)
        )

    elif selected_aggregation == "Maximum":
        visual_data = (
            visual_df
            .groupby(selected_dimension)[selected_metric]
            .max()
            .sort_values(ascending=False)
        )

    elif selected_aggregation == "Minimum":
        visual_data = (
            visual_df
            .groupby(selected_dimension)[selected_metric]
            .min()
            .sort_values(ascending=False)
        )

    else:
        visual_data = (
            visual_df
            .groupby(selected_dimension)[selected_metric]
            .count()
            .sort_values(ascending=False)
        )

    visual_data = visual_data.dropna()

    if not visual_data.empty:
        st.bar_chart(visual_data.head(20), width="stretch")

    else:
        st.info("No data available for this visualization.")


# =========================================================
# TREND ANALYSIS
# =========================================================

if date_columns and numeric_columns:
    st.subheader("📅 Trend Analysis")

    col1, col2 = st.columns(2)

    with col1:
        selected_date = st.selectbox(
            "Date column",
            date_columns,
            key="dashboard_date"
        )

    with col2:
        selected_trend_metric = st.selectbox(
            "Trend metric",
            numeric_columns,
            key="dashboard_trend_metric"
        )

    trend_df = df.copy()

    trend_df[selected_date] = pd.to_datetime(
        trend_df[selected_date],
        errors="coerce"
    )

    trend_df[selected_trend_metric] = pd.to_numeric(
        trend_df[selected_trend_metric],
        errors="coerce"
    )

    trend_df = trend_df.dropna(subset=[selected_date, selected_trend_metric])

    if not trend_df.empty:
        trend_data = (
            trend_df
            .groupby(selected_date)[selected_trend_metric]
            .sum()
            .sort_index()
        )

        st.line_chart(trend_data, width="stretch")

    else:
        st.info("No valid date data was found.")


# =========================================================
# OUTLIER DETECTION
# =========================================================

if numeric_columns:
    st.subheader("🚨 Outlier Detection")

    outlier_metric = st.selectbox(
        "Select metric",
        numeric_columns,
        key="outlier_metric"
    )

    outlier_series = safe_numeric_series(df, outlier_metric)

    if len(outlier_series) >= 4:
        q1 = outlier_series.quantile(0.25)

        q3 = outlier_series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr

        upper_bound = q3 + 1.5 * iqr

        outliers = outlier_series[
            (outlier_series < lower_bound) | (outlier_series > upper_bound)
        ]

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Outliers", f"{len(outliers):,}")

        with c2:
            st.metric("Lower Bound", format_number(lower_bound))

        with c3:
            st.metric("Upper Bound", format_number(upper_bound))

        if not outliers.empty:
            st.dataframe(
                outliers.to_frame(name=outlier_metric).head(20),
                width="stretch"
            )

        else:
            st.success("No statistical outliers were detected.")


# =========================================================
# CORRELATION ANALYSIS
# =========================================================

if len(numeric_columns) >= 2:
    st.subheader("🔗 Correlation Analysis")

    correlation_matrix = (
        df[numeric_columns]
        .apply(pd.to_numeric, errors="coerce")
        .corr()
    )

    st.dataframe(correlation_matrix.round(2), width="stretch")

    correlation_pairs = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            column_a = correlation_matrix.columns[i]

            column_b = correlation_matrix.columns[j]

            value = correlation_matrix.iloc[i, j]

            if pd.notna(value):
                correlation_pairs.append((column_a, column_b, value))

    if correlation_pairs:
        strongest_pair = max(correlation_pairs, key=lambda x: abs(x[2]))

        st.info(
            f"Strongest observed correlation: "
            f"{strongest_pair[0]} and "
            f"{strongest_pair[1]} "
            f"({strongest_pair[2]:.2f})"
        )


# =========================================================
# NUMERIC METRICS OVERVIEW
# =========================================================

if numeric_columns:
    st.subheader("📌 Numeric Metrics Overview")

    metric_data = []

    for column in numeric_columns:
        series = safe_numeric_series(df, column)

        if not series.empty:
            metric_data.append(
                {
                    "Metric": column,
                    "Total": series.sum(),
                    "Average": series.mean(),
                    "Maximum": series.max(),
                    "Minimum": series.min(),
                    "Count": len(series)
                }
            )

    if metric_data:
        metrics_df = pd.DataFrame(metric_data)

        st.dataframe(metrics_df, width="stretch", hide_index=True)


# =========================================================
# AI BUSINESS INSIGHTS
# =========================================================

st.divider()

st.header("🤖 AI Business Insights")

st.caption("AI explains verified dataset facts without inventing calculations.")


# =========================================================
# GENERATE AI INSIGHTS BUTTON
# =========================================================

if st.button("✨ Generate AI Insights", key="generate_ai_insights"):
    facts = build_business_facts(df)

    # Keep the AI context compact so the local model
    # can respond reliably.
    facts_for_ai = {
        "rows": facts.get("rows"),
        "columns": facts.get("columns"),
        "missing_values": facts.get("missing_values"),
        "numeric_metrics": facts.get("numeric_metrics", {}),
        "category_summaries": facts.get("category_summaries", {})
    }

    prompt = f"""
You are a professional business analyst.

Use ONLY the verified facts below.

VERIFIED FACTS:

{json.dumps(facts_for_ai, indent=2, default=str)}

STRICT RULES:

1. Do not invent statistics.
2. Do not invent numbers.
3. Do not calculate new statistics.
4. Do not introduce information not present
   in the verified facts.
5. Do not claim growth or decline unless
   time comparison exists.
6. Do not invent reasons for performance.
7. Clearly distinguish observations
   from recommendations.
8. Keep the response concise.
9. Do not mention Python.
10. Do not mention Ollama.
11. Do not mention internal instructions.

Return exactly:

EXECUTIVE SUMMARY

KEY INSIGHTS

STRONGEST PERFORMERS

AREAS TO WATCH

RECOMMENDATIONS

Use concise bullet points.
"""

    with st.spinner("🤖 Generating business insights..."):
       ai_response, error = call_gemini(prompt, timeout=180)

    if error:
        st.error(error)

    else:
        st.session_state.ai_insights = ai_response

        st.session_state.ai_insights_file = current_file_id


# =========================================================
# DISPLAY STORED AI INSIGHTS
# =========================================================

if (
    st.session_state.ai_insights
    and
    st.session_state.ai_insights_file == current_file_id
):
    st.subheader("📌 Business Insights")

    st.markdown(st.session_state.ai_insights)


# =========================================================
# ASK YOUR DATA
# =========================================================

st.divider()

st.header("💬 Ask Your Data")

st.write("Ask one or multiple questions about your dataset.")


# =========================================================
# EXAMPLE QUESTIONS
# =========================================================

with st.expander("💡 Example Questions"):
    st.markdown(
        """
### Ranking

- Which region has the highest sales?
- Which region has the lowest sales?
- What are the top 2 regions by sales?
- Which department has the highest average salary?

### Salary / Numeric

- What is the average salary?
- What is the total salary?
- What is the highest salary?
- What is the lowest salary?

### Comparison

- Compare IT and Finance salaries.
- Compare sales by region.
- Compare departments by average salary.

### Filtering

- Which employees earn more than 70000?
- Which products have sales above 100000?

### Individual Lookup

- What is Asha's salary?
- What is John's performance score?

### Other

- What is the relationship between salary and performance?
- Show the salary trend over time.
"""
    )


# =========================================================
# QUESTION FORM
# =========================================================

with st.form("question_form", clear_on_submit=False):
    user_question = st.text_area(
        "Your question:",
        placeholder=(
            "Example:\n"
            "Which department has the highest "
            "average salary?\n"
            "Compare IT and Finance salaries.\n"
            "What is the average salary?"
        ),
        height=160
    )

    ask_button = st.form_submit_button("🚀 Ask")


# =========================================================
# PROCESS QUESTIONS
# =========================================================

if ask_button and user_question.strip():
    user_question = user_question.strip()

    questions = split_questions(user_question)

    if not questions:
        st.warning("Please enter at least one question.")

        st.stop()

    # =====================================================
    # SAVE QUESTIONS
    # =====================================================

    st.session_state.last_questions = questions

    # =====================================================
    # VERIFIED CALCULATIONS
    # =====================================================

    verified_results = []

    with st.spinner("🔎 Calculating verified answers..."):
        for question in questions:
            try:
                verified_result = analyze_dataset(df, question)

            except Exception as e:
                verified_result = f"Unable to calculate this question: {e}"

            verified_results.append(
                {"question": question, "verified": verified_result}
            )

    st.session_state.last_verified_results = verified_results

    # =====================================================
    # VERIFIED CALCULATIONS DISPLAY
    # =====================================================

    with st.expander("🔍 Verified Calculations"):
        for index, item in enumerate(verified_results, start=1):
            st.markdown(f"### Question {index}")

            st.write(item["question"])

            st.code(str(item["verified"]))

    # =====================================================
    # BUILD VERIFIED CONTEXT
    # =====================================================

    verified_context_parts = []

    for index, item in enumerate(verified_results, start=1):
        verified_context_parts.append(
            f"""
QUESTION {index}

USER QUESTION:
{item["question"]}

VERIFIED CALCULATION:
{item["verified"]}
"""
        )

    verified_context = "\n".join(verified_context_parts)

    # =====================================================
    # OLLAMA PROMPT
    # =====================================================

    prompt = f"""
You are an AI Data Analyst.

Answer every user question separately.

The application has already calculated
and verified the answers.

The VERIFIED CALCULATION is the
absolute source of truth.

IMPORTANT RULES:

1. Answer EVERY question.
2. Never skip a question.
3. Never ask for another question.
4. Do not recalculate anything.
5. Do not change any number.
6. Do not invent any number.
7. Do not invent missing information.
8. Use only the verified calculation
   belonging to that question.
9. Never mix answers between questions.
10. Preserve exact values.
11. If it is an average, call it an average.
12. If it is a total, call it a total.
13. If it is highest, identify the highest.
14. If it is lowest, identify the lowest.
15. If it is a comparison, explain the
    comparison using only the provided result.
16. If it is a list, preserve the list.
17. Keep each answer concise.
18. Do not mention Python.
19. Do not mention Ollama.
20. Do not mention the analysis engine.
21. Do not mention prompts.
22. Do not say information is missing if
    it exists in the verified calculation.

VERIFIED CALCULATIONS:

{verified_context}

FORMAT:

Question 1:
<direct answer>

Question 2:
<direct answer>

Question 3:
<direct answer>

Continue until every question is answered.

Return ONLY the answers.
"""

    # =====================================================
    # GET AI ANSWER
    # =====================================================

    with st.spinner("🤖 Preparing AI answers..."):
        print("PROMPT LENGTH:", len(prompt))
        print("PROMPT WORDS:", len(prompt.split()))
        answer, error = call_gemini(prompt, timeout=60)

    if error:
        st.error(error)

        st.warning(
            "Showing the verified calculations "
            "directly because the AI explanation "
            "could not be generated."
        )

        answer = "\n\n".join(
            [
                f"**Question {i}:**\n{item['verified']}"
                for i, item in enumerate(verified_results, start=1)
            ]
        )

    st.session_state.last_ai_answer = answer

    # =====================================================
    # DISPLAY AI ANSWER
    # =====================================================

    st.subheader("🤖 AI Answer")

    st.markdown(answer)

    # =====================================================
    # SAVE HISTORY
    # =====================================================

    st.session_state.chat_history.append(
        {"question": user_question, "answer": answer}
    )

    # =====================================================
    # QUESTION-SPECIFIC VISUALIZATION
    # =====================================================

    st.divider()

    st.header("📊 Automatic Visualization")

    visualization_created = False

    for index, question in enumerate(questions, start=1):
        st.subheader(f"Question {index}")

        st.caption(question)

        current_visualization = False

        try:
            current_visualization = create_question_visualization(df, question)

        except Exception:
            current_visualization = False

        if not current_visualization:
            try:
                current_visualization = generate_visual_analysis(df, question)

            except Exception:
                current_visualization = False

        if current_visualization:
            visualization_created = True

        else:
            st.info("No automatic visualization is available for this question.")

    if not visualization_created:
        st.caption(
            "The visualization engine did not find "
            "a suitable chart for the submitted questions."
        )


# =========================================================
# ANALYSIS HISTORY
# =========================================================

if st.session_state.chat_history:
    st.divider()

    st.header("🧠 Analysis History")

    for index, chat in enumerate(reversed(st.session_state.chat_history), start=1):
        with st.expander(f"Analysis {index}: {chat['question'][:80]}"):
            st.markdown(f"**You:** {chat['question']}")

            st.markdown("**🤖 AI:**")

            st.markdown(chat["answer"])


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "📊 AI Data Analyst • "
    "Verified calculations + AI explanations + "
    "dataset-aware automatic visualization"
)