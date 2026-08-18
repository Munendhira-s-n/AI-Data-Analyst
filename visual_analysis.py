import re
import pandas as pd
import streamlit as st


# =========================================================
# AI DATA ANALYSIS + VISUALIZATION ENGINE
# =========================================================
#
# Supports:
# - Natural language questions
# - Smart metric detection
# - Smart dimension detection
# - Comparison
# - Average
# - Total
# - Highest / Lowest
# - Top N / Bottom N
# - Threshold filtering
# - Percentage / Share
# - Trend / Time analysis
# - Distribution
# - Correlation / Relationship
# - Business insights
# - Automatic chart selection
# - Ambiguity handling
# - Robust fallbacks
#
# Main function:
#
#     generate_visual_analysis(df, question)
#
# Returns:
#     True  -> analysis generated
#     False -> unable to generate
#
# =========================================================


# =========================================================
# 1. BASIC UTILITIES
# =========================================================

def _normalize(value):
    """Normalize text for matching."""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip().lower()
    )


def _compact(value):
    """Remove spaces and underscores for loose matching."""

    return (
        _normalize(value)
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def _clean_columns(df):
    """Clean column names without modifying the data."""

    df = df.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


def _numeric_columns(df):
    """Return numeric columns."""

    return (
        df
        .select_dtypes(include="number")
        .columns
        .tolist()
    )


def _categorical_columns(df):
    """Return categorical/text columns."""

    return (
        df
        .select_dtypes(
            include=[
                "object",
                "category",
                "string"
            ]
        )
        .columns
        .tolist()
    )


def _numeric(df, column):
    """Safely convert a column to numeric."""

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


# =========================================================
# 2. COLUMN DETECTION
# =========================================================

def _find_column(df, candidates):
    """
    Find a column using exact and partial matching.
    """

    normalized = {
        _compact(column): column
        for column in df.columns
    }

    # Exact
    for candidate in candidates:

        key = _compact(candidate)

        if key in normalized:
            return normalized[key]

    # Partial
    for column in df.columns:

        column_key = _compact(column)

        for candidate in candidates:

            candidate_key = _compact(candidate)

            if (
                candidate_key
                and candidate_key in column_key
            ):
                return column

    return None


# =========================================================
# 3. QUESTION INTENT
# =========================================================

def _comparison_requested(question):

    q = _normalize(question)

    patterns = [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\b",
        r"\bdifference\b",
        r"\bhigher than\b",
        r"\blower than\b",
    ]

    return any(
        re.search(pattern, q)
        for pattern in patterns
    )


def _average_requested(question):

    q = _normalize(question)

    return any(
        word in q
        for word in [
            "average",
            "avg",
            "mean"
        ]
    )


def _total_requested(question):

    q = _normalize(question)

    return any(
        word in q
        for word in [
            "total",
            "sum",
            "overall"
        ]
    )


def _highest_requested(question):

    q = _normalize(question)

    return any(
        word in q
        for word in [
            "highest",
            "maximum",
            "max",
            "top",
            "most",
            "best",
            "largest",
            "greatest"
        ]
    )


def _lowest_requested(question):

    q = _normalize(question)

    return any(
        word in q
        for word in [
            "lowest",
            "minimum",
            "min",
            "least",
            "worst",
            "smallest"
        ]
    )


def _is_distribution_question(question):

    q = _normalize(question)

    return any(
        phrase in q
        for phrase in [
            "distribution",
            "spread",
            "frequency",
            "histogram",
            "distribute"
        ]
    )


def _is_relationship_question(question):

    q = _normalize(question)

    return any(
        phrase in q
        for phrase in [
            "relationship",
            "correlation",
            "correlated",
            "relation between",
            "relationship between",
            "against"
        ]
    )


def _is_share_question(question):

    q = _normalize(question)

    return any(
        phrase in q
        for phrase in [
            "share",
            "percentage",
            "percent",
            "proportion",
            "contribution",
            "contribute",
            "breakdown"
        ]
    )


def _is_time_question(question):

    q = _normalize(question)

    return any(
        phrase in q
        for phrase in [
            "trend",
            "over time",
            "by month",
            "per month",
            "monthly",
            "month",
            "by year",
            "per year",
            "yearly",
            "year",
            "by day",
            "per day",
            "daily",
            "date",
            "dates",
            "timeline"
        ]
    )


def _is_count_question(question):

    q = _normalize(question)

    return any(
        phrase in q
        for phrase in [
            "count",
            "number of",
            "how many",
            "records",
            "employees",
            "customers"
        ]
    )


# =========================================================
# 4. METRIC ALIASES
# =========================================================

METRIC_ALIASES = {

    "sales": [
        "sales",
        "sale",
        "revenue",
        "turnover",
        "selling"
    ],

    "revenue": [
        "revenue",
        "sales",
        "turnover",
        "income"
    ],

    "salary": [
        "salary",
        "salaries",
        "pay",
        "wage",
        "wages",
        "income",
        "compensation"
    ],

    "profit": [
        "profit",
        "profits",
        "net profit",
        "margin"
    ],

    "quantity": [
        "quantity",
        "units",
        "volume"
    ],

    "performance": [
        "performance",
        "score",
        "rating"
    ],

    "experience": [
        "experience",
        "years",
        "tenure"
    ],

    "age": [
        "age"
    ],

    "price": [
        "price",
        "cost",
        "amount"
    ]
}


def _find_metric_column(df, question):

    q = _normalize(question)

    numeric_columns = _numeric_columns(df)

    if not numeric_columns:
        return None

    # -----------------------------------------------------
    # Exact column mention
    # -----------------------------------------------------

    for column in numeric_columns:

        column_text = _normalize(column)

        variants = [
            column_text,
            column_text.replace("_", " "),
            column_text.replace("-", " ")
        ]

        for variant in variants:

            if variant and variant in q:
                return column

    # -----------------------------------------------------
    # Alias detection
    # -----------------------------------------------------

    for column in numeric_columns:

        column_text = _compact(column)

        for alias_group in METRIC_ALIASES.values():

            for alias in alias_group:

                if _compact(alias) in column_text:

                    if any(
                        _compact(alias) in _compact(q)
                        for alias in alias_group
                    ):
                        return column

    # -----------------------------------------------------
    # Question contains metric alias
    # -----------------------------------------------------

    for canonical, aliases in METRIC_ALIASES.items():

        if any(
            alias in q
            for alias in aliases
        ):

            for column in numeric_columns:

                column_text = _compact(column)

                if (
                    _compact(canonical)
                    in column_text
                ):
                    return column

                if any(
                    _compact(alias)
                    in column_text
                    for alias in aliases
                ):
                    return column

    # -----------------------------------------------------
    # Single numeric column
    # -----------------------------------------------------

    if len(numeric_columns) == 1:
        return numeric_columns[0]

    return None


# =========================================================
# 5. DIMENSION DETECTION
# =========================================================

DIMENSION_WORDS = [
    "department",
    "dept",
    "region",
    "area",
    "location",
    "city",
    "state",
    "country",
    "category",
    "segment",
    "product",
    "team",
    "division",
    "gender",
    "customer",
    "employee",
    "name",
    "type",
    "status",
    "group"
]


def _mentioned_values(
    df,
    column,
    question
):

    question_text = _normalize(question)

    values = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    matches = []

    for value in values:

        value_text = _normalize(value)

        if not value_text:
            continue

        pattern = (
            r"(?<!\w)"
            + re.escape(value_text)
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            question_text
        ):

            matches.append(
                value
            )

    return matches


def _find_dimension_column(
    df,
    question
):

    categorical_columns = _categorical_columns(df)

    if not categorical_columns:
        return None

    q = _normalize(question)

    # -----------------------------------------------------
    # Explicit column
    # -----------------------------------------------------

    for column in categorical_columns:

        column_text = _normalize(column)

        if column_text in q:
            return column

        if (
            column_text.replace("_", " ")
            in q
        ):
            return column

    # -----------------------------------------------------
    # Mentioned dataset values
    # -----------------------------------------------------

    best_column = None
    best_score = 0

    for column in categorical_columns:

        matches = _mentioned_values(
            df,
            column,
            question
        )

        if len(matches) > best_score:

            best_score = len(matches)
            best_column = column

    if best_score >= 1:
        return best_column

    # -----------------------------------------------------
    # Common dimensions
    # -----------------------------------------------------

    for column in categorical_columns:

        column_text = _normalize(column)

        for word in DIMENSION_WORDS:

            if word in column_text:

                if word in q:
                    return column

    # -----------------------------------------------------
    # If only one categorical column
    # -----------------------------------------------------

    if len(categorical_columns) == 1:
        return categorical_columns[0]

    return None


# =========================================================
# 6. FIND SECOND NUMERIC METRIC
# =========================================================

def _find_second_numeric_column(
    df,
    question,
    first_metric
):

    numeric_columns = _numeric_columns(df)

    remaining = [
        column
        for column in numeric_columns
        if column != first_metric
    ]

    if not remaining:
        return None

    q = _normalize(question)

    # Explicit mention
    for column in remaining:

        column_text = _normalize(column)

        if column_text in q:
            return column

        if (
            column_text.replace("_", " ")
            in q
        ):
            return column

    # Common relationships
    relationship_pairs = {

        "salary": [
            "experience",
            "age",
            "performance"
        ],

        "sales": [
            "profit",
            "quantity",
            "units"
        ],

        "experience": [
            "salary",
            "performance",
            "age"
        ]
    }

    first_text = _normalize(
        first_metric
    )

    for key, possible in relationship_pairs.items():

        if key in first_text:

            for column in remaining:

                column_text = _normalize(
                    column
                )

                if any(
                    item in column_text
                    for item in possible
                ):
                    return column

    return remaining[0]


# =========================================================
# 7. AGGREGATION DETECTION
# =========================================================

def _detect_aggregation(question):

    if _average_requested(question):
        return "mean"

    if _total_requested(question):
        return "sum"

    if _is_count_question(question):
        return "count"

    return "sum"


# =========================================================
# 8. THRESHOLD DETECTION
# =========================================================

def _extract_threshold(question):

    q = _normalize(question)

    patterns = [

        (
            r"(?:more than|greater than|above|over|exceeding)"
            r"\s*\$?\s*([\d,]+(?:\.\d+)?)",
            ">"
        ),

        (
            r"(?:less than|below|under)"
            r"\s*\$?\s*([\d,]+(?:\.\d+)?)",
            "<"
        ),

        (
            r"(?:at least|minimum of)"
            r"\s*\$?\s*([\d,]+(?:\.\d+)?)",
            ">="
        ),

        (
            r"(?:at most|maximum of)"
            r"\s*\$?\s*([\d,]+(?:\.\d+)?)",
            "<="
        )
    ]

    for pattern, operator in patterns:

        match = re.search(
            pattern,
            q
        )

        if match:

            value = float(
                match.group(1)
                .replace(",", "")
            )

            return operator, value

    return None, None


# =========================================================
# 9. TOP N DETECTION
# =========================================================

def _extract_limit(question):

    q = _normalize(question)

    match = re.search(
        r"\btop\s+(\d+)\b",
        q
    )

    if match:
        return int(match.group(1))

    match = re.search(
        r"\bbottom\s+(\d+)\b",
        q
    )

    if match:
        return int(match.group(1))

    match = re.search(
        r"\b(?:first|last)\s+(\d+)\b",
        q
    )

    if match:
        return int(match.group(1))

    return 1


# =========================================================
# 10. INTENT CLASSIFICATION
# =========================================================

def _detect_intent(question):

    if _is_distribution_question(question):
        return "distribution"

    if _is_relationship_question(question):
        return "relationship"

    if _is_time_question(question):
        return "trend"

    if _is_share_question(question):
        return "share"

    operator, threshold = _extract_threshold(question)

    if operator is not None:
        return "threshold"

    if _comparison_requested(question):
        return "comparison"

    if _highest_requested(question):
        return "highest"

    if _lowest_requested(question):
        return "lowest"

    if _average_requested(question):
        return "average"

    if _total_requested(question):
        return "total"

    if _is_count_question(question):
        return "count"

    return "grouped"


# =========================================================
# 11. DATE DETECTION
# =========================================================

def _find_date_column(df):

    datetime_columns = (
        df
        .select_dtypes(
            include=[
                "datetime",
                "datetimetz"
            ]
        )
        .columns
        .tolist()
    )

    if datetime_columns:
        return datetime_columns[0]

    candidates = [
        "date",
        "datetime",
        "timestamp",
        "month",
        "year",
        "time"
    ]

    for column in df.columns:

        column_text = _normalize(
            column
        )

        if any(
            candidate in column_text
            for candidate in candidates
        ):

            converted = pd.to_datetime(
                df[column],
                errors="coerce"
            )

            if converted.notna().sum() >= 2:
                return column

    return None


# =========================================================
# 12. CHART RENDERING
# =========================================================

def _render_chart(
    data,
    chart_type,
    title=None,
    dimension_col=None,
    metric_col=None
):

    if data is None or data.empty:
        return False

    if title:
        st.markdown(
            f"### 📊 {title}"
        )

    try:

        # -------------------------------------------------
        # BAR
        # -------------------------------------------------

        if chart_type == "bar":

            st.bar_chart(
                data,
                use_container_width=True
            )

            return True

        # -------------------------------------------------
        # LINE
        # -------------------------------------------------

        if chart_type == "line":

            st.line_chart(
                data,
                use_container_width=True
            )

            return True

        # -------------------------------------------------
        # PLOTLY
        # -------------------------------------------------

        import plotly.express as px

        # -------------------------------------------------
        # PIE
        # -------------------------------------------------

        if chart_type == "pie":

            if (
                dimension_col is None
                or metric_col is None
            ):
                return False

            fig = px.pie(
                data,
                names=dimension_col,
                values=metric_col
            )

            fig.update_layout(
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            return True

        # -------------------------------------------------
        # SCATTER
        # -------------------------------------------------

        if chart_type == "scatter":

            if (
                dimension_col is None
                or metric_col is None
            ):
                return False

            fig = px.scatter(
                data,
                x=dimension_col,
                y=metric_col
            )

            fig.update_layout(
                margin=dict(
                    l=20,
                    r=20,
                    t=30,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            return True

        # -------------------------------------------------
        # HISTOGRAM
        # -------------------------------------------------

        if chart_type == "histogram":

            if metric_col is None:
                return False

            fig = px.histogram(
                data,
                x=metric_col,
                nbins=20
            )

            fig.update_layout(
                margin=dict(
                    l=20,
                    r=20,
                    t=30,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            return True

    except Exception:

        # -------------------------------------------------
        # FALLBACK
        # -------------------------------------------------

        try:

            if chart_type == "scatter":

                st.scatter_chart(
                    data,
                    x=dimension_col,
                    y=metric_col,
                    use_container_width=True
                )

                return True

            if chart_type == "line":

                st.line_chart(
                    data,
                    use_container_width=True
                )

                return True

            if chart_type in [
                "pie",
                "histogram"
            ]:

                st.bar_chart(
                    data,
                    use_container_width=True
                )

                return True

        except Exception:
            return False

    return False


# =========================================================
# 13. SMART CHART SELECTION
# =========================================================

def _choose_chart_type(
    question,
    intent,
    dimension_col=None,
    metric_col=None,
    df=None
):

    if intent == "distribution":
        return "histogram"

    if intent == "relationship":

        if (
            df is not None
            and len(
                _numeric_columns(df)
            ) >= 2
        ):
            return "scatter"

    if intent == "trend":
        return "line"

    if intent == "share":
        return "pie"

    if intent in [
        "comparison",
        "highest",
        "lowest",
        "average",
        "total",
        "count",
        "threshold"
    ]:
        return "bar"

    if (
        dimension_col is not None
        and metric_col is not None
    ):
        return "bar"

    return "bar"


# =========================================================
# 14. INSIGHT GENERATOR
# =========================================================

def _format_value(value):

    try:

        value = float(value)

        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:,.2f}M"

        if abs(value) >= 1_000:
            return f"{value:,.2f}"

        return f"{value:,.2f}"

    except Exception:
        return str(value)


def _show_insight(
    winner,
    value,
    metric,
    direction="highest"
):

    value_text = _format_value(value)

    if direction == "highest":

        st.success(
            f"{winner} has the highest "
            f"{metric.lower()} at "
            f"{value_text}."
        )

    elif direction == "lowest":

        st.info(
            f"{winner} has the lowest "
            f"{metric.lower()} at "
            f"{value_text}."
        )


# =========================================================
# 15. GROUPED ANALYSIS
# =========================================================

def _group_data(
    df,
    dimension_col,
    metric_col,
    aggregation
):

    data = df.copy()

    data[metric_col] = _numeric(
        data,
        metric_col
    )

    data = data.dropna(
        subset=[
            dimension_col,
            metric_col
        ]
    )

    if data.empty:
        return None

    if aggregation == "mean":

        grouped = (
            data
            .groupby(dimension_col)[metric_col]
            .mean()
        )

    elif aggregation == "count":

        grouped = (
            data
            .groupby(dimension_col)[metric_col]
            .count()
        )

    else:

        grouped = (
            data
            .groupby(dimension_col)[metric_col]
            .sum()
        )

    return grouped.sort_values(
        ascending=False
    )


# =========================================================
# 16. HIGHEST / LOWEST
# =========================================================

def _show_ranking(
    df,
    question,
    dimension_col,
    metric_col,
    descending=True
):

    aggregation = _detect_aggregation(
        question
    )

    grouped = _group_data(
        df,
        dimension_col,
        metric_col,
        aggregation
    )

    if grouped is None or grouped.empty:
        return False

    limit = _extract_limit(
        question
    )

    grouped = grouped.sort_values(
        ascending=not descending
    )

    result = grouped.head(
        limit
    )

    label = (
        "Average"
        if aggregation == "mean"
        else "Total"
        if aggregation == "sum"
        else "Count"
    )

    if descending:

        title = (
            f"Top {limit} "
            f"{dimension_col} by "
            f"{label} {metric_col}"
            if limit > 1
            else
            f"Highest {label} {metric_col} "
            f"by {dimension_col}"
        )

    else:

        title = (
            f"Bottom {limit} "
            f"{dimension_col} by "
            f"{label} {metric_col}"
            if limit > 1
            else
            f"Lowest {label} {metric_col} "
            f"by {dimension_col}"
        )

    chart_data = (
        grouped
        .rename(
            f"{label} {metric_col}"
        )
    )

    _render_chart(
        chart_data,
        "bar",
        title,
        dimension_col,
        metric_col
    )

    table = (
        grouped
        .rename(
            f"{label} {metric_col}"
        )
        .reset_index()
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    winner = result.index[0]
    value = result.iloc[0]

    _show_insight(
        winner,
        value,
        metric_col,
        "highest"
        if descending
        else "lowest"
    )

    return True


# =========================================================
# 17. CATEGORY COMPARISON
# =========================================================

def _show_comparison(
    df,
    question,
    dimension_col,
    metric_col
):

    values = _mentioned_values(
        df,
        dimension_col,
        question
    )

    if len(values) < 2:
        return False

    selected = values[:2]

    aggregation = _detect_aggregation(
        question
    )

    data = df.copy()

    data[metric_col] = _numeric(
        data,
        metric_col
    )

    data["__dimension__"] = (
        data[dimension_col]
        .astype(str)
        .str.strip()
    )

    data = data.dropna(
        subset=[metric_col]
    )

    data = data[
        data["__dimension__"].isin(
            selected
        )
    ]

    if data.empty:
        return False

    if aggregation == "mean":

        grouped = (
            data
            .groupby("__dimension__")[metric_col]
            .mean()
        )

        label = "Average"

    elif aggregation == "count":

        grouped = (
            data
            .groupby("__dimension__")[metric_col]
            .count()
        )

        label = "Count"

    else:

        grouped = (
            data
            .groupby("__dimension__")[metric_col]
            .sum()
        )

        label = "Total"

    grouped = grouped.reindex(
        selected
    )

    if grouped.isna().any():
        return False

    chart_data = (
        grouped
        .rename(
            f"{label} {metric_col}"
        )
    )

    _render_chart(
        chart_data,
        "bar",
        f"{selected[0]} vs {selected[1]}",
        dimension_col,
        metric_col
    )

    table = (
        grouped
        .rename(
            f"{label} {metric_col}"
        )
        .reset_index()
    )

    table.columns = [
        dimension_col,
        f"{label} {metric_col}"
    ]

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    first = float(
        grouped.iloc[0]
    )

    second = float(
        grouped.iloc[1]
    )

    difference = abs(
        first - second
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Difference",
            _format_value(difference)
        )

    with col2:

        if first > second:
            higher = selected[0]

        elif second > first:
            higher = selected[1]

        else:
            higher = "Equal"

        st.metric(
            "Higher",
            higher
        )

    if first > second:

        st.success(
            f"{selected[0]} has higher "
            f"{label.lower()} {metric_col.lower()} "
            f"than {selected[1]}."
        )

    elif second > first:

        st.success(
            f"{selected[1]} has higher "
            f"{label.lower()} {metric_col.lower()} "
            f"than {selected[0]}."
        )

    else:

        st.info(
            f"{selected[0]} and {selected[1]} "
            f"have equal {label.lower()} "
            f"{metric_col.lower()}."
        )

    return True


# =========================================================
# 18. THRESHOLD ANALYSIS
# =========================================================

def _show_threshold(
    df,
    question,
    metric_col,
    dimension_col=None
):

    operator, threshold = _extract_threshold(
        question
    )

    if operator is None:
        return False

    data = df.copy()

    data[metric_col] = _numeric(
        data,
        metric_col
    )

    data = data.dropna(
        subset=[metric_col]
    )

    if operator == ">":
        result = data[
            data[metric_col] > threshold
        ]

    elif operator == "<":
        result = data[
            data[metric_col] < threshold
        ]

    elif operator == ">=":
        result = data[
            data[metric_col] >= threshold
        ]

    else:
        result = data[
            data[metric_col] <= threshold
        ]

    st.markdown(
        f"### 📊 {metric_col} "
        f"{operator} "
        f"{threshold:,.2f}"
    )

    if result.empty:

        st.warning(
            "No matching records were found."
        )

        return True

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )

    if dimension_col:

        chart_data = (
            result
            .groupby(dimension_col)[metric_col]
            .sum()
            .sort_values(
                ascending=False
            )
        )

    else:

        chart_data = (
            result[metric_col]
            .sort_values(
                ascending=False
            )
        )

    _render_chart(
        chart_data,
        "bar"
    )

    st.info(
        f"{len(result):,} matching "
        f"record(s) were found."
    )

    return True


# =========================================================
# 19. AVERAGE / TOTAL / COUNT
# =========================================================

def _show_grouped_analysis(
    df,
    question,
    dimension_col,
    metric_col
):

    aggregation = _detect_aggregation(
        question
    )

    grouped = _group_data(
        df,
        dimension_col,
        metric_col,
        aggregation
    )

    if grouped is None or grouped.empty:
        return False

    if aggregation == "mean":
        label = "Average"

    elif aggregation == "count":
        label = "Count"

    else:
        label = "Total"

    chart_data = (
        grouped
        .rename(
            f"{label} {metric_col}"
        )
    )

    _render_chart(
        chart_data,
        "bar",
        f"{label} {metric_col} "
        f"by {dimension_col}",
        dimension_col,
        metric_col
    )

    table = (
        grouped
        .rename(
            f"{label} {metric_col}"
        )
        .reset_index()
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    return True


# =========================================================
# 20. SHARE ANALYSIS
# =========================================================

def _show_share(
    df,
    dimension_col,
    metric_col
):

    grouped = _group_data(
        df,
        dimension_col,
        metric_col,
        "sum"
    )

    if grouped is None or grouped.empty:
        return False

    grouped = grouped.head(15)

    chart_data = (
        grouped
        .rename(metric_col)
        .reset_index()
    )

    _render_chart(
        chart_data,
        "pie",
        f"{metric_col} Share by "
        f"{dimension_col}",
        dimension_col,
        metric_col
    )

    total = grouped.sum()

    table = (
        grouped
        .rename("Value")
        .reset_index()
    )

    table["Percentage"] = (
        table["Value"]
        / total
        * 100
    )

    table["Percentage"] = (
        table["Percentage"]
        .round(2)
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    return True


# =========================================================
# 21. TREND ANALYSIS
# =========================================================

def _show_time_series(
    df,
    metric_col
):

    date_col = _find_date_column(
        df
    )

    if date_col is None:
        return False

    data = df[
        [
            date_col,
            metric_col
        ]
    ].copy()

    data[date_col] = pd.to_datetime(
        data[date_col],
        errors="coerce"
    )

    data[metric_col] = _numeric(
        data,
        metric_col
    )

    data = data.dropna()

    if data.empty:
        return False

    grouped = (
        data
        .groupby(date_col)[metric_col]
        .sum()
        .sort_index()
    )

    chart_data = grouped.rename(
        metric_col
    )

    _render_chart(
        chart_data,
        "line",
        f"{metric_col} Trend",
        date_col,
        metric_col
    )

    table = (
        grouped
        .rename(metric_col)
        .reset_index()
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    # Trend insight
    if len(grouped) >= 2:

        first = float(
            grouped.iloc[0]
        )

        last = float(
            grouped.iloc[-1]
        )

        if last > first:

            st.success(
                f"{metric_col} increased from "
                f"{_format_value(first)} to "
                f"{_format_value(last)}."
            )

        elif last < first:

            st.warning(
                f"{metric_col} decreased from "
                f"{_format_value(first)} to "
                f"{_format_value(last)}."
            )

        else:

            st.info(
                f"{metric_col} remained unchanged "
                f"between the first and last periods."
            )

    return True


# =========================================================
# 22. DISTRIBUTION
# =========================================================

def _show_distribution(
    df,
    metric_col
):

    data = df[
        [metric_col]
    ].copy()

    data[metric_col] = _numeric(
        data,
        metric_col
    )

    data = data.dropna()

    if data.empty:
        return False

    _render_chart(
        data,
        "histogram",
        f"Distribution of {metric_col}",
        None,
        metric_col
    )

    st.dataframe(
        data.describe()
        .round(2)
        .reset_index(),
        use_container_width=True,
        hide_index=True
    )

    return True


# =========================================================
# 23. CORRELATION
# =========================================================

def _show_relationship(
    df,
    question,
    metric_col
):

    second_metric = _find_second_numeric_column(
        df,
        question,
        metric_col
    )

    if second_metric is None:
        return False

    data = df[
        [
            second_metric,
            metric_col
        ]
    ].copy()

    data[second_metric] = _numeric(
        data,
        second_metric
    )

    data[metric_col] = _numeric(
        data,
        metric_col
    )

    data = data.dropna()

    if len(data) < 2:
        return False

    _render_chart(
        data,
        "scatter",
        f"{metric_col} vs {second_metric}",
        second_metric,
        metric_col
    )

    correlation = (
        data[
            [
                second_metric,
                metric_col
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    st.metric(
        "Correlation",
        f"{correlation:.2f}"
    )

    if correlation > 0.7:

        st.success(
            "Strong positive relationship detected."
        )

    elif correlation > 0.3:

        st.info(
            "Moderate positive relationship detected."
        )

    elif correlation < -0.7:

        st.success(
            "Strong negative relationship detected."
        )

    elif correlation < -0.3:

        st.info(
            "Moderate negative relationship detected."
        )

    else:

        st.info(
            "Weak or limited linear relationship detected."
        )

    return True


# =========================================================
# 24. GLOBAL DATASET SUMMARY
# =========================================================

def _dataset_summary(df):

    numeric = _numeric_columns(df)
    categorical = _categorical_columns(df)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric": numeric,
        "categorical": categorical
    }


# =========================================================
# 25. SMART AMBIGUITY HANDLING
# =========================================================

def _show_ambiguity_help(
    df,
    question,
    metric_col=None,
    dimension_col=None
):

    if metric_col is None:

        numeric = _numeric_columns(df)

        if numeric:

            st.warning(
                "I couldn't confidently identify "
                "the metric you want to analyze."
            )

            st.info(
                "Available numeric columns: "
                + ", ".join(
                    map(str, numeric)
                )
            )

        return True

    if dimension_col is None:

        categorical = _categorical_columns(df)

        if categorical:

            st.warning(
                "I found the metric, but couldn't "
                "confidently identify the grouping."
            )

            st.info(
                "Available dimensions: "
                + ", ".join(
                    map(str, categorical)
                )
            )

            return True

    return False


# =========================================================
# 26. MAIN ENGINE
# =========================================================

def generate_visual_analysis(
    df,
    question=None
):
    """
    Complete AI-style data analysis engine.

    Example questions:

    Which region has the highest sales?

    Compare North and South sales.

    What is the average salary by department?

    What is the total revenue by region?

    Show the sales trend by month.

    What percentage of sales comes from each region?

    Show the distribution of salary.

    Is salary related to experience?

    Which employees have salary above 50000?

    Top 5 regions by sales.

    Bottom 3 departments by salary.
    """

    # =====================================================
    # VALIDATION
    # =====================================================

    if (
        df is None
        or df.empty
        or not question
    ):
        return False

    data = _clean_columns(
        df
    )

    question = str(
        question
    ).strip()

    if not question:
        return False

    numeric_columns = _numeric_columns(
        data
    )

    categorical_columns = _categorical_columns(
        data
    )

    if not numeric_columns:

        st.warning(
            "This dataset does not contain "
            "numeric columns required for analysis."
        )

        return False

    # =====================================================
    # DETECT INTENT
    # =====================================================

    intent = _detect_intent(
        question
    )

    # =====================================================
    # DETECT METRIC
    # =====================================================

    metric_col = _find_metric_column(
        data,
        question
    )

    # =====================================================
    # DETECT DIMENSION
    # =====================================================

    dimension_col = _find_dimension_column(
        data,
        question
    )

    # =====================================================
    # IF METRIC UNKNOWN
    # =====================================================

    if metric_col is None:

        if len(numeric_columns) == 1:

            metric_col = numeric_columns[0]

        else:

            return _show_ambiguity_help(
                data,
                question,
                None,
                dimension_col
            )

    # =====================================================
    # 1. DISTRIBUTION
    # =====================================================

    if intent == "distribution":

        return _show_distribution(
            data,
            metric_col
        )

    # =====================================================
    # 2. RELATIONSHIP
    # =====================================================

    if intent == "relationship":

        if _show_relationship(
            data,
            question,
            metric_col
        ):
            return True

    # =====================================================
    # 3. TREND
    # =====================================================

    if intent == "trend":

        if _show_time_series(
            data,
            metric_col
        ):
            return True

        st.warning(
            "I couldn't find a valid date/time "
            "column for a trend analysis."
        )

        return True

    # =====================================================
    # 4. SHARE
    # =====================================================

    if intent == "share":

        if dimension_col:

            if _show_share(
                data,
                dimension_col,
                metric_col
            ):
                return True

        return _show_ambiguity_help(
            data,
            question,
            metric_col,
            dimension_col
        )

    # =====================================================
    # 5. THRESHOLD
    # =====================================================

    if intent == "threshold":

        if _show_threshold(
            data,
            question,
            metric_col,
            dimension_col
        ):
            return True

    # =====================================================
    # 6. COMPARISON
    # =====================================================

    if intent == "comparison":

        if dimension_col:

            if _show_comparison(
                data,
                question,
                dimension_col,
                metric_col
            ):
                return True

        # If explicit categories were not detected,
        # fall back to grouped comparison.

        if dimension_col:

            if _show_grouped_analysis(
                data,
                question,
                dimension_col,
                metric_col
            ):
                return True

        return _show_ambiguity_help(
            data,
            question,
            metric_col,
            dimension_col
        )

    # =====================================================
    # 7. HIGHEST
    # =====================================================

    if intent == "highest":

        if dimension_col:

            if _show_ranking(
                data,
                question,
                dimension_col,
                metric_col,
                descending=True
            ):
                return True

    # =====================================================
    # 8. LOWEST
    # =====================================================

    if intent == "lowest":

        if dimension_col:

            if _show_ranking(
                data,
                question,
                dimension_col,
                metric_col,
                descending=False
            ):
                return True

    # =====================================================
    # 9. AVERAGE
    # =====================================================

    if intent == "average":

        if dimension_col:

            if _show_grouped_analysis(
                data,
                question,
                dimension_col,
                metric_col
            ):
                return True

        # Dataset-wide average

        values = _numeric(
            data,
            metric_col
        ).dropna()

        if not values.empty:

            average = values.mean()

            st.metric(
                f"Average {metric_col}",
                _format_value(average)
            )

            st.success(
                f"The average {metric_col.lower()} "
                f"is {_format_value(average)}."
            )

            return True

    # =====================================================
    # 10. TOTAL
    # =====================================================

    if intent == "total":

        if dimension_col:

            if _show_grouped_analysis(
                data,
                question,
                dimension_col,
                metric_col
            ):
                return True

        values = _numeric(
            data,
            metric_col
        ).dropna()

        if not values.empty:

            total = values.sum()

            st.metric(
                f"Total {metric_col}",
                _format_value(total)
            )

            st.success(
                f"The total {metric_col.lower()} "
                f"is {_format_value(total)}."
            )

            return True

    # =====================================================
    # 11. COUNT
    # =====================================================

    if intent == "count":

        if dimension_col:

            counts = (
                data
                .groupby(dimension_col)
                .size()
                .sort_values(
                    ascending=False
                )
            )

            _render_chart(
                counts.rename("Count"),
                "bar",
                f"Count by {dimension_col}",
                dimension_col,
                "Count"
            )

            table = (
                counts
                .rename("Count")
                .reset_index()
            )

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True
            )

            return True

        count = len(data)

        st.metric(
            "Number of Records",
            f"{count:,}"
        )

        return True

    # =====================================================
    # 12. GENERIC GROUPED ANALYSIS
    # =====================================================

    if dimension_col:

        if _show_grouped_analysis(
            data,
            question,
            dimension_col,
            metric_col
        ):
            return True

    # =====================================================
    # 13. SINGLE-METRIC FALLBACK
    # =====================================================

    values = _numeric(
        data,
        metric_col
    ).dropna()

    if not values.empty:

        st.metric(
            metric_col,
            _format_value(
                values.sum()
            )
        )

        st.info(
            f"I interpreted the question using "
            f"{metric_col}."
        )

        return True

    # =====================================================
    # 14. NOTHING FOUND
    # =====================================================

    st.warning(
        "I couldn't determine a suitable analysis "
        "for this question."
    )

    return False