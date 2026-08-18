import re
import math
import pandas as pd
import numpy as np
import streamlit as st


# ============================================================
# AI DATA ANALYST
# COMPLETE 8-STAGE VISUALIZATION ENGINE
# ============================================================
#
# STAGES
#
# 1. Question understanding
# 2. Metric detection
# 3. Dimension detection
# 4. Ranking / Top-N / Bottom-N
# 5. Aggregation intelligence
# 6. Trend / distribution / relationship / share
# 7. Correct chart selection
# 8. Clean result + table + business explanation
#
# PUBLIC API
#
# detect_visualization(question, df)
# render_visualization(df, config)
# generate_visual_analysis(df, question)
# visualize(df, question)
#
# ============================================================


# ============================================================
# 1. BASIC HELPERS
# ============================================================

def normalize(value):
    """Normalize text for matching."""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip().lower()
    )


def clean_column_name(value):
    """Normalize column names for matching."""

    return (
        normalize(value)
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def clean_dataframe(df):
    """Return a safe dataframe copy."""

    data = df.copy()

    data.columns = [
        str(column).strip()
        for column in data.columns
    ]

    return data


def numeric_columns(df):
    """Return numeric columns."""

    return (
        df
        .select_dtypes(include="number")
        .columns
        .tolist()
    )


def categorical_columns(df):
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


def numeric_series(df, column):
    """Safely convert a column to numeric."""

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def safe_float(value):
    """Safely convert a value to float."""

    try:
        return float(value)
    except Exception:
        return None


def format_value(value):
    """Business-friendly number formatting."""

    number = safe_float(value)

    if number is None:
        return str(value)

    if abs(number) >= 1_000_000:
        return f"{number:,.2f}"

    return f"{number:,.2f}"


# ============================================================
# 2. COLUMN DETECTION
# ============================================================

def find_column(df, possible_names):
    """
    Find a column using:
    1. exact normalized match
    2. partial match
    """

    columns = list(df.columns)

    normalized_map = {
        clean_column_name(column): column
        for column in columns
    }

    # Exact
    for name in possible_names:

        key = clean_column_name(name)

        if key in normalized_map:
            return normalized_map[key]

    # Partial
    for column in columns:

        column_key = clean_column_name(column)

        for name in possible_names:

            name_key = clean_column_name(name)

            if (
                name_key
                and name_key in column_key
            ):
                return column

    return None


# ============================================================
# METRIC ALIASES
# ============================================================

METRIC_ALIASES = {

    "sales": [
        "sales",
        "sale",
        "revenue",
        "turnover",
        "income",
        "gross sales",
        "total sales",
        "sales amount"
    ],

    "revenue": [
        "revenue",
        "sales",
        "turnover",
        "income"
    ],

    "profit": [
        "profit",
        "profits",
        "net profit",
        "gross profit",
        "profit margin"
    ],

    "salary": [
        "salary",
        "salaries",
        "pay",
        "wage",
        "wages",
        "compensation"
    ],

    "quantity": [
        "quantity",
        "quantities",
        "units",
        "unit",
        "volume"
    ],

    "price": [
        "price",
        "unit price",
        "selling price",
        "cost"
    ],

    "age": [
        "age"
    ],

    "experience": [
        "experience",
        "years experience",
        "years_experience",
        "tenure"
    ],

    "performance": [
        "performance",
        "performance score",
        "score",
        "rating",
        "ratings"
    ],

    "discount": [
        "discount",
        "discount percentage",
        "discount percent"
    ],

    "orders": [
        "orders",
        "order count",
        "number of orders"
    ]
}


def find_metric_column(df, question):
    """
    Intelligent numeric metric detection.
    """

    nums = numeric_columns(df)

    if not nums:
        return None

    q = normalize(question)

    # --------------------------------------------------------
    # 1. Explicit column name
    # --------------------------------------------------------

    for column in nums:

        column_text = normalize(column)

        variants = [
            column_text,
            column_text.replace("_", " "),
            column_text.replace("-", " "),
            clean_column_name(column)
        ]

        for variant in variants:

            if variant and variant in q:
                return column

    # --------------------------------------------------------
    # 2. Alias matching
    # --------------------------------------------------------

    for metric_group, aliases in METRIC_ALIASES.items():

        mentioned = any(
            normalize(alias) in q
            for alias in aliases
        )

        if not mentioned:
            continue

        for column in nums:

            column_key = clean_column_name(column)

            if any(
                clean_column_name(alias)
                in column_key
                for alias in aliases
            ):
                return column

    # --------------------------------------------------------
    # 3. Business priority
    # --------------------------------------------------------

    priority = [
        "sales",
        "revenue",
        "profit",
        "salary",
        "amount",
        "value",
        "quantity",
        "score",
        "price"
    ]

    for word in priority:

        for column in nums:

            if word in clean_column_name(column):
                return column

    # --------------------------------------------------------
    # 4. Single numeric column
    # --------------------------------------------------------

    if len(nums) == 1:
        return nums[0]

    # --------------------------------------------------------
    # 5. First numeric column
    # --------------------------------------------------------

    return nums[0]


# ============================================================
# DIMENSION DETECTION
# ============================================================

DIMENSION_ALIASES = [
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
    "person",
    "staff",
    "manager",
    "branch",
    "store",
    "channel"
]


def mentioned_values(df, column, question):
    """Find categorical values explicitly mentioned."""

    q = normalize(question)

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

        value_text = normalize(value)

        if not value_text:
            continue

        pattern = (
            r"(?<!\w)"
            + re.escape(value_text)
            + r"(?!\w)"
        )

        if re.search(pattern, q):

            matches.append(value)

    return matches


def find_dimension_column(df, question):
    """
    Intelligent categorical dimension detection.
    """

    cats = categorical_columns(df)

    if not cats:
        return None

    q = normalize(question)

    # --------------------------------------------------------
    # 1. Explicit column mention
    # --------------------------------------------------------

    for column in cats:

        variants = [
            normalize(column),
            normalize(column).replace("_", " "),
            normalize(column).replace("-", " "),
            clean_column_name(column)
        ]

        for variant in variants:

            if (
                variant
                and (
                    variant in q
                    or clean_column_name(variant)
                    in clean_column_name(q)
                )
            ):
                return column

    # --------------------------------------------------------
    # 2. Explicit values
    # --------------------------------------------------------

    best_column = None
    best_matches = 0

    for column in cats:

        matches = mentioned_values(
            df,
            column,
            question
        )

        if len(matches) > best_matches:

            best_matches = len(matches)
            best_column = column

    if best_column is not None:
        return best_column

    # --------------------------------------------------------
    # 3. Dimension aliases
    # --------------------------------------------------------

    for column in cats:

        column_key = clean_column_name(column)

        for alias in DIMENSION_ALIASES:

            alias_key = clean_column_name(alias)

            if alias_key in column_key:

                if (
                    alias in q
                    or alias.rstrip("s") in q
                ):
                    return column

    # --------------------------------------------------------
    # 4. Business-friendly categorical column
    # --------------------------------------------------------

    candidates = []

    for column in cats:

        unique_count = (
            df[column]
            .dropna()
            .nunique()
        )

        if 2 <= unique_count <= 50:

            candidates.append(
                (
                    unique_count,
                    column
                )
            )

    if candidates:

        # Prefer the lowest-cardinality sensible dimension
        candidates.sort(
            key=lambda x: x[0]
        )

        return candidates[0][1]

    return None


# ============================================================
# 3. QUESTION INTENT
# ============================================================

def comparison_requested(question):

    q = normalize(question)

    patterns = [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\b",
        r"\bdifference\b",
        r"\bbetween\b"
    ]

    return any(
        re.search(pattern, q)
        for pattern in patterns
    )


def average_requested(question):

    q = normalize(question)

    return any(
        word in q
        for word in [
            "average",
            "avg",
            "mean"
        ]
    )


def total_requested(question):

    q = normalize(question)

    return any(
        word in q
        for word in [
            "total",
            "sum",
            "overall"
        ]
    )


def highest_requested(question):

    q = normalize(question)

    return any(
        word in q
        for word in [
            "highest",
            "maximum",
            "max",
            "most",
            "best",
            "largest",
            "top"
        ]
    )


def lowest_requested(question):

    q = normalize(question)

    return any(
        word in q
        for word in [
            "lowest",
            "minimum",
            "min",
            "least",
            "worst",
            "smallest",
            "bottom"
        ]
    )


def top_requested(question):

    return bool(
        re.search(
            r"\btop\s+\d+",
            normalize(question)
        )
    )


def bottom_requested(question):

    return bool(
        re.search(
            r"\bbottom\s+\d+",
            normalize(question)
        )
    )


def threshold_requested(question):

    q = normalize(question)

    words = [
        "more than",
        "greater than",
        "above",
        "over",
        "exceeding",
        "less than",
        "below",
        "under",
        "at least",
        "at most",
        "minimum of",
        "maximum of"
    ]

    return any(
        word in q
        for word in words
    )


def distribution_requested(question):

    q = normalize(question)

    return any(
        word in q
        for word in [
            "distribution",
            "spread",
            "frequency",
            "histogram",
            "distributed"
        ]
    )


def relationship_requested(question):

    q = normalize(question)

    return any(
        word in q
        for word in [
            "relationship",
            "correlation",
            "correlated",
            "relation",
            "against"
        ]
    )


def share_requested(question):

    q = normalize(question)

    return any(
        word in q
        for word in [
            "share",
            "percentage",
            "percent",
            "proportion",
            "contribution",
            "breakdown"
        ]
    )


def time_requested(question):

    q = normalize(question)

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
            "timeline"
        ]
    )


# ============================================================
# TOP N
# ============================================================

def extract_top_n(question):

    q = normalize(question)

    patterns = [
        r"\btop\s+(\d+)",
        r"\bbottom\s+(\d+)",
        r"\bfirst\s+(\d+)",
        r"\blast\s+(\d+)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            q
        )

        if match:

            return max(
                1,
                int(match.group(1))
            )

    return 1


# ============================================================
# 4. DATE DETECTION
# ============================================================

def find_date_column(df):

    # Real datetime columns
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

    # Date-like column names
    date_words = [
        "date",
        "datetime",
        "timestamp",
        "month",
        "year",
        "time"
    ]

    for column in df.columns:

        column_text = normalize(column)

        if any(
            word in column_text
            for word in date_words
        ):

            converted = pd.to_datetime(
                df[column],
                errors="coerce"
            )

            if (
                converted.notna().mean()
                >= 0.50
            ):
                return column

    # Object columns
    for column in categorical_columns(df):

        converted = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        if (
            converted.notna().mean()
            >= 0.70
        ):
            return column

    return None


# ============================================================
# 5. AGGREGATION
# ============================================================

def determine_aggregation(question):

    if average_requested(question):
        return "mean"

    if total_requested(question):
        return "sum"

    return "sum"


def aggregate_grouped(
    df,
    dimension,
    metric,
    aggregation="sum"
):

    if (
        dimension is None
        or metric is None
        or dimension not in df.columns
        or metric not in df.columns
    ):
        return pd.Series(dtype="float64")

    data = df.copy()

    data[metric] = numeric_series(
        data,
        metric
    )

    data = data.dropna(
        subset=[
            dimension,
            metric
        ]
    )

    if data.empty:
        return pd.Series(dtype="float64")

    grouped_object = (
        data
        .groupby(
            dimension,
            dropna=True
        )[metric]
    )

    if aggregation == "mean":

        result = grouped_object.mean()

    elif aggregation == "count":

        result = grouped_object.count()

    elif aggregation == "min":

        result = grouped_object.min()

    elif aggregation == "max":

        result = grouped_object.max()

    else:

        result = grouped_object.sum()

    return result.dropna()


# ============================================================
# SECOND NUMERIC COLUMN
# ============================================================

def find_second_numeric_column(
    df,
    question,
    first_metric
):

    nums = numeric_columns(df)

    remaining = [
        column
        for column in nums
        if column != first_metric
    ]

    if not remaining:
        return None

    q = normalize(question)

    # Explicit mention
    for column in remaining:

        column_text = normalize(column)

        if column_text in q:
            return column

        if (
            column_text.replace("_", " ")
            in q
        ):
            return column

    # Relationship hints
    relationship_pairs = {

        "sales": [
            "profit",
            "quantity",
            "units",
            "price"
        ],

        "revenue": [
            "profit",
            "quantity",
            "price"
        ],

        "salary": [
            "experience",
            "age",
            "performance"
        ],

        "experience": [
            "salary",
            "performance",
            "age"
        ],

        "price": [
            "quantity",
            "sales",
            "revenue"
        ]
    }

    first_text = normalize(
        first_metric
    )

    for key, possible in relationship_pairs.items():

        if key in first_text:

            for column in remaining:

                column_text = normalize(
                    column
                )

                if any(
                    item in column_text
                    for item in possible
                ):
                    return column

    return remaining[0]


# ============================================================
# 6. PLOTLY CHART ENGINE
# ============================================================

def _plotly_available():

    try:

        import plotly.express

        return True

    except Exception:

        return False


def render_chart(
    data,
    chart_type,
    title=None,
    x=None,
    y=None
):

    if data is None:
        return False

    if isinstance(data, pd.Series):

        if data.empty:
            return False

        chart_df = (
            data
            .rename("Value")
            .reset_index()
        )

        chart_df.columns = [
            "Category",
            "Value"
        ]

        x = "Category"
        y = "Value"

    else:

        if data.empty:
            return False

        chart_df = data.copy()

    if title:
        st.markdown(
            f"### 📊 {title}"
        )

    # --------------------------------------------------------
    # PLOTLY
    # --------------------------------------------------------

    if _plotly_available():

        try:

            import plotly.express as px

            if chart_type == "bar":

                fig = px.bar(
                    chart_df,
                    x=x,
                    y=y,
                    text=y
                )

                fig.update_traces(
                    texttemplate="%{text:,.0f}",
                    textposition="outside",
                    cliponaxis=False
                )

            elif chart_type == "line":

                fig = px.line(
                    chart_df,
                    x=x,
                    y=y,
                    markers=True
                )

            elif chart_type == "pie":

                fig = px.pie(
                    chart_df,
                    names=x,
                    values=y,
                    hole=0.35
                )

            elif chart_type == "scatter":

                fig = px.scatter(
                    chart_df,
                    x=x,
                    y=y,
                    trendline="ols"
                )

            elif chart_type == "histogram":

                fig = px.histogram(
                    chart_df,
                    x=x,
                    nbins=20
                )

            else:

                return False

            fig.update_layout(
                height=450,
                margin=dict(
                    l=20,
                    r=20,
                    t=50,
                    b=50
                ),
                template="plotly_dark",
                hovermode="closest"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )

            return True

        except Exception:
            pass

    # --------------------------------------------------------
    # STREAMLIT FALLBACK
    # --------------------------------------------------------

    try:

        if chart_type == "bar":

            st.bar_chart(
                chart_df.set_index(x)[y]
                if x and y
                else chart_df,
                use_container_width=True
            )

            return True

        if chart_type == "line":

            st.line_chart(
                chart_df.set_index(x)[y]
                if x and y
                else chart_df,
                use_container_width=True
            )

            return True

        if chart_type == "scatter":

            st.scatter_chart(
                chart_df,
                x=x,
                y=y,
                use_container_width=True
            )

            return True

        if chart_type == "histogram":

            st.bar_chart(
                chart_df,
                use_container_width=True
            )

            return True

    except Exception:
        pass

    return False


# ============================================================
# 7. RESULT TABLE
# ============================================================

def show_result_table(
    grouped,
    dimension,
    metric_label
):

    if grouped is None or grouped.empty:
        return

    table = (
        grouped
        .rename(metric_label)
        .reset_index()
    )

    table.columns = [
        dimension,
        metric_label
    ]

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# COMPARISON
# ============================================================

def show_comparison(
    df,
    question,
    dimension,
    metric
):

    values = mentioned_values(
        df,
        dimension,
        question
    )

    if len(values) < 2:
        return False

    selected = values[:2]

    aggregation = determine_aggregation(
        question
    )

    grouped = aggregate_grouped(
        df,
        dimension,
        metric,
        aggregation
    )

    if grouped.empty:
        return False

    comparison = (
        grouped
        .reindex(selected)
        .dropna()
    )

    if len(comparison) != 2:
        return False

    label = (
        "Average"
        if aggregation == "mean"
        else "Total"
    )

    # Explicit dataframe prevents unwanted categories
    chart_df = (
        comparison
        .rename("Value")
        .reset_index()
    )

    chart_df.columns = [
        dimension,
        "Value"
    ]

    render_chart(
        chart_df,
        "bar",
        f"{selected[0]} vs {selected[1]}",
        dimension,
        "Value"
    )

    show_result_table(
        comparison,
        dimension,
        f"{label} {metric}"
    )

    first = float(
        comparison.iloc[0]
    )

    second = float(
        comparison.iloc[1]
    )

    difference = abs(
        first - second
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Difference",
            f"{difference:,.2f}"
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
            f"{label.lower()} {metric.lower()} "
            f"than {selected[1]}."
        )

    elif second > first:

        st.success(
            f"{selected[1]} has higher "
            f"{label.lower()} {metric.lower()} "
            f"than {selected[0]}."
        )

    else:

        st.info(
            f"{selected[0]} and {selected[1]} "
            f"have equal {label.lower()} "
            f"{metric.lower()}."
        )

    return True


# ============================================================
# RANKING
# ============================================================

def show_ranking(
    df,
    question,
    dimension,
    metric
):

    q = normalize(question)

    is_highest = (
        highest_requested(question)
        and not bottom_requested(question)
    )

    is_lowest = (
        lowest_requested(question)
        or bottom_requested(question)
    )

    if not is_highest and not is_lowest:
        return False

    aggregation = determine_aggregation(
        question
    )

    grouped = aggregate_grouped(
        df,
        dimension,
        metric,
        aggregation
    )

    if grouped.empty:
        return False

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    if is_lowest:

        sorted_data = grouped.sort_values(
            ascending=True
        )

    else:

        sorted_data = grouped.sort_values(
            ascending=False
        )

    # --------------------------------------------------------
    # LIMIT
    # --------------------------------------------------------

    if top_requested(question):

        limit = extract_top_n(question)

        sorted_data = (
            grouped
            .sort_values(
                ascending=False
            )
            .head(limit)
        )

    elif bottom_requested(question):

        limit = extract_top_n(question)

        sorted_data = (
            grouped
            .sort_values(
                ascending=True
            )
            .head(limit)
        )

    else:

        limit = 1

        sorted_data = (
            sorted_data
            .head(1)
        )

    if sorted_data.empty:
        return False

    label = (
        "Average"
        if aggregation == "mean"
        else "Total"
    )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    if top_requested(question):

        title = (
            f"Top {limit} {dimension} by {metric}"
        )

    elif bottom_requested(question):

        title = (
            f"Bottom {limit} {dimension} by {metric}"
        )

    elif is_highest:

        title = (
            f"Highest {metric} by {dimension}"
        )

    else:

        title = (
            f"Lowest {metric} by {dimension}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Build a brand-new dataframe from ONLY the ranking result.
    #
    # This prevents the chart from accidentally showing
    # all regions/categories when the table contains Top N.
    # --------------------------------------------------------

    chart_df = (
        sorted_data
        .rename("Value")
        .reset_index()
    )

    chart_df.columns = [
        dimension,
        "Value"
    ]

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    render_chart(
        chart_df,
        "bar",
        title,
        dimension,
        "Value"
    )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    show_result_table(
        sorted_data,
        dimension,
        f"{label} {metric}"
    )

    # --------------------------------------------------------
    # WINNER
    # --------------------------------------------------------

    winner = sorted_data.index[0]

    winner_value = float(
        sorted_data.iloc[0]
    )

    if limit == 1:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                dimension,
                str(winner)
            )

        with col2:

            st.metric(
                metric,
                format_value(winner_value)
            )

        if is_highest:

            st.success(
                f"{winner} has the highest "
                f"{metric.lower()} at "
                f"{format_value(winner_value)}."
            )

        else:

            st.info(
                f"{winner} has the lowest "
                f"{metric.lower()} at "
                f"{format_value(winner_value)}."
            )

    else:

        if is_highest:

            st.success(
                f"{winner} ranks #1 with "
                f"{format_value(winner_value)}."
            )

        else:

            st.info(
                f"{winner} ranks #1 among the "
                f"lowest-performing groups with "
                f"{format_value(winner_value)}."
            )

    return True


# ============================================================
# GROUPED ANALYSIS
# ============================================================

def show_grouped_analysis(
    df,
    question,
    dimension,
    metric
):

    aggregation = determine_aggregation(
        question
    )

    grouped = aggregate_grouped(
        df,
        dimension,
        metric,
        aggregation
    )

    if grouped.empty:
        return False

    grouped = grouped.sort_values(
        ascending=False
    )

    label = (
        "Average"
        if aggregation == "mean"
        else "Total"
    )

    title = (
        f"{label} {metric} by {dimension}"
    )

    chart_df = (
        grouped
        .rename("Value")
        .reset_index()
    )

    chart_df.columns = [
        dimension,
        "Value"
    ]

    render_chart(
        chart_df,
        "bar",
        title,
        dimension,
        "Value"
    )

    show_result_table(
        grouped,
        dimension,
        f"{label} {metric}"
    )

    return True


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def extract_threshold(question):

    q = normalize(question)

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


def show_threshold(
    df,
    question,
    dimension,
    metric
):

    operator, threshold = (
        extract_threshold(question)
    )

    if operator is None:
        return False

    data = df.copy()

    data[metric] = numeric_series(
        data,
        metric
    )

    data = data.dropna(
        subset=[metric]
    )

    if data.empty:
        return False

    if operator == ">":

        result = data[
            data[metric] > threshold
        ]

    elif operator == "<":

        result = data[
            data[metric] < threshold
        ]

    elif operator == ">=":

        result = data[
            data[metric] >= threshold
        ]

    else:

        result = data[
            data[metric] <= threshold
        ]

    result = result.sort_values(
        metric,
        ascending=False
    )

    st.markdown(
        f"### 📊 {metric} {operator} "
        f"{threshold:,.2f}"
    )

    if result.empty:

        st.info(
            "No matching records were found."
        )

        return True

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    if dimension is not None:

        chart_data = (
            result
            .groupby(dimension)[metric]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        if not chart_data.empty:

            chart_df = (
                chart_data
                .rename("Value")
                .reset_index()
            )

            chart_df.columns = [
                dimension,
                "Value"
            ]

            render_chart(
                chart_df,
                "bar",
                f"{metric} Matching Threshold",
                dimension,
                "Value"
            )

    st.success(
        f"{len(result):,} matching "
        f"record(s) found."
    )

    return True


# ============================================================
# TIME SERIES
# ============================================================

def show_time_series(
    df,
    question,
    metric
):

    date_column = find_date_column(
        df
    )

    if date_column is None:
        return False

    data = df[
        [
            date_column,
            metric
        ]
    ].copy()

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )

    data[metric] = numeric_series(
        data,
        metric
    )

    data = data.dropna(
        subset=[
            date_column,
            metric
        ]
    )

    if data.empty:
        return False

    q = normalize(question)

    # --------------------------------------------------------
    # MONTHLY
    # --------------------------------------------------------

    if any(
        phrase in q
        for phrase in [
            "monthly",
            "by month",
            "per month",
            "month"
        ]
    ):

        data["__period__"] = (
            data[date_column]
            .dt.to_period("M")
            .dt.to_timestamp()
        )

    # --------------------------------------------------------
    # YEARLY
    # --------------------------------------------------------

    elif any(
        phrase in q
        for phrase in [
            "yearly",
            "by year",
            "per year",
            "year"
        ]
    ):

        data["__period__"] = (
            data[date_column]
            .dt.to_period("Y")
            .dt.to_timestamp()
        )

    # --------------------------------------------------------
    # DAILY
    # --------------------------------------------------------

    else:

        data["__period__"] = (
            data[date_column]
            .dt.floor("D")
        )

    grouped = (
        data
        .groupby("__period__")[metric]
        .sum()
        .sort_index()
    )

    if grouped.empty:
        return False

    table = (
        grouped
        .rename(metric)
        .reset_index()
    )

    table.columns = [
        date_column,
        metric
    ]

    chart_df = table.copy()

    render_chart(
        chart_df,
        "line",
        f"{metric} Trend",
        date_column,
        metric
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # TREND INSIGHT
    # --------------------------------------------------------

    if len(grouped) >= 2:

        first = float(
            grouped.iloc[0]
        )

        last = float(
            grouped.iloc[-1]
        )

        if last > first:

            change = last - first

            st.success(
                f"{metric} shows an overall "
                f"upward trend. Difference between "
                f"the first and last periods: "
                f"{format_value(change)}."
            )

        elif last < first:

            change = first - last

            st.info(
                f"{metric} shows an overall "
                f"downward trend. Difference between "
                f"the first and last periods: "
                f"{format_value(change)}."
            )

        else:

            st.info(
                f"{metric} remained stable over "
                f"the analyzed period."
            )

    return True


# ============================================================
# DISTRIBUTION
# ============================================================

def show_distribution(
    df,
    question,
    metric
):

    data = pd.DataFrame({
        metric: numeric_series(
            df,
            metric
        )
    }).dropna()

    if data.empty:
        return False

    render_chart(
        data,
        "histogram",
        f"Distribution of {metric}",
        metric
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Minimum",
            format_value(
                data[metric].min()
            )
        )

    with col2:

        st.metric(
            "Average",
            format_value(
                data[metric].mean()
            )
        )

    with col3:

        st.metric(
            "Maximum",
            format_value(
                data[metric].max()
            )
        )

    return True


# ============================================================
# RELATIONSHIP / CORRELATION
# ============================================================

def show_relationship(
    df,
    question,
    metric
):

    second_metric = (
        find_second_numeric_column(
            df,
            question,
            metric
        )
    )

    if second_metric is None:
        return False

    data = df[
        [
            second_metric,
            metric
        ]
    ].copy()

    data[second_metric] = numeric_series(
        data,
        second_metric
    )

    data[metric] = numeric_series(
        data,
        metric
    )

    data = data.dropna()

    if len(data) < 2:
        return False

    chart_df = data[
        [
            second_metric,
            metric
        ]
    ].copy()

    render_chart(
        chart_df,
        "scatter",
        f"{metric} vs {second_metric}",
        second_metric,
        metric
    )

    correlation = (
        data[
            [
                second_metric,
                metric
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    if pd.isna(correlation):
        return True

    st.metric(
        "Correlation",
        f"{correlation:.2f}"
    )

    absolute_corr = abs(
        correlation
    )

    if absolute_corr >= 0.7:
        strength = "strong"

    elif absolute_corr >= 0.4:
        strength = "moderate"

    else:
        strength = "weak"

    if correlation > 0:
        direction = "positive"

    elif correlation < 0:
        direction = "negative"

    else:
        direction = "no"

    st.info(
        f"The data shows a {strength} "
        f"{direction} relationship between "
        f"{metric} and {second_metric}."
    )

    return True


# ============================================================
# SHARE / PERCENTAGE
# ============================================================

def show_share(
    df,
    question,
    dimension,
    metric
):

    grouped = aggregate_grouped(
        df,
        dimension,
        metric,
        "sum"
    )

    if grouped.empty:
        return False

    # Keep all categories for accurate percentage
    grouped = grouped.sort_values(
        ascending=False
    )

    chart_df = (
        grouped
        .rename("Value")
        .reset_index()
    )

    chart_df.columns = [
        dimension,
        "Value"
    ]

    render_chart(
        chart_df,
        "pie",
        f"{metric} Share by {dimension}",
        dimension,
        "Value"
    )

    total = grouped.sum()

    table = (
        grouped
        .rename("Value")
        .reset_index()
    )

    if total != 0:

        table["Percentage"] = (
            table["Value"]
            / total
            * 100
        ).round(2)

    else:

        table["Percentage"] = 0

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    return True


# ============================================================
# GENERIC FALLBACK
# ============================================================

def generic_visualization(
    df,
    question
):

    nums = numeric_columns(df)

    if not nums:
        return False

    metric = find_metric_column(
        df,
        question
    )

    if metric is None:
        metric = nums[0]

    dimension = find_dimension_column(
        df,
        question
    )

    if dimension is not None:

        return show_grouped_analysis(
            df,
            question,
            dimension,
            metric
        )

    return show_distribution(
        df,
        question,
        metric
    )


# ============================================================
# 8. CONFIGURATION ENGINE
# ============================================================

def detect_visualization(
    question,
    df
):

    if (
        df is None
        or df.empty
        or not question
    ):
        return None

    data = clean_dataframe(
        df
    )

    question = str(
        question
    ).strip()

    metric = find_metric_column(
        data,
        question
    )

    dimension = find_dimension_column(
        data,
        question
    )

    # --------------------------------------------------------
    # DISTRIBUTION
    # --------------------------------------------------------

    if distribution_requested(question):

        if metric:

            return {
                "type": "distribution",
                "metric": metric
            }

    # --------------------------------------------------------
    # RELATIONSHIP
    # --------------------------------------------------------

    if relationship_requested(question):

        if metric:

            second = (
                find_second_numeric_column(
                    data,
                    question,
                    metric
                )
            )

            if second:

                return {
                    "type": "relationship",
                    "metric": metric,
                    "second_metric": second
                }

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    if time_requested(question):

        if (
            metric
            and find_date_column(data)
        ):

            return {
                "type": "time_series",
                "metric": metric
            }

    # --------------------------------------------------------
    # SHARE
    # --------------------------------------------------------

    if share_requested(question):

        if (
            metric
            and dimension
        ):

            return {
                "type": "share",
                "metric": metric,
                "dimension": dimension
            }

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    if threshold_requested(question):

        if metric:

            return {
                "type": "threshold",
                "metric": metric,
                "dimension": dimension
            }

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    if comparison_requested(question):

        if (
            metric
            and dimension
        ):

            return {
                "type": "comparison",
                "metric": metric,
                "dimension": dimension
            }

    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    if (
        highest_requested(question)
        or lowest_requested(question)
        or top_requested(question)
        or bottom_requested(question)
    ):

        if (
            metric
            and dimension
        ):

            if (
                bottom_requested(question)
                or lowest_requested(question)
            ):
                ranking_type = "lowest"

            else:
                ranking_type = "highest"

            return {
                "type": "ranking",
                "metric": metric,
                "dimension": dimension,
                "limit": extract_top_n(question),
                "ranking_type": ranking_type
            }

    # --------------------------------------------------------
    # GROUPED
    # --------------------------------------------------------

    if (
        metric
        and dimension
    ):

        return {
            "type": "grouped",
            "metric": metric,
            "dimension": dimension,
            "aggregation":
                determine_aggregation(question)
        }

    # --------------------------------------------------------
    # FINAL FALLBACK
    # --------------------------------------------------------

    if metric:

        return {
            "type": "distribution",
            "metric": metric
        }

    return None


# ============================================================
# CONFIGURATION RENDERER
# ============================================================

def render_visualization(
    df,
    config
):

    if (
        df is None
        or df.empty
    ):
        return False

    if not config:
        return False

    data = clean_dataframe(
        df
    )

    chart_type = config.get(
        "type"
    )

    metric = config.get(
        "metric"
    )

    dimension = config.get(
        "dimension"
    )

    if (
        metric is not None
        and metric not in data.columns
    ):
        return False

    # --------------------------------------------------------
    # DISTRIBUTION
    # --------------------------------------------------------

    if chart_type == "distribution":

        return show_distribution(
            data,
            "",
            metric
        )

    # --------------------------------------------------------
    # RELATIONSHIP
    # --------------------------------------------------------

    if chart_type == "relationship":

        return show_relationship(
            data,
            "",
            metric
        )

    # --------------------------------------------------------
    # TIME SERIES
    # --------------------------------------------------------

    if chart_type == "time_series":

        return show_time_series(
            data,
            "",
            metric
        )

    # --------------------------------------------------------
    # SHARE
    # --------------------------------------------------------

    if chart_type == "share":

        if not dimension:
            return False

        return show_share(
            data,
            "",
            dimension,
            metric
        )

    # --------------------------------------------------------
    # THRESHOLD
    #
    # Threshold requires the original question, so the main
    # generate_visual_analysis function handles it.
    # --------------------------------------------------------

    if chart_type == "threshold":

        return False

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    if chart_type == "comparison":

        if not dimension:
            return False

        return show_grouped_analysis(
            data,
            "",
            dimension,
            metric
        )

    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    if chart_type == "ranking":

        grouped = aggregate_grouped(
            data,
            dimension,
            metric,
            "mean"
            if config.get("aggregation") == "mean"
            else "sum"
        )

        if grouped.empty:
            return False

        limit = config.get(
            "limit",
            1
        )

        ranking_type = config.get(
            "ranking_type",
            "highest"
        )

        if ranking_type == "lowest":

            result = (
                grouped
                .sort_values(
                    ascending=True
                )
                .head(limit)
            )

            title = (
                f"Bottom {limit} "
                f"{dimension} by {metric}"
                if limit > 1
                else
                f"Lowest {metric} by {dimension}"
            )

        else:

            result = (
                grouped
                .sort_values(
                    ascending=False
                )
                .head(limit)
            )

            title = (
                f"Top {limit} "
                f"{dimension} by {metric}"
                if limit > 1
                else
                f"Highest {metric} by {dimension}"
            )

        chart_df = (
            result
            .rename("Value")
            .reset_index()
        )

        chart_df.columns = [
            dimension,
            "Value"
        ]

        render_chart(
            chart_df,
            "bar",
            title,
            dimension,
            "Value"
        )

        show_result_table(
            result,
            dimension,
            metric
        )

        return True

    # --------------------------------------------------------
    # GROUPED
    # --------------------------------------------------------

    if chart_type == "grouped":

        if (
            not dimension
            or not metric
        ):
            return False

        return show_grouped_analysis(
            data,
            "",
            dimension,
            metric
        )

    return False


# ============================================================
# MAIN VISUALIZATION ENGINE
# ============================================================

def generate_visual_analysis(
    df,
    question=None
):
    """
    Main 8-stage visualization engine.

    Returns:
        True  -> visualization generated
        False -> no suitable visualization
    """

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if (
        df is None
        or df.empty
        or not question
    ):
        return False

    data = clean_dataframe(
        df
    )

    question = str(
        question
    ).strip()

    if not question:
        return False

    # --------------------------------------------------------
    # DETECT METRIC
    # --------------------------------------------------------

    metric = find_metric_column(
        data,
        question
    )

    if metric is None:

        nums = numeric_columns(
            data
        )

        if nums:
            metric = nums[0]

        else:

            st.info(
                "No numeric metric was found "
                "for this question."
            )

            return False

    # --------------------------------------------------------
    # DETECT DIMENSION
    # --------------------------------------------------------

    dimension = find_dimension_column(
        data,
        question
    )

    # ========================================================
    # STAGE 1
    # DISTRIBUTION
    # ========================================================

    if distribution_requested(
        question
    ):

        return show_distribution(
            data,
            question,
            metric
        )

    # ========================================================
    # STAGE 2
    # RELATIONSHIP
    # ========================================================

    if relationship_requested(
        question
    ):

        if show_relationship(
            data,
            question,
            metric
        ):
            return True

    # ========================================================
    # STAGE 3
    # TIME TREND
    # ========================================================

    if time_requested(
        question
    ):

        if show_time_series(
            data,
            question,
            metric
        ):
            return True

    # ========================================================
    # STAGE 4
    # SHARE
    # ========================================================

    if (
        share_requested(question)
        and dimension is not None
    ):

        if show_share(
            data,
            question,
            dimension,
            metric
        ):
            return True

    # ========================================================
    # STAGE 5
    # THRESHOLD
    # ========================================================

    if threshold_requested(
        question
    ):

        if show_threshold(
            data,
            question,
            dimension,
            metric
        ):
            return True

    # ========================================================
    # STAGE 6
    # COMPARISON
    # ========================================================

    if (
        comparison_requested(question)
        and dimension is not None
    ):

        if show_comparison(
            data,
            question,
            dimension,
            metric
        ):
            return True

    # ========================================================
    # STAGE 7
    # RANKING
    # ========================================================

    if (
        highest_requested(question)
        or lowest_requested(question)
        or top_requested(question)
        or bottom_requested(question)
    ):

        if dimension is not None:

            if show_ranking(
                data,
                question,
                dimension,
                metric
            ):
                return True

    # ========================================================
    # STAGE 8
    # GROUPED ANALYSIS
    # ========================================================

    if dimension is not None:

        if (
            average_requested(question)
            or total_requested(question)
        ):

            if show_grouped_analysis(
                data,
                question,
                dimension,
                metric
            ):
                return True

    # ========================================================
    # GENERIC FALLBACK
    # ========================================================

    return generic_visualization(
        data,
        question
    )


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def visualize(
    df,
    question
):

    return generate_visual_analysis(
        df,
        question
    )


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "detect_visualization",
    "render_visualization",
    "generate_visual_analysis",
    "visualize",
    "find_column",
    "find_metric_column",
    "find_dimension_column",
    "extract_top_n"
]