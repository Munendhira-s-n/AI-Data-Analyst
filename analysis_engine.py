import pandas as pd
import re

from question_parser import parse_question


# =========================================================
# TEXT HELPERS
# =========================================================

def normalize(text):
    """
    Normalize text for matching.

    Example:
        'Customer Segment' -> 'customersegment'
        'Performance_Score' -> 'performancescore'
    """
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(text).lower()
    )


# =========================================================
# FIND NUMERIC COLUMN
# =========================================================

def find_numeric_column(df, question, metric_hint=None):
    """
    Find the numeric column most relevant to the question.
    """

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if not numeric_columns:
        return None

    question_lower = question.lower()
    question_normalized = normalize(question)

    # -----------------------------------------------------
    # Metric hint from parser
    # -----------------------------------------------------

    if metric_hint:

        hint_aliases = {

            "salary": [
                "salary",
                "pay",
                "wage",
                "compensation",
                "earnings"
            ],

            "performance": [
                "performance",
                "performancescore",
                "score",
                "rating"
            ],

            "experience": [
                "experience",
                "years",
                "tenure"
            ],

            "sales": [
                "sales",
                "sale",
                "revenue",
                "turnover",
                "income"
            ],

            "profit": [
                "profit",
                "earnings",
                "netprofit"
            ],

            "quantity": [
                "quantity",
                "units",
                "unit",
                "volume"
            ]
        }

        for column in numeric_columns:

            normalized_column = normalize(column)

            for word in hint_aliases.get(
                metric_hint,
                []
            ):

                if normalize(word) in normalized_column:
                    return column

    # -----------------------------------------------------
    # Direct column matching
    # -----------------------------------------------------

    for column in numeric_columns:

        normalized_column = normalize(column)

        if normalized_column in question_normalized:
            return column

    # -----------------------------------------------------
    # Semantic aliases
    # -----------------------------------------------------

    aliases = {

        "sales": [
            "sales",
            "sale",
            "revenue",
            "income",
            "turnover"
        ],

        "revenue": [
            "revenue",
            "sales",
            "income",
            "turnover"
        ],

        "profit": [
            "profit",
            "earnings",
            "netprofit"
        ],

        "salary": [
            "salary",
            "pay",
            "income",
            "wage",
            "compensation"
        ],

        "performance": [
            "performance",
            "score",
            "rating",
            "points"
        ],

        "experience": [
            "experience",
            "years",
            "tenure"
        ],

        "quantity": [
            "quantity",
            "units",
            "unit",
            "volume"
        ],

        "amount": [
            "amount",
            "value",
            "price",
            "cost"
        ],

        "cost": [
            "cost",
            "expense",
            "spend",
            "expenditure"
        ],

        "spend": [
            "spend",
            "cost",
            "expense"
        ]
    }

    for key, words in aliases.items():

        if key in question_lower:

            for column in numeric_columns:

                normalized_column = normalize(column)

                for word in words:

                    if normalize(word) in normalized_column:
                        return column

    # -----------------------------------------------------
    # Only one numeric column
    # -----------------------------------------------------

    if len(numeric_columns) == 1:
        return numeric_columns[0]

    # -----------------------------------------------------
    # Business metric priority
    # -----------------------------------------------------

    priority = [
        "sales",
        "revenue",
        "profit",
        "salary",
        "amount",
        "quantity",
        "performance",
        "experience",
        "score",
        "cost",
        "price",
        "spend",
        "value"
    ]

    for priority_word in priority:

        for column in numeric_columns:

            if priority_word in normalize(column):
                return column

    return numeric_columns[0]


# =========================================================
# FIND DIMENSION COLUMN
# =========================================================

def find_dimension_column(df, question):
    """
    Find the categorical dimension relevant to the question.
    """

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if not categorical_columns:
        return None

    question_normalized = normalize(question)
    question_lower = question.lower()

    # -----------------------------------------------------
    # Explicit dimension words
    # -----------------------------------------------------

    aliases = {

        "product": [
            "product",
            "item",
            "model",
            "sku"
        ],

        "region": [
            "region",
            "area",
            "location",
            "city",
            "state",
            "country"
        ],

        "customer": [
            "customer",
            "client",
            "buyer"
        ],

        "employee": [
            "employee",
            "employees",
            "staff",
            "worker",
            "person",
            "people",
            "who"
        ],

        "department": [
            "department",
            "departments",
            "team",
            "division"
        ],

        "category": [
            "category",
            "type",
            "class"
        ],

        "campaign": [
            "campaign",
            "advertisement",
            "ad"
        ],

        "channel": [
            "channel",
            "source",
            "platform"
        ],

        "segment": [
            "segment",
            "group"
        ]
    }

    # -----------------------------------------------------
    # Direct column matching
    # -----------------------------------------------------

    for column in categorical_columns:

        column_normalized = normalize(column)

        if column_normalized in question_normalized:
            return column

    # -----------------------------------------------------
    # Semantic matching
    # -----------------------------------------------------

    for key, words in aliases.items():

        if any(
            word in question_lower
            for word in words
        ):

            for column in categorical_columns:

                normalized_column = normalize(column)

                for word in words:

                    if normalize(word) in normalized_column:
                        return column

    # -----------------------------------------------------
    # Detect category values
    # -----------------------------------------------------

    best_column = None
    best_match_count = 0

    for column in categorical_columns:

        match_count = 0

        for value in df[column].dropna().unique():

            value_normalized = normalize(value)

            if (
                value_normalized
                and value_normalized in question_normalized
            ):
                match_count += 1

        if match_count > best_match_count:

            best_match_count = match_count
            best_column = column

    if best_column is not None:
        return best_column

    # -----------------------------------------------------
    # If only one categorical column
    # -----------------------------------------------------

    if len(categorical_columns) == 1:
        return categorical_columns[0]

    return None


# =========================================================
# FIND EMPLOYEE / PERSON COLUMN
# =========================================================

def find_person_column(df):
    """
    Detect employee/person/name column.
    """

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    priority = [
        "employee",
        "employee_name",
        "employeename",
        "name",
        "staff",
        "person",
        "worker"
    ]

    # Exact / normalized priority
    for priority_word in priority:

        for column in categorical_columns:

            if normalize(column) == normalize(
                priority_word
            ):
                return column

    # Partial match
    for column in categorical_columns:

        normalized_column = normalize(column)

        for priority_word in priority:

            if normalize(priority_word) in normalized_column:
                return column

    return None


# =========================================================
# FIND CATEGORY VALUES
# =========================================================

def find_category_values(df, dimension, question):
    """
    Find actual category values mentioned in question.
    """

    question_normalized = normalize(question)

    matches = []

    for value in df[dimension].dropna().unique():

        value_normalized = normalize(value)

        if (
            value_normalized
            and value_normalized in question_normalized
        ):
            matches.append(value)

    return matches


# =========================================================
# FIND PERSON NAME
# =========================================================

def find_person_name(df, question):
    """
    Detect a person's name mentioned in the question.

    Example:
        What is Asha's salary?

    Returns:
        Asha
    """

    person_column = find_person_column(df)

    if not person_column:
        return None, None

    question_normalized = normalize(question)

    for value in df[person_column].dropna().unique():

        value_normalized = normalize(value)

        if (
            value_normalized
            and value_normalized in question_normalized
        ):
            return person_column, value

    return None, None


# =========================================================
# FIND NUMBER IN QUESTION
# =========================================================

def find_number(question):
    """
    Extract numeric threshold.

    Example:
        employees earn more than 70000

    Returns:
        70000
    """

    match = re.search(
        r"\b(\d+(?:\.\d+)?)\b",
        question
    )

    if match:
        return float(match.group(1))

    return None


# =========================================================
# FORMAT NUMBER
# =========================================================

def format_number(value):
    """
    Format numeric values cleanly.
    """

    if pd.isna(value):
        return "N/A"

    if float(value).is_integer():
        return f"{int(value):,}"

    return f"{value:,.2f}"


# =========================================================
# MAIN ANALYSIS ENGINE
# =========================================================

def analyze_dataset(df, question):
    """
    Main Python analysis engine.

    Python performs ALL calculations.

    Ollama is only responsible for explaining
    the verified result.
    """

    original_question = question

    question = question.lower().strip()

    intent = parse_question(question)

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    metric_hint = intent.get(
        "metric_hint"
    )

    # =====================================================
    # 1. PERSON-SPECIFIC LOOKUP
    # =====================================================

    person_column, person_name = find_person_name(
        df,
        question
    )

    if person_column and person_name:

        metric = find_numeric_column(
            df,
            question,
            metric_hint
        )

        if metric:

            person_rows = df[
                df[person_column] == person_name
            ]

            if not person_rows.empty:

                # If multiple records exist for person,
                # return all values rather than inventing.
                if len(person_rows) == 1:

                    value = person_rows.iloc[0][metric]

                    return f"""
Verified calculation:

Question:
{original_question}

Person:
{person_name}

Metric:
{metric}

Value:
{format_number(value)}
"""

                values = person_rows[metric]

                return f"""
Verified calculation:

Question:
{original_question}

Person:
{person_name}

Metric:
{metric}

Values:
{values.to_list()}

Records:
{len(person_rows)}
"""

    # =====================================================
    # 2. THRESHOLD / FILTER QUESTIONS
    # =====================================================

    threshold_patterns = [
        "more than",
        "greater than",
        "above",
        "over",
        "higher than",
        "less than",
        "lower than",
        "below",
        "under"
    ]

    threshold_word = None

    for word in threshold_patterns:

        if word in question:

            threshold_word = word
            break

    if threshold_word:

        threshold = find_number(question)

        metric = find_numeric_column(
            df,
            question,
            metric_hint
        )

        if (
            threshold is not None
            and metric is not None
        ):

            if threshold_word in [
                "more than",
                "greater than",
                "above",
                "over",
                "higher than"
            ]:

                filtered = df[
                    df[metric] > threshold
                ]

                operator = ">"

            else:

                filtered = df[
                    df[metric] < threshold
                ]

                operator = "<"

            person_column = find_person_column(df)

            if person_column:

                result_columns = [
                    person_column,
                    metric
                ]

                results = filtered[
                    result_columns
                ]

                return f"""
Verified calculation:

Question:
{original_question}

Metric:
{metric}

Condition:
{metric} {operator} {format_number(threshold)}

Matching Employees:
{len(results)}

Results:

{results.to_string(index=False)}
"""

            return f"""
Verified calculation:

Question:
{original_question}

Metric:
{metric}

Condition:
{metric} {operator} {format_number(threshold)}

Matching Rows:
{len(filtered)}

Results:

{filtered.to_string(index=False)}
"""

    # =====================================================
    # 3. COMPARISON
    # =====================================================

    if intent.get(
        "comparison",
        False
    ):

        dimension = find_dimension_column(
            df,
            question
        )

        if dimension:

            categories = find_category_values(
                df,
                dimension,
                question
            )

            if len(categories) >= 2:

                categories = categories[:2]

                metric = find_numeric_column(
                    df,
                    question,
                    metric_hint
                )

                if metric:

                    filtered_df = df[
                        df[dimension].isin(
                            categories
                        )
                    ]

                    # -------------------------------------------------
                    # Determine aggregation
                    # -------------------------------------------------

                    explicit_average = (
                        intent.get(
                            "aggregation"
                        ) == "mean"
                        or "average" in question
                        or "mean" in question
                        or "avg" in question
                    )

                    explicit_total = (
                        intent.get(
                            "aggregation"
                        ) == "sum"
                        or "total" in question
                        or "sum" in question
                    )

                    # Default comparison = average
                    if (
                        explicit_average
                        or not explicit_total
                    ):

                        comparison = (
                            filtered_df
                            .groupby(dimension)[metric]
                            .mean()
                            .reindex(categories)
                        )

                        difference = (
                            comparison.iloc[0]
                            - comparison.iloc[1]
                        )

                        higher_category = (
                            comparison.idxmax()
                        )

                        lower_category = (
                            comparison.idxmin()
                        )

                        return f"""
Verified calculation:

Question:
{original_question}

Metric:
{metric}

Dimension:
{dimension}

Categories:
{categories}

Comparison:
{comparison.to_string()}

Aggregation:
Average

Highest:
{higher_category}

Highest Average:
{format_number(comparison.max())}

Lowest:
{lower_category}

Lowest Average:
{format_number(comparison.min())}

Difference:
{format_number(abs(difference))}
"""

                    # -------------------------------------------------
                    # Total comparison
                    # -------------------------------------------------

                    comparison = (
                        filtered_df
                        .groupby(dimension)[metric]
                        .sum()
                        .reindex(categories)
                    )

                    difference = (
                        comparison.iloc[0]
                        - comparison.iloc[1]
                    )

                    return f"""
Verified calculation:

Question:
{original_question}

Metric:
{metric}

Dimension:
{dimension}

Categories:
{categories}

Comparison:
{comparison.to_string()}

Aggregation:
Total

Highest:
{comparison.idxmax()}

Highest Total:
{format_number(comparison.max())}

Lowest:
{comparison.idxmin()}

Lowest Total:
{format_number(comparison.min())}

Difference:
{format_number(abs(difference))}
"""

    # =====================================================
    # 4. HIGHEST / TOP
    # =====================================================

    if intent.get("ranking") == "highest":

        metric = find_numeric_column(
            df,
            question,
            metric_hint
        )

        # -------------------------------------------------
        # Employee-level question
        # -------------------------------------------------

        person_question = (
            "who" in question
            or "employee" in question
            or "person" in question
            or "staff" in question
        )

        if person_question:

            person_column = find_person_column(df)

            if (
                metric
                and person_column
            ):

                highest_row = df.loc[
                    df[metric].idxmax()
                ]

                return f"""
Verified calculation:

Question:
{original_question}

Metric:
{metric}

Grouped by:
{person_column}

Highest:
{highest_row[person_column]}

Highest Value:
{format_number(highest_row[metric])}
"""

        # -------------------------------------------------
        # Department / dimension ranking
        # -------------------------------------------------

        dimension = find_dimension_column(
            df,
            question
        )

        if metric and dimension:

            use_average = (
                intent.get(
                    "aggregation"
                ) == "mean"
                or "average" in question
                or "mean" in question
                or "avg" in question
            )

            if use_average:

                grouped = (
                    df.groupby(dimension)[metric]
                    .mean()
                    .sort_values(
                        ascending=False
                    )
                )

                aggregation_name = "Average"

            else:

                grouped = (
                    df.groupby(dimension)[metric]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )
                aggregation_name = "Total"

            # -------------------------------------------------
            # Top N
            # -------------------------------------------------

            if intent.get("limit"):

                top_n = grouped.head(
                    intent["limit"]
                )

                return f"""
Verified calculation:

Question:
{original_question}

Metric:
{metric}

Grouped by:
{dimension}

Aggregation:
{aggregation_name}

Top {intent["limit"]}:

{top_n.to_string()}

Highest:
{grouped.index[0]}

Highest {aggregation_name}:
{format_number(grouped.iloc[0])}
"""

            return f"""
Verified calculation:

Question:
{original_question}

Metric:
{metric}

Grouped by:
{dimension}

Aggregation:
{aggregation_name}

Values:

{grouped.to_string()}

Highest:
{grouped.index[0]}

Highest {aggregation_name}:
{format_number(grouped.iloc[0])}
"""

    # =====================================================
    # 5. LOWEST
    # =====================================================

    if intent.get("ranking") == "lowest":

        metric = find_numeric_column(
            df,
            question,
            metric_hint
        )

        # -------------------------------------------------
        # Employee-level
        # -------------------------------------------------

        person_question = (
            "who" in question
            or "employee" in question
            or "person" in question
            or "staff" in question
        )

        if person_question:

            person_column = find_person_column(
                df
            )

            if (
                metric
                and person_column
            ):

                lowest_row = df.loc[
                    df[metric].idxmin()
                ]

                return f"""
Verified calculation:

Question:
{original_question}

Metric:
{metric}

Grouped by:
{person_column}

Lowest:
{lowest_row[person_column]}

Lowest Value:
{format_number(lowest_row[metric])}
"""

        # -------------------------------------------------
        # Dimension-level
        # -------------------------------------------------

        dimension = find_dimension_column(
            df,
            question
        )

        if metric and dimension:

            use_average = (
                intent.get(
                    "aggregation"
                ) == "mean"
                or "average" in question
                or "mean" in question
                or "avg" in question
            )

            if use_average:

                grouped = (
                    df.groupby(dimension)[metric]
                    .mean()
                    .sort_values(
                        ascending=True
                    )
                )

                aggregation_name = "Average"

            else:

                grouped = (
                    df.groupby(dimension)[metric]
                    .sum()
                    .sort_values(
                        ascending=True
                    )
                )

                aggregation_name = "Total"

            return f"""
Verified calculation:

Question:
{original_question}

Metric:
{metric}

Grouped by:
{dimension}

Aggregation:
{aggregation_name}

Values:

{grouped.to_string()}

Lowest:
{grouped.index[0]}

Lowest {aggregation_name}:
{format_number(grouped.iloc[0])}
"""

    # =====================================================
    # 6. AVERAGE
    # =====================================================

    if (
        "average" in question
        or "mean" in question
        or "avg" in question
    ):

        metric = find_numeric_column(
            df,
            question,
            metric_hint
        )

        if metric:

            average = df[metric].mean()

            return f"""
Verified calculation:

Question:
{original_question}

Metric:
{metric}

Average:
{format_number(average)}
"""

    # =====================================================
    # 7. TOTAL
    # =====================================================

    if (
        "total" in question
        or "sum" in question
    ):

        metric = find_numeric_column(
            df,
            question,
            metric_hint
        )

        if metric:

            total = df[metric].sum()

            return f"""
Verified calculation:

Question:
{original_question}

Metric:
{metric}

Total:
{format_number(total)}
"""

    # =====================================================
    # 8. QUANTITY / UNITS
    # =====================================================

    if (
        "quantity" in question
        or "units" in question
        or "unit" in question
    ):

        metric = find_numeric_column(
            df,
            question,
            metric_hint
        )

        dimension = find_dimension_column(
            df,
            question
        )

        if metric and dimension:

            grouped = (
                df.groupby(dimension)[metric]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            return f"""
Verified calculation:

Question:
{original_question}

Metric:
{metric}

Grouped by:
{dimension}

Aggregation:
Total

Values:

{grouped.to_string()}

Highest:
{grouped.index[0]}

Highest Quantity:
{format_number(grouped.iloc[0])}
"""

    # =====================================================
    # 9. PERCENTAGE
    # =====================================================

    if (
        "percentage" in question
        or "percent" in question
        or "%" in question
    ):

        metric = find_numeric_column(
            df,
            question,
            metric_hint
        )

        if metric:

            total = df[metric].sum()

            dimension = find_dimension_column(
                df,
                question
            )

            if dimension:

                grouped = (
                    df.groupby(dimension)[metric]
                    .sum()
                )

                percentages = (
                    grouped / total * 100
                ).sort_values(
                    ascending=False
                )

                return f"""
Verified calculation:

Question:
{original_question}

Metric:
{metric}

Total:
{format_number(total)}

Percentage by {dimension}:

{percentages.to_string()}
"""

    # =====================================================
    # 10. DATASET INFORMATION
    # =====================================================

    if (
        "columns" in question
        or "dataset" in question
        or "fields" in question
    ):

        return f"""
Dataset information:

Rows:
{len(df)}

Columns:
{len(df.columns)}

Column names:
{list(df.columns)}

Numeric columns:
{numeric_columns}

Categorical columns:
{categorical_columns}
"""

    # =====================================================
    # 11. FALLBACK
    # =====================================================

    return f"""
The analysis engine could not identify a specialized
calculation for this question.

Question:
{original_question}

Available columns:
{list(df.columns)}

Numeric columns:
{numeric_columns}

Categorical columns:
{categorical_columns}
"""