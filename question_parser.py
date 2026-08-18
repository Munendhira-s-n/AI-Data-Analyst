import re


# =========================================================
# TEXT HELPERS
# =========================================================

def _clean_text(value):
    """Normalize user question text."""
    value = str(value or "").lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _contains_any(text, words):
    """Return True when any complete word/phrase is present."""
    text = _clean_text(text)

    for word in words:
        word = _clean_text(word)

        if not word:
            continue

        if re.search(
            r"(?<!\w)" + re.escape(word) + r"(?!\w)",
            text
        ):
            return True

    return False


# =========================================================
# AGGREGATION DETECTION
# =========================================================

def _detect_aggregation(question):

    q = _clean_text(question)

    if _contains_any(
        q,
        [
            "average",
            "avg",
            "mean",
            "on average",
        ]
    ):
        return "mean"

    if _contains_any(
        q,
        [
            "total",
            "sum",
            "overall",
        ]
    ):
        return "sum"

    if _contains_any(
        q,
        [
            "count",
            "number of",
            "how many",
            "no of",
            "number",
        ]
    ):
        return "count"

    if _contains_any(
        q,
        [
            "median",
        ]
    ):
        return "median"

    if _contains_any(
        q,
        [
            "minimum",
            "min",
        ]
    ):
        return "min"

    if _contains_any(
        q,
        [
            "maximum",
            "max",
        ]
    ):
        return "max"

    return None


# =========================================================
# RANKING DETECTION
# =========================================================

def _detect_ranking(question):

    q = _clean_text(question)

    if _contains_any(
        q,
        [
            "highest",
            "top",
            "most",
            "maximum",
            "max",
            "best",
            "largest",
            "greatest",
        ]
    ):
        return "highest"

    if _contains_any(
        q,
        [
            "lowest",
            "least",
            "minimum",
            "min",
            "worst",
            "smallest",
            "lowest",
        ]
    ):
        return "lowest"

    return None


# =========================================================
# TOP N DETECTION
# =========================================================

def _detect_limit(question):

    q = _clean_text(question)

    patterns = [
        r"\btop\s+(\d+)\b",
        r"\bfirst\s+(\d+)\b",
        r"\bbest\s+(\d+)\b",
        r"\bhighest\s+(\d+)\b",
        r"\blowest\s+(\d+)\b",
        r"\bbottom\s+(\d+)\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            q
        )

        if match:
            return int(
                match.group(1)
            )

    return None


# =========================================================
# COMPARISON DETECTION
# =========================================================

def _detect_comparison(question):

    q = _clean_text(question)

    comparison_patterns = [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\b",
        r"\bvs\.\b",
        r"\bdifference between\b",
        r"\bcompare between\b",
        r"\bcompared with\b",
        r"\bcompared to\b",
        r"\bhigher than\b",
        r"\blower than\b",
        r"\bmore than\b",
        r"\bless than\b",
        r"\bbetter than\b",
        r"\bworse than\b",
        r"\bdifferent between\b",
    ]

    return any(
        re.search(
            pattern,
            q
        )
        for pattern in comparison_patterns
    )


# =========================================================
# ANALYSIS TYPE DETECTION
# =========================================================

def _detect_analysis_type(question):

    q = _clean_text(question)

    # -----------------------------------------------------
    # TREND
    # -----------------------------------------------------

    if _contains_any(
        q,
        [
            "trend",
            "trends",
            "over time",
            "time trend",
            "monthly trend",
            "yearly trend",
            "weekly trend",
            "daily trend",
            "month by month",
            "year by year",
            "day by day",
            "growth over time",
            "change over time",
        ]
    ):
        return "trend"

    # -----------------------------------------------------
    # DISTRIBUTION
    # -----------------------------------------------------

    if _contains_any(
        q,
        [
            "distribution",
            "spread",
            "frequency distribution",
            "how are",
            "range of",
            "dispersion",
            "histogram",
        ]
    ):
        return "distribution"

    # -----------------------------------------------------
    # CORRELATION / RELATIONSHIP
    # -----------------------------------------------------

    if _contains_any(
        q,
        [
            "correlation",
            "relationship between",
            "relation between",
            "relationship of",
            "related to",
            "correlated",
            "correlate",
            "association between",
        ]
    ):
        return "relationship"

    # -----------------------------------------------------
    # SHARE / PERCENTAGE
    # -----------------------------------------------------

    if _contains_any(
        q,
        [
            "share",
            "percentage",
            "percent",
            "%",
            "proportion",
            "contribution",
            "contributes",
            "portion",
        ]
    ):
        return "share"

    # -----------------------------------------------------
    # COMPARISON
    # -----------------------------------------------------

    if _detect_comparison(q):
        return "comparison"

    # -----------------------------------------------------
    # RANKING
    # -----------------------------------------------------

    if _detect_ranking(q):
        return "ranking"

    # -----------------------------------------------------
    # AGGREGATION
    # -----------------------------------------------------

    aggregation = _detect_aggregation(q)

    if aggregation:
        return "aggregation"

    return "general"


# =========================================================
# METRIC DETECTION
# =========================================================

def _detect_metric(question):

    q = _clean_text(question)

    # -----------------------------------------------------
    # Salary
    # -----------------------------------------------------

    salary_words = [
        "salary",
        "salaries",
        "pay",
        "paid",
        "pays",
        "paying",
        "wage",
        "wages",
        "compensation",
        "earnings",
        "income",
    ]

    # -----------------------------------------------------
    # Performance
    # -----------------------------------------------------

    performance_words = [
        "performance",
        "performance score",
        "performing",
        "rating",
        "ratings",
        "score",
        "scores",
    ]

    # -----------------------------------------------------
    # Experience
    # -----------------------------------------------------

    experience_words = [
        "experience",
        "experienced",
        "years experience",
        "years",
        "tenure",
    ]

    # -----------------------------------------------------
    # Sales
    # -----------------------------------------------------

    sales_words = [
        "sales",
        "sale",
        "revenue",
        "turnover",
    ]

    # -----------------------------------------------------
    # Profit
    # -----------------------------------------------------

    profit_words = [
        "profit",
        "profits",
        "profitability",
        "margin",
        "margins",
    ]

    # -----------------------------------------------------
    # Quantity
    # -----------------------------------------------------

    quantity_words = [
        "quantity",
        "quantities",
        "units",
        "unit",
        "volume",
    ]

    # -----------------------------------------------------
    # Cost
    # -----------------------------------------------------

    cost_words = [
        "cost",
        "costs",
        "expense",
        "expenses",
        "spending",
        "spend",
    ]

    # -----------------------------------------------------
    # Price
    # -----------------------------------------------------

    price_words = [
        "price",
        "prices",
        "unit price",
        "selling price",
    ]

    # -----------------------------------------------------
    # Age
    # -----------------------------------------------------

    age_words = [
        "age",
        "ages",
    ]

    # -----------------------------------------------------
    # Ordered detection
    # -----------------------------------------------------

    metric_groups = [
        ("salary", salary_words),
        ("performance", performance_words),
        ("experience", experience_words),
        ("sales", sales_words),
        ("profit", profit_words),
        ("quantity", quantity_words),
        ("cost", cost_words),
        ("price", price_words),
        ("age", age_words),
    ]

    for metric, words in metric_groups:

        if _contains_any(
            q,
            words
        ):
            return metric

    return None


# =========================================================
# DIMENSION HINT DETECTION
# =========================================================

def _detect_dimension(question):

    q = _clean_text(question)

    dimensions = [
        (
            "department",
            [
                "department",
                "dept",
                "team",
                "division",
            ]
        ),

        (
            "region",
            [
                "region",
                "area",
                "territory",
                "zone",
            ]
        ),

        (
            "location",
            [
                "location",
                "city",
                "state",
                "country",
                "place",
            ]
        ),

        (
            "product",
            [
                "product",
                "products",
                "item",
                "items",
            ]
        ),

        (
            "category",
            [
                "category",
                "categories",
                "type",
                "class",
            ]
        ),

        (
            "segment",
            [
                "segment",
                "customer segment",
                "market segment",
            ]
        ),

        (
            "customer",
            [
                "customer",
                "customers",
                "client",
                "clients",
            ]
        ),

        (
            "employee",
            [
                "employee",
                "employees",
                "staff",
                "person",
                "people",
            ]
        ),

        (
            "month",
            [
                "month",
                "monthly",
            ]
        ),

        (
            "year",
            [
                "year",
                "yearly",
                "annual",
                "annually",
            ]
        ),

        (
            "quarter",
            [
                "quarter",
                "quarterly",
            ]
        ),

        (
            "day",
            [
                "day",
                "daily",
            ]
        ),
    ]

    for dimension, words in dimensions:

        if _contains_any(
            q,
            words
        ):
            return dimension

    return None


# =========================================================
# TIME ANALYSIS DETECTION
# =========================================================

def _detect_time_granularity(question):

    q = _clean_text(question)

    if _contains_any(
        q,
        [
            "daily",
            "day by day",
            "per day",
            "each day",
            "by day",
        ]
    ):
        return "day"

    if _contains_any(
        q,
        [
            "weekly",
            "week by week",
            "per week",
            "each week",
            "by week",
        ]
    ):
        return "week"

    if _contains_any(
        q,
        [
            "monthly",
            "month by month",
            "per month",
            "each month",
            "by month",
        ]
    ):
        return "month"

    if _contains_any(
        q,
        [
            "quarterly",
            "quarter by quarter",
            "per quarter",
            "each quarter",
            "by quarter",
        ]
    ):
        return "quarter"

    if _contains_any(
        q,
        [
            "yearly",
            "year by year",
            "per year",
            "each year",
            "annual",
            "annually",
            "by year",
        ]
    ):
        return "year"

    return None


# =========================================================
# DISTRIBUTION DETECTION
# =========================================================

def _detect_distribution(question):

    q = _clean_text(question)

    return _contains_any(
        q,
        [
            "distribution",
            "spread",
            "frequency",
            "histogram",
            "dispersion",
            "range",
        ]
    )


# =========================================================
# RELATIONSHIP DETECTION
# =========================================================

def _detect_relationship(question):

    q = _clean_text(question)

    return _contains_any(
        q,
        [
            "correlation",
            "relationship between",
            "relation between",
            "relationship of",
            "related to",
            "correlated",
            "correlate",
            "association between",
        ]
    )


# =========================================================
# SHARE DETECTION
# =========================================================

def _detect_share(question):

    q = _clean_text(question)

    return _contains_any(
        q,
        [
            "share",
            "percentage",
            "percent",
            "%",
            "proportion",
            "contribution",
            "portion",
        ]
    )


# =========================================================
# FILTER DETECTION
# =========================================================

def _detect_filter(question):

    q = _clean_text(question)

    patterns = [

        (
            r"(?:greater than|more than|above|over|exceeding)"
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
        ),

        (
            r">=\s*\$?\s*([\d,]+(?:\.\d+)?)",
            ">="
        ),

        (
            r"<=\s*\$?\s*([\d,]+(?:\.\d+)?)",
            "<="
        ),

        (
            r">\s*\$?\s*([\d,]+(?:\.\d+)?)",
            ">"
        ),

        (
            r"<\s*\$?\s*([\d,]+(?:\.\d+)?)",
            "<"
        ),
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
# EMPLOYEE / PERSON LOOKUP
# =========================================================

def _detect_person_lookup(question):

    q = _clean_text(question)

    patterns = [
        r"\bwhat is .+['’]s",
        r"\bwhat is the salary of",
        r"\bwhat is the performance of",
        r"\bwhat is the experience of",
        r"\bwhat is the age of",
        r"\bhow much does .+ earn",
        r"\bhow much does .+ make",
        r"\bhow much experience does",
        r"\bhow old is",
        r"\btell me about .+",
    ]

    return any(
        re.search(
            pattern,
            q
        )
        for pattern in patterns
    )


# =========================================================
# NATURAL LANGUAGE RANKING
# =========================================================

def _apply_natural_language_rules(
    question,
    metric_hint,
    ranking
):

    q = _clean_text(question)

    # -----------------------------------------------------
    # Salary
    # -----------------------------------------------------

    if _contains_any(
        q,
        [
            "pays the most",
            "pay the most",
            "pays the highest",
            "pay the highest",
            "pays more",
            "paid the most",
            "earns the most",
        ]
    ):

        metric_hint = "salary"
        ranking = "highest"

    if _contains_any(
        q,
        [
            "pays the least",
            "pay the least",
            "pays the lowest",
            "paid the least",
            "earns the least",
        ]
    ):

        metric_hint = "salary"
        ranking = "lowest"

    # -----------------------------------------------------
    # Performance
    # -----------------------------------------------------

    if _contains_any(
        q,
        [
            "best performing",
            "highest performing",
            "performs best",
            "perform best",
        ]
    ):

        metric_hint = "performance"
        ranking = "highest"

    if _contains_any(
        q,
        [
            "worst performing",
            "lowest performing",
            "performs worst",
            "perform worst",
        ]
    ):

        metric_hint = "performance"
        ranking = "lowest"

    # -----------------------------------------------------
    # Experience
    # -----------------------------------------------------

    if _contains_any(
        q,
        [
            "most experienced",
            "highest experience",
            "more experienced",
        ]
    ):

        metric_hint = "experience"
        ranking = "highest"

    if _contains_any(
        q,
        [
            "least experienced",
            "lowest experience",
            "less experienced",
        ]
    ):

        metric_hint = "experience"
        ranking = "lowest"

    return metric_hint, ranking


# =========================================================
# MULTIPLE INTENTS
# =========================================================

def _build_multiple_intents(
    question,
    aggregation,
    ranking,
    metric_hint,
    comparison,
    analysis_type,
    dimension_hint,
    limit
):

    intents = []

    # -----------------------------------------------------
    # Comparison
    # -----------------------------------------------------

    if comparison:

        intents.append({
            "type": "comparison",
            "metric": metric_hint,
            "dimension": dimension_hint,
            "aggregation": aggregation or "sum"
        })

    # -----------------------------------------------------
    # Trend
    # -----------------------------------------------------

    if analysis_type == "trend":

        intents.append({
            "type": "trend",
            "metric": metric_hint,
            "dimension": dimension_hint,
            "aggregation": aggregation or "sum"
        })

    # -----------------------------------------------------
    # Distribution
    # -----------------------------------------------------

    if analysis_type == "distribution":

        intents.append({
            "type": "distribution",
            "metric": metric_hint
        })

    # -----------------------------------------------------
    # Relationship
    # -----------------------------------------------------

    if analysis_type == "relationship":

        intents.append({
            "type": "relationship",
            "metric": metric_hint
        })

    # -----------------------------------------------------
    # Share
    # -----------------------------------------------------

    if analysis_type == "share":

        intents.append({
            "type": "share",
            "metric": metric_hint,
            "dimension": dimension_hint
        })

    # -----------------------------------------------------
    # Ranking
    # -----------------------------------------------------

    if analysis_type == "ranking":

        intents.append({
            "type": "ranking",
            "metric": metric_hint,
            "ranking": ranking,
            "limit": limit
        })

    # -----------------------------------------------------
    # Grouped aggregation
    # -----------------------------------------------------

    if (
        aggregation
        and dimension_hint
        and analysis_type == "aggregation"
    ):

        intents.append({
            "type": "grouped",
            "metric": metric_hint,
            "aggregation": aggregation,
            "dimension": dimension_hint,
            "ranking": ranking
        })

    return intents


# =========================================================
# MAIN QUESTION PARSER
# =========================================================

def parse_question(question):
    """
    Generic natural-language question parser.

    This function does NOT depend on a particular dataset.

    Supported concepts:

    - aggregation
    - ranking
    - top N
    - comparison
    - metric hint
    - dimension hint
    - trend
    - distribution
    - relationship / correlation
    - percentage / share
    - employee/person lookup
    - numeric filters
    - time granularity
    - multiple intents

    The returned dictionary preserves the original keys used
    by the application while adding generic analysis metadata.
    """

    question = str(
        question or ""
    ).strip()

    if not question:

        return {
            "aggregation": None,
            "ranking": None,
            "limit": None,
            "comparison": False,
            "metric_hint": None,
            "dimension_hint": None,
            "analysis_type": "general",
            "time_granularity": None,
            "distribution": False,
            "relationship": False,
            "share": False,
            "employee_lookup": False,
            "filter_operator": None,
            "filter_value": None,
            "multiple_intents": []
        }

    q = _clean_text(
        question
    )

    # =====================================================
    # BASIC INTENTS
    # =====================================================

    aggregation = _detect_aggregation(
        q
    )

    ranking = _detect_ranking(
        q
    )

    limit = _detect_limit(
        q
    )

    comparison = _detect_comparison(
        q
    )

    metric_hint = _detect_metric(
        q
    )

    dimension_hint = _detect_dimension(
        q
    )

    analysis_type = _detect_analysis_type(
        q
    )

    time_granularity = _detect_time_granularity(
        q
    )

    distribution = _detect_distribution(
        q
    )

    relationship = _detect_relationship(
        q
    )

    share = _detect_share(
        q
    )

    employee_lookup = _detect_person_lookup(
        q
    )

    # =====================================================
    # FILTER
    # =====================================================

    filter_operator, filter_value = _detect_filter(
        q
    )

    # =====================================================
    # NATURAL LANGUAGE OVERRIDES
    # =====================================================

    metric_hint, ranking = _apply_natural_language_rules(
        q,
        metric_hint,
        ranking
    )

    # =====================================================
    # SPECIAL ANALYSIS TYPES
    # =====================================================

    if distribution:
        analysis_type = "distribution"

    elif relationship:
        analysis_type = "relationship"

    elif share:
        analysis_type = "share"

    elif time_granularity:
        analysis_type = "trend"

    elif comparison:
        analysis_type = "comparison"

    elif ranking:
        analysis_type = "ranking"

    # =====================================================
    # DEFAULT AGGREGATION
    # =====================================================

    # Comparison questions normally need an aggregation.
    #
    # We deliberately do not force a metric-specific value
    # here. The calculation layer can choose sum/mean based
    # on the actual question and dataset.
    #
    # This keeps the parser dataset-independent.

    if (
        comparison
        and aggregation is None
    ):
        aggregation = None

    # =====================================================
    # MULTIPLE INTENTS
    # =====================================================

    multiple_intents = _build_multiple_intents(
        q,
        aggregation,
        ranking,
        metric_hint,
        comparison,
        analysis_type,
        dimension_hint,
        limit
    )

    # =====================================================
    # ADD FILTER INTENT
    # =====================================================

    if filter_operator:

        multiple_intents.append({
            "type": "filter",
            "operator": filter_operator,
            "value": filter_value,
            "metric": metric_hint
        })

    # =====================================================
    # ADD LOOKUP INTENT
    # =====================================================

    if employee_lookup:

        multiple_intents.append({
            "type": "lookup",
            "metric": metric_hint
        })

    # =====================================================
    # ADD TREND METADATA
    # =====================================================

    if analysis_type == "trend":

        multiple_intents.append({
            "type": "time_analysis",
            "metric": metric_hint,
            "granularity": time_granularity
        })

    # =====================================================
    # ADD DISTRIBUTION METADATA
    # =====================================================

    if analysis_type == "distribution":

        multiple_intents.append({
            "type": "distribution",
            "metric": metric_hint
        })

    # =====================================================
    # ADD RELATIONSHIP METADATA
    # =====================================================

    if analysis_type == "relationship":

        multiple_intents.append({
            "type": "correlation",
            "metric": metric_hint
        })

    # =====================================================
    # ADD SHARE METADATA
    # =====================================================

    if analysis_type == "share":

        multiple_intents.append({
            "type": "percentage_share",
            "metric": metric_hint,
            "dimension": dimension_hint
        })

    # =====================================================
    # RETURN
    # =====================================================

    return {
        # -------------------------------------------------
        # Existing application fields
        # -------------------------------------------------

        "aggregation": aggregation,

        "ranking": ranking,

        "limit": limit,

        "comparison": comparison,

        "metric_hint": metric_hint,

        "employee_lookup": employee_lookup,

        "filter_operator": filter_operator,

        "filter_value": filter_value,

        "multiple_intents": multiple_intents,

        # -------------------------------------------------
        # New generic fields
        # -------------------------------------------------

        "dimension_hint": dimension_hint,

        "analysis_type": analysis_type,

        "time_granularity": time_granularity,

        "distribution": distribution,

        "relationship": relationship,

        "share": share
    }