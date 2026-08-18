import pandas as pd


def profile_dataset(df):
    """Create a structured profile of any uploaded dataset."""

    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_details": []
    }

    for column in df.columns:

        series = df[column]

        # ---------------------------------------------
        # Detect numeric columns
        # ---------------------------------------------
        if pd.api.types.is_numeric_dtype(series):

            data_type = "numeric"

        # ---------------------------------------------
        # Detect existing datetime columns
        # ---------------------------------------------
        elif pd.api.types.is_datetime64_any_dtype(series):

            data_type = "date"

        # ---------------------------------------------
        # Detect date-looking text columns
        # ---------------------------------------------
        else:

            converted_dates = pd.to_datetime(
                series,
                errors="coerce"
            )

            valid_date_ratio = converted_dates.notna().mean()

            if valid_date_ratio >= 0.80:

                data_type = "date"

            else:

                data_type = "categorical/text"

        # ---------------------------------------------
        # Basic column information
        # ---------------------------------------------

        column_info = {
            "name": column,
            "type": data_type,
            "missing": int(series.isna().sum()),
            "unique_values": int(series.nunique())
        }

        # ---------------------------------------------
        # Numeric statistics
        # ---------------------------------------------

        if data_type == "numeric":

            column_info["min"] = float(series.min())
            column_info["max"] = float(series.max())
            column_info["average"] = float(series.mean())

        # ---------------------------------------------
        # Date information
        # ---------------------------------------------

        elif data_type == "date":

            converted_dates = pd.to_datetime(
                series,
                errors="coerce"
            )

            valid_dates = converted_dates.dropna()

            if not valid_dates.empty:

                column_info["min"] = str(
                    valid_dates.min().date()
                )

                column_info["max"] = str(
                    valid_dates.max().date()
                )

        # ---------------------------------------------
        # Store column information
        # ---------------------------------------------

        profile["column_details"].append(
            column_info
        )

    return profile