import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger("haven.data_analysis")

class DataAnalysisModule:
    """
    Automated Exploratory Data Analysis (EDA) and Interactive Visualization Module.
    Inspects CSV/Excel datasets, generates summary statistics, and builds Plotly figures.
    """

    @classmethod
    def load_dataset(cls, file_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Loads CSV or Excel file into a Pandas DataFrame."""
        try:
            ext = file_path.suffix.lower()
            if ext == ".csv":
                df = pd.read_csv(file_path)
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            else:
                return None, f"Unsupported spreadsheet format: {ext}"
            return df, None
        except Exception as e:
            return None, f"Failed to load dataset: {str(e)}"

    @classmethod
    def analyze_dataset(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates comprehensive statistical summary, missing value profile, and datatypes.
        """
        num_rows, num_cols = df.shape
        missing_counts = df.isnull().sum().to_dict()
        data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(df.select_dtypes(include=["str", "category"]).columns)

        stats_summary = {}
        if numeric_cols:
            stats_summary = df[numeric_cols].describe().to_dict()

        return {
            "rows": num_rows,
            "columns": num_cols,
            "column_names": list(df.columns),
            "data_types": data_types,
            "missing_values": missing_counts,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "statistics": stats_summary
        }

    @classmethod
    def generate_chart(cls, df: pd.DataFrame, chart_type: str = "auto", x_col: Optional[str] = None, y_col: Optional[str] = None) -> Optional[go.Figure]:
        """
        Creates an interactive Plotly figure based on detected column types or user selection.
        """
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(df.select_dtypes(include=["str", "category"]).columns)

        if not x_col and df.columns.size > 0:
            x_col = cat_cols[0] if cat_cols else df.columns[0]
        if not y_col and numeric_cols:
            y_col = numeric_cols[0]

        try:
            if chart_type == "bar" or (chart_type == "auto" and x_col and y_col):
                fig = px.bar(
                    df.head(25),
                    x=x_col,
                    y=y_col,
                    title=f"Distribution Analysis: {y_col} by {x_col}",
                    template="plotly_dark",
                    color_discrete_sequence=["#34d399"]
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                return fig

            elif chart_type == "scatter" and len(numeric_cols) >= 2:
                fig = px.scatter(
                    df.head(100),
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    title=f"Correlation Scatter: {numeric_cols[0]} vs {numeric_cols[1]}",
                    template="plotly_dark",
                    color_discrete_sequence=["#60a5fa"]
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                return fig

            elif numeric_cols:
                fig = px.histogram(
                    df,
                    x=numeric_cols[0],
                    title=f"Histogram Frequency Distribution: {numeric_cols[0]}",
                    template="plotly_dark",
                    color_discrete_sequence=["#a78bfa"]
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                return fig

        except Exception as e:
            logger.error("Chart generation error: %s", e)

        return None

