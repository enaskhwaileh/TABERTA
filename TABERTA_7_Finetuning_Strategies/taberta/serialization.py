"""
Table serialization views (paper Section 3.1).

Three views convert a TableRecord into a text string consumed by the encoder:

  SchemaView(T) = [TABLE] N_T [SCHEMA] c1, c2, ..., cm
  RowView(T)    = [TABLE] N_T [ROWS] [ROW] c1:v11 | c2:v12 ... [ROW] c1:v21 | ...
  FullView(T)   = [DB] N_DB [TABLE] N_T [SCHEMA] c1, ..., cm [ROWS] r1 | r2 | ...
"""

from taberta.data_loading import TableRecord


def schema_view(table: TableRecord) -> str:
    """
    Schema-only view — encodes table identity and column headers,
    omitting all cell values.

    Format:
        [TABLE]<table_name> [SCHEMA]<col1>,<col2>,...,<colm>

    Used by: TC-Opt (combined), Hybrid stage 2 (combined).
    """
    schema_str = ",".join(table.columns)
    return f"[TABLE]{table.table_name} [SCHEMA]{schema_str}"


def row_view(table: TableRecord) -> str:
    """
    Row-level view — represents table content as column:value pairs per row.

    Format:
        [TABLE]<table_name> [ROWS][ROW] c1:v11 | c2:v12 [ROW] c1:v21 | c2:v22

    Used by: PC (Pairwise Contrastive).
    """
    parts = [f"[TABLE]{table.table_name} [ROWS]"]
    for row in table.rows:
        pairs = " | ".join(
            f"{col}:{row.get(col, '')}" for col in table.columns
        )
        parts.append(f"[ROW] {pairs}")
    return "".join(parts)


def full_view(table: TableRecord) -> str:
    """
    Full structured view — concatenates database context, schema, and content.

    Format:
        [DB]<db_name> [TABLE]<table_name> [SCHEMA]<col1>,...,<colm> [ROWS]<row1>|<row2>|...

    Each row is a comma-separated sequence of values.

    Used by: TC, TC-SB, TC-Opt (combined), SS-C, MLM, Hybrid.
    """
    schema_str = ",".join(table.columns)
    rows_str = "|".join(
        ",".join(str(row.get(col, "")) for col in table.columns)
        for row in table.rows
    )
    return (
        f"[DB]{table.database_name} "
        f"[TABLE]{table.table_name} "
        f"[SCHEMA]{schema_str} "
        f"[ROWS]{rows_str}"
    )


# ---------------------------------------------------------------------------
# Convenience: serialize a table under any named view
# ---------------------------------------------------------------------------
_VIEW_REGISTRY = {
    "schema": schema_view,
    "row": row_view,
    "full": full_view,
}


def serialize(table: TableRecord, view: str) -> str:
    """
    Serialize a table using the named view.

    Parameters
    ----------
    table : TableRecord
    view : str
        One of ``"schema"``, ``"row"``, ``"full"``.

    Returns
    -------
    str
        The serialized text representation.
    """
    fn = _VIEW_REGISTRY.get(view)
    if fn is None:
        raise ValueError(f"Unknown view '{view}'. Choose from {list(_VIEW_REGISTRY)}")
    return fn(table)
