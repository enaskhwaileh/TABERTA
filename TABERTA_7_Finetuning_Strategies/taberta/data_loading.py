"""
WikiDBs corpus loader — reads from on-disk CSV/JSON format.

Each database is a folder containing:
  - info_full.json   (metadata, schema, FK info, column datatypes)
  - tables/*.csv     (actual table data)
"""

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ForeignKey:
    """A foreign key link from one table to another within the same database."""
    column: str
    property_id: str
    reference_table: str


@dataclass
class TableRecord:
    """A single table within a database."""
    table_name: str
    database_name: str
    columns: List[str]
    column_datatypes: List[str]
    rows: List[Dict[str, str]]
    foreign_keys: List[ForeignKey] = field(default_factory=list)
    filepath: str = ""
    num_rows_total: int = 0

    @property
    def num_rows(self) -> int:
        return len(self.rows)

    @property
    def num_columns(self) -> int:
        return len(self.columns)


@dataclass
class DatabaseRecord:
    """A relational database consisting of multiple tables."""
    database_name: str
    tables: Dict[str, TableRecord]
    info_full: Optional[str] = None

    @property
    def table_names(self) -> List[str]:
        return list(self.tables.keys())

    @property
    def num_tables(self) -> int:
        return len(self.tables)

    def get_fk_targets(self, table_name: str) -> List[str]:
        """Return table names that the given table has FK references to."""
        table = self.tables.get(table_name)
        if not table:
            return []
        return [
            fk.reference_table
            for fk in table.foreign_keys
            if fk.reference_table in self.tables
        ]


class WikiDBsCorpus:
    """
    Loads the WikiDBs corpus from disk.

    Parameters
    ----------
    root_path : str or Path
        Path to the ``databases/`` directory containing one folder per database.
    row_limit : int
        Maximum number of rows to load per table (default 3, matching the paper).
    """

    def __init__(self, root_path: str, row_limit: int = 3):
        self.root_path = Path(root_path)
        self.row_limit = row_limit
        self._db_names: Optional[List[str]] = None
        self._cache: Dict[str, DatabaseRecord] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def database_names(self) -> List[str]:
        """List all database folder names (lazy, cached)."""
        if self._db_names is None:
            self._db_names = sorted(
                d for d in os.listdir(self.root_path)
                if (self.root_path / d / "info_full.json").exists()
            )
        return self._db_names

    @property
    def num_databases(self) -> int:
        return len(self.database_names)

    def load_database(self, db_name: str) -> DatabaseRecord:
        """Load a single database (cached after first read)."""
        if db_name in self._cache:
            return self._cache[db_name]

        db_path = self.root_path / db_name
        info_path = db_path / "info_full.json"

        if not info_path.exists():
            raise FileNotFoundError(f"No info_full.json in {db_path}")

        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        tables_meta = info.get("TABLES", {})
        info_text = json.dumps(info.get("INFO", {}))

        tables: Dict[str, TableRecord] = {}
        for table_name, meta in tables_meta.items():
            csv_filename = meta.get("FILEPATH", "")
            csv_path = db_path / "tables" / csv_filename

            # Read rows from CSV
            rows = self._read_csv(csv_path) if csv_path.exists() else []

            # Parse FK info
            fks = [
                ForeignKey(
                    column=fk_entry["FOREIGN_KEY"][1] if len(fk_entry.get("FOREIGN_KEY", [])) > 1 else "",
                    property_id=fk_entry["FOREIGN_KEY"][0] if fk_entry.get("FOREIGN_KEY") else "",
                    reference_table=fk_entry.get("REFERENCE_TABLE", ""),
                )
                for fk_entry in meta.get("FOREIGN_KEYS", [])
            ]

            tables[table_name] = TableRecord(
                table_name=table_name,
                database_name=db_name,
                columns=meta.get("COLUMNS", []),
                column_datatypes=meta.get("COLUMN_DATATYPES", []),
                rows=rows,
                foreign_keys=fks,
                filepath=csv_filename,
                num_rows_total=meta.get("NUM_ROWS", len(rows)),
            )

        record = DatabaseRecord(
            database_name=db_name,
            tables=tables,
            info_full=info_text,
        )
        self._cache[db_name] = record
        return record

    def iter_databases(self):
        """Iterate over all databases, yielding DatabaseRecord objects."""
        for name in self.database_names:
            try:
                yield self.load_database(name)
            except Exception as e:
                logger.warning(f"Skipping database '{name}': {e}")

    def iter_tables(self):
        """Iterate over all tables across all databases."""
        for db in self.iter_databases():
            for table in db.tables.values():
                yield table

    def clear_cache(self):
        """Free memory by clearing the database cache."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_csv(self, csv_path: Path) -> List[Dict[str, str]]:
        """Read up to ``row_limit`` rows from a CSV file as list of dicts."""
        rows = []
        try:
            with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= self.row_limit:
                        break
                    rows.append(dict(row))
        except Exception as e:
            logger.warning(f"Error reading {csv_path}: {e}")
        return rows
