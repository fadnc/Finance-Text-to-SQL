"""Schema-aware guardrails for SQL returned by the inference API.

The model is small and sometimes emits non-SQLite functions, repeated joins, or
column/table hallucinations.  The API uses these helpers to normalize obvious
dialect differences, remove duplicate joins, and validate the statement against
an in-memory database built from the supplied schema before returning it.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Optional, Tuple

_WRITE_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX)\b",
    re.IGNORECASE,
)
_ALLOWED_START = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_COMPACT_TABLE = re.compile(r"([A-Za-z_][\w]*)\s*\(([^()]*)\)")
_TABLE_NAME = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"`\[]?([A-Za-z_][\w]*)",
    re.IGNORECASE,
)
_DATE_INTERVAL = re.compile(
    r"\bDATE_(SUB|ADD)\s*\(\s*"
    r"(CURDATE\s*\(\s*\)|CURRENT_DATE|NOW\s*\(\s*\)|[A-Za-z_][\w.]*|'[^']*')\s*,\s*"
    r"INTERVAL\s+(\d+)\s+(DAY|MONTH|YEAR|HOUR|MINUTE|SECOND)S?\s*\)",
    re.IGNORECASE,
)
_JOIN_START = re.compile(
    r"\b(?:(LEFT|RIGHT|FULL|INNER|OUTER|CROSS)\s+)?JOIN\s+"
    r"([A-Za-z_][\w]*)(?:\s+(?:AS\s+)?([A-Za-z_][\w]*))?\s+ON\b",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY = re.compile(
    r"\b(?:JOIN|LEFT|RIGHT|FULL|INNER|OUTER|CROSS|WHERE|GROUP\s+BY|ORDER\s+BY|"
    r"LIMIT|HAVING|UNION|OFFSET|WINDOW)\b",
    re.IGNORECASE,
)


def schema_to_ddl(schema: str) -> str:
    """Return executable SQLite DDL from either full or compact schema text."""
    if re.search(r"\bCREATE\s+TABLE\b", schema, re.IGNORECASE):
        return schema

    statements = []
    for table, columns in _COMPACT_TABLE.findall(schema):
        names = []
        for column in columns.split(","):
            name = column.strip().split()[0] if column.strip() else ""
            if name:
                names.append(name)
        if names:
            statements.append(f"CREATE TABLE {table} ({', '.join(names)});")
    return "\n".join(statements)


def table_names(schema: str) -> set[str]:
    """Return lower-cased table names declared by a schema."""
    return {match.group(1).lower() for match in _TABLE_NAME.finditer(schema_to_ddl(schema))}


def column_names(schema: str, table: str) -> set[str]:
    """Return lower-cased column names for one table in the schema."""
    if not re.fullmatch(r"[A-Za-z_][\w]*", table):
        return set()

    conn = _connect(schema)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {row[1].lower() for row in rows}
    finally:
        conn.close()


def _connect(schema: str) -> sqlite3.Connection:
    """Create an empty in-memory database from the supplied schema."""
    ddl = schema_to_ddl(schema)
    if not ddl.strip():
        raise ValueError("Schema does not contain any CREATE TABLE statements")

    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(ddl)
    except Exception:
        conn.close()
        raise
    return conn


def normalize_sqlite_compat(sql: str) -> str:
    """Rewrite common MySQL/Postgres date expressions into SQLite equivalents."""
    sql = _DATE_INTERVAL.sub(_replace_interval, sql)
    sql = re.sub(r"\bCURDATE\s*\(\s*\)", "date('now')", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bCURRENT_DATE\b", "date('now')", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bNOW\s*\(\s*\)", "datetime('now')", sql, flags=re.IGNORECASE)
    return sql


def _replace_interval(match: re.Match[str]) -> str:
    operation, base, amount, unit = match.groups()
    base = base.strip()
    lower_base = base.lower()
    if lower_base.startswith("curdate") or lower_base in {"current_date", "now()"}:
        base_sql = "'now'"
    else:
        base_sql = base

    sign = "-" if operation.upper() == "SUB" else "+"
    unit = unit.lower()
    if int(amount) != 1:
        unit += "s"
    return f"date({base_sql}, '{sign}{amount} {unit}')"


def _clause_end(sql: str, start: int) -> int:
    """Find the end of an ON clause without splitting nested subqueries."""
    depth = 0
    quote: Optional[str] = None
    index = start
    while index < len(sql):
        char = sql[index]
        if quote:
            if char == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if char in "'\"`":
            quote = char
            index += 1
            continue
        if char == "(":
            depth += 1
            index += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            index += 1
            continue
        if depth == 0:
            if char == ";":
                return index
            if _CLAUSE_BOUNDARY.match(sql, index):
                return index
        index += 1
    return len(sql)


def deduplicate_joins(sql: str) -> str:
    """Drop repeated, unaliased joins so the table is not mentioned twice.

    The model often produces output such as ``JOIN accounts ... JOIN accounts``.
    SQLite treats the second reference as ambiguous, so keep the first join and
    remove later unaliased joins for the same table.  Self-joins with explicit
    aliases are left untouched.
    """
    matches = list(_JOIN_START.finditer(sql))
    if not matches:
        return sql

    seen: set[str] = set()
    removals: list[tuple[int, int]] = []
    for match in matches:
        table = match.group(2).lower()
        alias = (match.group(3) or "").lower()
        key = alias or table
        if not alias and key in seen:
            removals.append((match.start(), _clause_end(sql, match.end())))
        else:
            seen.add(key)

    for start, end in reversed(removals):
        sql = f"{sql[:start]} {sql[end:]}"
    return re.sub(r"\s+", " ", sql).strip()


def validate_sql(sql: str, schema: str) -> Tuple[bool, Optional[str]]:
    """Check that a statement is read-only and valid for the supplied schema."""
    statement = (sql or "").strip()
    if not statement:
        return False, "empty SQL"

    if _WRITE_KEYWORDS.search(statement):
        return False, "only read-only SELECT queries are supported"
    if not _ALLOWED_START.match(statement):
        return False, "only SELECT or WITH queries are supported"

    body = statement.rstrip().rstrip(";")
    if ";" in body:
        return False, "multiple SQL statements are not supported"

    try:
        conn = _connect(schema)
        try:
            conn.execute("EXPLAIN QUERY PLAN " + body)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)

    return True, None


def prepare_sql(sql: str, schema: str) -> Tuple[str, bool, Optional[str], str]:
    """Normalize a model response and validate it against the schema.

    Returns ``(sql, valid, error, source)`` where source is ``model`` when the
    original statement was valid, ``model-normalized`` after dialect cleanup, or
    ``invalid`` when the statement does not match the schema.

    Duplicate-join removal is intentionally not automatic here: dropping a join
    can make a query executable while changing its meaning.  The API prefers a
    schema-aware fallback and only uses :func:`deduplicate_joins` as a last
    resort for custom schemas.
    """
    normalized = normalize_sqlite_compat(sql)
    valid, error = validate_sql(normalized, schema)
    if valid:
        source = "model-normalized" if normalized.strip() != sql.strip() else "model"
        return normalized, True, None, source
    return normalized, False, error, "invalid"
