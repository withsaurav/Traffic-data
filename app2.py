# app.py
import os
import re
from typing import Dict, List

import streamlit as st
from google.cloud import bigquery
from google.api_core.exceptions import BadRequest

from google.oauth2 import service_account


import json




BQ_LOCATION = "US"  # adjust if needed

if "GOOGLE_CREDENTIALS" in st.secrets:
    info = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
    # normalize private_key just in case the \n are literal
    pk = info.get("private_key", "")
    if "\\n" in pk and "\n" not in pk:
        info["private_key"] = pk.replace("\\n", "\n")

    creds = service_account.Credentials.from_service_account_info(info)
    bq = bigquery.Client(project=info["project_id"], credentials=creds, location=BQ_LOCATION)
else:
    bq = bigquery.Client(project="steam-airfoil-341409", location=BQ_LOCATION)




# =========================
# Project constants (edit)
# =========================
PROJECT = "steam-airfoil-341409"
DATASET = "License_Plate"
TABLE = "Traffic_data"

# --- Time filtering config ---
TIME_FROM_COL = "from_time"   # change if your column name differs
TIME_TO_COL   = "to_time"     # change if your column name differs
TIME_IS_STRING = True        # True if your time columns are stored as 'HH:MM:SS' strings
DEFAULT_ORDER_BY = f"{TIME_FROM_COL} ASC"
#---------------------------------

# Your BigQuery ML remote model (already created) that points to Gemini 2.5 Pro
LLM_DATASET = "License_Plate"
LLM_MODEL = "gemini_remote"   # created with ENDPOINT 'gemini-2.5-pro'

# Guardrails: only allow queries within these datasets
ALLOWED_DATASETS = {DATASET}

# Optional: set default project env var (handy outside GCP)
os.environ.setdefault("GCLOUD_PROJECT", PROJECT)

# BigQuery client
bq = bigquery.Client(project=PROJECT)

st.set_page_config(page_title="Traffic Data NL → SQL (BigQuery + Gemini)", layout="wide")
st.title("Ask your Traffic Data (Gemini → SQL → BigQuery)")

# =========================
# Schema Introspection (fixed: no 'description' error)
# =========================
@st.cache_data(show_spinner=False, ttl=600)
def get_schema(dataset: str) -> Dict[str, List[Dict[str, str]]]:
    sql = f"""
    WITH cols AS (
      SELECT table_name, column_name, data_type, is_nullable, ordinal_position
      FROM `{PROJECT}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
    ),
    descs AS (
      SELECT table_name, column_name, description
      FROM `{PROJECT}.{dataset}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`
      WHERE field_path = column_name
    )
    SELECT
      c.table_name,
      c.column_name,
      c.data_type,
      c.is_nullable,
      d.description
    FROM cols c
    LEFT JOIN descs d
    USING (table_name, column_name)
    ORDER BY c.table_name, c.ordinal_position
    """
    rows = bq.query(sql).result()
    out: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        out.setdefault(r["table_name"], []).append({
            "name": r["column_name"],
            "type": r["data_type"],
            "nullable": r["is_nullable"],
            "description": (r["description"] or "")
        })
    return out

def schema_to_text(schema: Dict[str, List[Dict[str, str]]]) -> str:
    lines = []
    for t, cols in schema.items():
        col_parts = [f"{c['name']} ({c['type']})" for c in cols]
        lines.append(f"- {DATASET}.{t}: " + ", ".join(col_parts))
    return "\n".join(lines)



# =========================
# Foe time related query
# =========================

def _user_requested_ordering(q: str) -> bool:
    ql = q.lower()
    # Only treat these as explicit sort asks
    return bool(re.search(r'\b(order by|sort|sorted|newest|latest|oldest|ascending|descending|asc|desc)\b', ql))

def _strip_trailing_order_by(sql: str) -> str:
    # Remove a trailing ORDER BY ... (keep LIMIT if present)
    return re.sub(r'(?is)\border\s+by\b.+?(?=(\blimit\b|$))', '', sql).strip()


def inject_time_filter(sql: str, user_query: str,
                       from_col: str = TIME_FROM_COL,
                       to_col: str   = TIME_TO_COL,
                       default_order_by: str = DEFAULT_ORDER_BY) -> str:
    """
    Force a simple LIKE filter on from_time when user mentions HH:MM:SS.
    - Strips any TIME()/PARSE_TIME() comparisons the LLM may have added.
    - Injects: WHERE from_time LIKE '%HH:MM:SS%'
    - Safely merges with existing WHERE and preserves ORDER BY / LIMIT.
    """
    m = re.search(r'\b([01]\d|2[0-3]):([0-5]\d):([0-5]\d)\b', user_query)
    if not m:
        return sql
    hhmmss = m.group(0)

    s = sql.strip().rstrip(';').strip()

    # Separate tail (ORDER BY / LIMIT) to reattach later
    tail = ""
    m_tail = re.search(r'(?is)\b(ORDER\s+BY|LIMIT)\b.*$', s)
    if m_tail:
        tail = s[m_tail.start():].strip()
        s = s[:m_tail.start()].rstrip()

    # --- 1) REMOVE any time-function predicates (both orders) ---
    time_cmp = r"(?is)(?:\s+(?:AND|OR)\s+)?(?:(?:SAFE\.)?PARSE_TIME\([^)]*\)|TIME\([^)]*\))\s*(?:<=|>=|=)\s*TIME\s*'[^']+'"
    s = re.sub(time_cmp, "", s)
    rev_time_cmp = r"(?is)(?:\s+(?:AND|OR)\s+)?TIME\s*'[^']+'\s*(?:<=|>=|=)\s*(?:(?:SAFE\.)?PARSE_TIME\([^)]*\)|TIME\([^)]*\))"
    s = re.sub(rev_time_cmp, "", s)

    # Clean any 'WHERE AND' / trailing connectors left behind
    s = re.sub(r'(?is)\bWHERE\s+(?:AND|OR)\s+', 'WHERE ', s).strip()
    s = re.sub(r'(?is)\s+(?:AND|OR)\s*$', '', s).strip()

    # --- 2) INJECT our simple LIKE predicate on from_time ---
    pred = f"{from_col} LIKE '%{hhmmss}%'"

    m_where = re.search(r'(?is)\bWHERE\b(.*)$', s)
    if m_where:
        head = s[:m_where.start()].rstrip()
        where_body = (m_where.group(1) or "").strip()

        # If it already contains a LIKE on from_col for the same time, don't double add
        already = re.search(rf"(?is){re.escape(from_col)}\s+LIKE\s+['\"]%{re.escape(hhmmss)}%['\"]", where_body)
        # Clean trailing AND/OR in the existing body
        where_body = re.sub(r'(?is)\s*(?:AND|OR)\s*$', '', where_body).strip()

        new_where = where_body if already else (f"{where_body} AND {pred}" if where_body else pred)
        s = f"{head}\nWHERE {new_where}".strip()
    else:
        s = f"{s}\nWHERE {pred}"

    # Final cleanup again (just in case)
    s = re.sub(r'(?is)\bWHERE\s+(?:AND|OR)\s+', 'WHERE ', s).strip()
    s = re.sub(r'(?is)\s+(?:AND|OR)\s*$', '', s).strip()

    # Reattach tail (do NOT auto-add ORDER BY here)
    if tail:
        s = f"{s}\n{tail}"
    return s
# =========================
# Prompt Construction (no LIMIT in few-shots)
# =========================
FEW_SHOTS = """
Example 1
User: How many silver cars are recorded?
SQL:
SELECT COUNT(*) AS count
FROM `steam-airfoil-341409.License_Plate.Traffic_data`
WHERE car_color = 'Silver';

Example 2
User: Show the number plates of white cars, newest first.
SQL:
SELECT license_number
FROM `steam-airfoil-341409.License_Plate.Traffic_data`
WHERE car_color = 'White'
ORDER BY from_time DESC;

Example 3
User: List license_number, car_color and from_time for vehicles between 2024-07-01 and 2024-07-31.
SQL:
SELECT license_number, car_color, from_time
FROM `steam-airfoil-341409.License_Plate.Traffic_data`
WHERE from_time >= TIMESTAMP('2024-07-01') AND from_time < TIMESTAMP('2024-08-01')
ORDER BY from_time DESC;

Example 4
User: Which different car colors are found in the records?
SQL:
SELECT DISTINCT car_color
FROM `steam-airfoil-341409.License_Plate.Traffic_data`
ORDER BY car_color;
"""



def build_system_prompt(schema_text: str, user_question: str) -> str:
    rules = f"""
You are an expert BigQuery SQL generator.
Return ONLY ONE BigQuery SELECT statement. No comments, no code fences, no explanations.

Rules:
- Use GoogleSQL (standard SQL).
- READ-ONLY: no DDL/DML (no CREATE/INSERT/UPDATE/DELETE/MERGE/TRUNCATE/ALTER/DROP).
- Only use dataset(s): {', '.join(sorted(ALLOWED_DATASETS))}.
- Use fully-qualified table names like `{PROJECT}.{DATASET}.{TABLE}`.
- Prefer deterministic counts (COUNT(*)) for “how many” questions; alias as `count`.
- Include LIMIT N when the user explicitly asks for a bounded result (e.g., "top 5", "first 5", "only 5", "5 records/rows"). Otherwise, do not include LIMIT.
- When the user asks for "different", "unique", or "distinct" values, use SELECT DISTINCT and add an ORDER BY on that column.
- Match exact case for enum-like values (e.g., car_color values are 'Silver', 'Blue', etc.).
- - If the user mentions an exact time (HH:MM:SS), prefer filtering with:
    WHERE from_time LIKE '%HH:MM:SS%'
- Always include a WHERE when time is specified.
- Output pure SQL (no markdown fences).
- NEVER use ML.GENERATE_TEXT or any ML.* function in your output SQL.

Relevant columns include (not exhaustive): car_id, license_number, license_plate_number, license_number_score, frame_nbr, file_name, from_time, to_time, car_color.

Available schema:
{schema_text}

{FEW_SHOTS}

User question: {user_question}
SQL:
"""
    return rules.strip()


# =========================
# LLM call via BigQuery ML + retry
# =========================
def call_gemini_generate_sql(prompt_text: str) -> str:
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("prompt", "STRING", prompt_text)]
    )
    sql = f"""
    SELECT ml_generate_text_llm_result AS llm_text
    FROM ML.GENERATE_TEXT(
      MODEL `{PROJECT}.{LLM_DATASET}.{LLM_MODEL}`,
      (SELECT @prompt AS prompt),
      STRUCT(
        0.15 AS temperature,
        1024 AS max_output_tokens,
        TRUE AS flatten_json_output
      )
    )
    """
    df = bq.query(sql, job_config=job_config).result().to_dataframe()
    if df.empty:
        return ""
    return str(df.iloc[0]["llm_text"] or "").strip()

def generate_sql_with_retry(schema_text: str, user_question: str) -> str:
    """Try full system prompt; if empty, try minimal prompt."""
    p1 = build_system_prompt(schema_text, user_question)
    t1 = call_gemini_generate_sql(p1)
    if t1:
        return t1

    p2 = f"""
Return only one BigQuery SELECT statement (no comments, no code fences).
Use only the fully-qualified table `{PROJECT}.{DATASET}.{TABLE}`.
Do not include LIMIT unless the user explicitly asks for a top-N.
Question: {user_question}
SQL:
""".strip()
    t2 = call_gemini_generate_sql(p2)
    return t2 or ""

# =========================
# Heuristic SQL fallback (for simple plate/color/time asks)
# =========================
@st.cache_data(show_spinner=False, ttl=300)
def _distinct_values(table_fqn: str, col: str):
    try:
        df = bq.query(
            f"SELECT DISTINCT {col} AS v FROM {table_fqn} WHERE {col} IS NOT NULL"
        ).result().to_dataframe()
        return sorted([str(x) for x in df["v"].dropna().unique()])
    except Exception:
        return []

def _pick_col(schema: dict, table: str, candidates: list, contains_any: list = None):
    cols = [c["name"] for c in schema.get(table, [])]
    lowmap = {c.lower(): c for c in cols}
    for c in candidates:
        if c in lowmap:
            return lowmap[c]
    if contains_any:
        for c in cols:
            lc = c.lower()
            if any(tok in lc for tok in contains_any):
                return c
    return None

def fallback_sql_from_question(question: str, schema: dict) -> str:
    """
    Heuristic builder for simple asks:
      - 'show number plate of silver cars'
      - 'number plate and time for blue cars'
    Returns '' if we can't infer safely.
    """
    q = question.lower()
    table = TABLE
    table_fqn = f"`{PROJECT}.{DATASET}.{table}`"

    # Best-guess column names from schema
    plate_col = _pick_col(
        schema, table,
        candidates=[
            "license_number", "license_plate_number", "plate_number",
            "registration_number", "licence_number", "licence_plate_number"
        ],
        contains_any=["plate", "license", "licence", "regist"]
    )
    time_col = _pick_col(
        schema, table,
        candidates=["from_time", "to_time", "timestamp", "event_time", "time", "created_at"],
        contains_any=["time", "timestamp"]
    )
    color_col = _pick_col(
        schema, table,
        candidates=["car_color", "color"],
        contains_any=["color"]
    )

    if not plate_col:
        return ""  # cannot build a sensible fallback without a plate column

    # Detect color(s) mentioned, map to proper case from table values
    where_clause = ""
    if color_col:
        all_vals = _distinct_values(table_fqn, color_col)
        low_to_proper = {s.lower(): s for s in all_vals}
        ask_colors = [c for c in low_to_proper if c in q]
        if len(ask_colors) == 1:
            where_clause = f"WHERE {color_col} = '{low_to_proper[ask_colors[0]]}'"
        elif len(ask_colors) > 1:
            in_list = ", ".join([f"'{low_to_proper[c]}'" for c in ask_colors])
            where_clause = f"WHERE {color_col} IN ({in_list})"

    # Build SELECT list (plate, and include time if the question implies it)
    want_time = ("time" in q) or ("from_time" in q) or ("to_time" in q) or ("newest" in q) or ("latest" in q)
    select_cols = [plate_col] + ([time_col] if want_time and time_col else [])
    select_sql = ", ".join(select_cols)

    # ORDER BY if time present or user hints recency
    order_clause = f"ORDER BY {time_col} DESC" if time_col and ("newest" in q or "latest" in q or want_time) else ""

    # No LIMIT unless user says top/first explicitly
    top_n = re.search(r"\btop\s+(\d+)|\bfirst\s+(\d+)", q)
    limit_clause = f"LIMIT {int(next(g for g in top_n.groups() if g))}" if top_n else ""

    # Assemble query
    parts = [f"SELECT {select_sql}", f"FROM {table_fqn}"]
    if where_clause:
        parts.append(where_clause)
    if order_clause:
        parts.append(order_clause)
    if limit_clause:
        parts.append(limit_clause)
    return "\n".join(parts).strip()

# =========================
# SQL Guardrails (read-only, dataset-scoped, no LIMIT unless asked)
# =========================
FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|DROP|ALTER|TRUNCATE)\b", re.I)

def extract_sql(raw: str) -> str:
    # If the model used code fences, take only the fenced content
    m = re.search(r"```(?:sql)?\s*(.*?)```", raw, flags=re.I | re.S)
    sql = (m.group(1) if m else raw).strip()
    # Keep only up to the first terminating semicolon if prose follows
    parts = re.split(r";\s*(?=\S)", sql, maxsplit=1)
    sql = parts[0]
    # Normalize trailing semicolons/periods/whitespace
    sql = re.sub(r"[;.\s]+$", "", sql)
    return sql

def _requested_limit_from_question(q: str):
    ql = q.lower()

    # Patterns like "top 5", "first 5", "only 5", "last 5"
    m = re.search(r"\b(?:top|first|only|last)\s+(\d+)\b", ql)
    if m:
        return int(m.group(1))

    # Patterns like "5 records", "5 rows", "5 entries"
    m = re.search(r"\b(\d+)\s+(?:records|rows|entries)\b", ql)
    if m:
        return int(m.group(1))

    # Explicit "limit 5"
    m = re.search(r"\blimit\s+(\d+)\b", ql)
    if m:
        return int(m.group(1))

    return None

def harden_sql(raw: str, user_question: str) -> str:
    sql = extract_sql(raw)

    # Must be SELECT
    if not re.match(r"(?is)^\s*select\b", sql):
        raise ValueError("Only SELECT statements are allowed.")

    # No DDL/DML
    if FORBIDDEN.search(sql):
        raise ValueError("Forbidden DDL/DML keyword found.")

    # Enforce dataset whitelist on fully-qualified refs
    fqt = re.findall(r"`?([a-z0-9_\-]+)\.([a-z0-9_]+)\.([a-z0-9_]+)`?", sql, flags=re.I)
    for proj, ds, _tbl in fqt:
        if proj.lower() != PROJECT.lower() or ds != DATASET:
            raise ValueError(f"Unauthorized dataset reference: {proj}.{ds}")

    # Limit policy: allow only if the user asked for it; otherwise remove.
    n = _requested_limit_from_question(user_question)
    if n is not None:
        # If there's a LIMIT already, normalize it to the requested N; otherwise append one.
        sql = re.sub(r"(?is)\blimit\s+\d+(\s+offset\s+\d+)?\b", f"LIMIT {n}", sql)
        if not re.search(r"(?is)\blimit\s+\d+\b", sql):
            sql = sql.rstrip() + f" LIMIT {n}"
    else:
        # No limit requested → strip any trailing LIMIT
        sql = re.sub(r"(?is)\s+limit\s+\d+(\s+offset\s+\d+)?\s*$", "", sql).rstrip()

    # ✅ Time filter injection (always apply here)
    sql = inject_time_filter(sql, user_question)

    if not _user_requested_ordering(user_question):
        sql = _strip_trailing_order_by(sql)

    return sql

  

# =========================
# Execute & Verbalize
# =========================
def run_sql(sql: str):
    """Execute SQL and return a pandas DataFrame."""
    return bq.query(sql).result().to_dataframe()

def _pick_display_columns(df, question: str):
    lower_to_orig = {c.lower(): c for c in df.columns}
    q = question.lower()

    plate_candidates = [
        "license_plate_number", "license_number", "plate_number",
        "registration_number", "licence_number", "licence_plate_number"
    ]
    time_candidates = ["from_time", "to_time", "timestamp", "event_time", "time", "created_at"]
    color_candidates = ["car_color", "color"]

    want_plate = any(w in q for w in ["plate", "number plate", "license", "licence", "registration"])
    want_time = any(w in q for w in ["time", "from_time", "to_time", "newest", "latest"])

    cols = []
    if want_plate:
        for c in plate_candidates:
            if c in lower_to_orig:
                cols.append(lower_to_orig[c]); break
    if want_time:
        for c in time_candidates:
            if c in lower_to_orig and lower_to_orig[c] not in cols:
                cols.append(lower_to_orig[c]); break

    common_priority = plate_candidates + time_candidates + color_candidates + ["frame_nbr", "file_name"]
    for c in common_priority:
        if c in lower_to_orig and lower_to_orig[c] not in cols:
            cols.append(lower_to_orig[c])
        if len(cols) >= 3:
            break

    if not cols:
        cols = list(df.columns[:3])
    return cols


def natural_join(items):
    """English-friendly join: 'A', 'A and B', 'A, B, and C'."""
    items = [str(x) for x in items if str(x).strip() != ""]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"

def _detect_color_in_question(question: str):
    """Return a normalized color name if the user mentioned one, else None."""
    colors = [
        "silver","gray","grey","white","blue","black","red",
        "green","yellow","orange","brown","violet","purple","gold","pink"
    ]
    q = question.lower()
    for c in colors:
        if re.search(rf"\b{re.escape(c)}\b", q):
            # normalize common variants
            if c == "grey":
                return "Gray"
            return c.capitalize()
    return None



def textify(df, question: str, preview_on_single_column: bool = False) -> str:
    if df is None or df.empty:
        return "No matching records found."

    # (1) Single numeric count → say the count plainly
    if df.shape == (1, 1):
        col = df.columns[0].lower()
        val = df.iloc[0, 0]
        try:
            intval = int(val)
            if float(val) == intval:
                val = intval
        except Exception:
            pass
        if isinstance(val, (int, float)) or "count" in col:
            return f"There are {val} records matching your query."

    # (2) Single-column list → produce a friendly sentence
    if df.shape[1] == 1:
        col = df.columns[0]
        # unique list, keep original casing but sort case-insensitively
        values = (
            df[col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        values = sorted(values, key=lambda s: s.casefold())

        ql = question.lower()
        col_l = col.lower()

        # Friendly labels (singular/plural)
        def _label_sp(base_singular: str, base_plural: str):
            # If user literally typed "car id", mirror that casing/wording
            if base_plural.lower() == "car ids" and re.search(r"\bcar id\b", ql):
                return ("car id", "car ids")
            return (base_singular, base_plural)

        if "color" in col_l:
            singular, plural = ("car color", "car colors")
        elif "car_id" in col_l or "carid" in col_l or re.search(r"\bcar[_\s]?id\b", col_l):
            singular, plural = _label_sp("car ID", "car IDs")
        elif any(k in col_l for k in ["license", "licence", "plate", "registration"]):
            singular, plural = ("license plate", "license plates")
        else:
            # Fallback to the raw column name (best-effort pluralization)
            singular, plural = (col, col if col.endswith("s") else f"{col}s")

        # Optional context like "of Gray cars" if the user mentioned a color
        try:
            color_ctx = _detect_color_in_question(question)  # assumes helper present
        except NameError:
            color_ctx = None
        context_phrase = f" of {color_ctx} cars" if color_ctx and "color" not in col_l else ""

        # If the user asked for "different/unique/distinct", reflect that
        wants_distinct = any(w in ql for w in ["different", "unique", "distinct"])

        # Choose singular/plural label based on number of values
        friendly = singular if len(values) == 1 else plural
        if wants_distinct:
            friendly = f"distinct {friendly}"

        subject = f"The {friendly}{context_phrase}"
        joined = natural_join(values)
        verb = "is" if len(values) == 1 else "are"

        sentence = f"{subject} {verb} {joined}."
        if not preview_on_single_column:
            return sentence

        # Optional preview under the sentence
        lines = [f" • {col}: {v}" for v in values[:15]]
        more = "" if len(values) <= 15 else f"\n… and {len(values)-15} more."
        return sentence + "\n\nPreview:\n" + "\n".join(lines) + more

    # (3) Multi-column preview (plate + time aware) — your existing behavior
    display_cols = _pick_display_columns(df, question)
    lines = []
    for _, row in df.head(15).iterrows():
        bits = [f"{c}: {row[c]}" for c in display_cols]
        lines.append(" • " + " — ".join(bits))
    more = "" if len(df) <= 15 else f"\n… and {len(df)-15} more rows."
    return f"Found {len(df)} rows.\n" + "\n".join(lines) + more


# =========================
# UI (Ask + Reset) — nonce-based, no direct writes to widget state
# =========================
for k, v in {
    "gen_sql": None,
    "answer": None,
    "df": None,
    "show_table": False,
    "prompt_nonce": 0,          # NEW: changes the widget key on reset
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.subheader("Query")

# Use a unique widget key derived from the nonce
prompt_key = f"prompt_{st.session_state['prompt_nonce']}"
user_prompt = st.text_input(
    "Ask a question about your traffic data:",
    key=prompt_key,
    placeholder="e.g., How many silver cars are recorded? Show number plates of white cars.",
)

c1, c2 = st.columns([1, 1])
ask_clicked = c1.button("Ask", type="primary", use_container_width=True)
reset_clicked = c2.button("Reset", use_container_width=True)

with st.expander("Options / Diagnostics", expanded=False):
    st.write("Model:", f"`{PROJECT}.{LLM_DATASET}.{LLM_MODEL}`")
    st.write("Target table:", f"`{PROJECT}.{DATASET}.{TABLE}`")
    st.checkbox("Also show results table", key="show_table")

if reset_clicked:
    # Bump the nonce to give the text_input a brand-new key,
    # then clear any derived state and rerun.
    st.session_state["prompt_nonce"] += 1
    st.session_state["gen_sql"] = None
    st.session_state["answer"] = None
    st.session_state["df"] = None
    st.rerun()

if ask_clicked and user_prompt.strip():
    try:
        with st.spinner("Reading schema…"):
            schema = get_schema(DATASET)
            schema_text = schema_to_text(schema)

        with st.spinner("Generating SQL with Gemini…"):
            sql_gen = None  # <-- ensure the variable exists

            # 1) Try LLM (with retry)
            llm_text = generate_sql_with_retry(schema_text, user_prompt)
            if llm_text:
                sql_gen = harden_sql(llm_text, user_prompt)  # pass the question for LIMIT policy

            # 2) If LLM empty, try heuristic fallback
            if not sql_gen:
                fb_sql = fallback_sql_from_question(user_prompt, schema)
                if fb_sql:
                    sql_gen = harden_sql(fb_sql, user_prompt)  # pass the question for LIMIT policy

            # 3) If still nothing, bail out clearly
            if not sql_gen:
                raise RuntimeError("LLM returned empty response and no heuristic could be built.")

            # 4) Persist and execute
            st.session_state["gen_sql"] = sql_gen

        with st.spinner("Running SQL…"):
            df = run_sql(st.session_state["gen_sql"])
            st.session_state["df"] = df
            st.session_state["answer"] = textify(df, user_prompt)

        st.rerun()

    except BadRequest as e:
        st.error(f"BigQuery error: {e.message}")
    except Exception as e:
        st.error(f"Failed: {e}")



# Render last results (if any)
if st.session_state["gen_sql"]:
    st.subheader("Generated SQL")
    st.code(st.session_state["gen_sql"], language="sql")

if st.session_state["answer"]:
    st.subheader("Answer")
    st.write(st.session_state["answer"])

if st.session_state["df"] is not None and st.session_state["show_table"]:
    st.subheader("Results (preview)")
    st.dataframe(st.session_state["df"], use_container_width=True)
