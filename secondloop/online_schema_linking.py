import json
import sqlite3
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
try:
    import sqlglot
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False
    print("[WARN] sqlglot not installed. SQL extraction will use substring matching fallback.")

import sys
baseline_dir = Path(__file__).resolve().parent.parent
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))

from baseline.llm import LLMConfig, get_llm_chat_model, get_ollama_config, get_ollama_server_config
from baseline.parse import json_file, get_response_text, extract_sql


# ============================================================================
# INSTRUCTIONS (from Schema-Linking/RSL-SQL-Bird/src/configs/Instruction.py)
# ============================================================================

TABLE_AUG_INSTRUCTION = '''
You are an intelligent agent responsible for identifying the database tables involved based on the user's questions and database structure information. Your main tasks are:

1. Understand user questions: parse user questions and extract keywords and intentions.
2. Obtain database structure information: Based on the provided database structure information, understand all tables and their relationships.
3. Identify relevant tables:
   - Based on the keywords and intentions in the user's questions, identify directly related tables.
   - Consider the situation of intermediate tables, such as connection tables or cross tables, which may involve the tables in the user's questions.
4. Generate a list of tables: Integrate directly related tables and intermediate tables to form the final list of tables.
5. Return the results in json format, the format is {"tables": ["table1", "table2", ...],"columns":["table1.`column1`","table2.`column2`",...]}

### Input:
- Database structure information: including table names, fields, and relationships between tables (such as foreign keys, etc.).
- User questions: queries or questions in natural language form.

### Output:
- List of database tables involved: including directly related tables and intermediate tables.

### Operation steps:
1. Parse user questions: extract keywords and intentions from the questions.
2. Identify key tables: preliminarily identify the direct tables related to the user's questions.
3. Check intermediate tables: Based on the database structure information, identify intermediate tables related to the direct tables.
4. Integrate the results: integrate direct tables and intermediate tables to form the final list of tables.
5. Output the results: return all table lists involved in the user's questions. Select the top 15 columns most relevant to the question for each table.

### Note:
- Ensure that all possible intermediate tables are considered, especially tables involving many-to-many relationships.
- Ensure that the output table list is unique and without duplicates.

'''

SQL_GENERATION_INSTRUCTION = '''
You are a smart agent responsible for generating the correct SQL statements based on the following information:
- A small number of SQL Q&A pairs: used for reference and learning common query patterns.
- Database structure information: including table names, fields, relationships between tables (such as foreign keys, etc.).
- The first three rows of values in the table: sample data for understanding the content and data distribution of the table.
- User questions: natural language queries or questions.
- Query requirements and conditions: specific query requirements and conditions in user questions.
- Tables involved in SQL statements: tables involved in user questions.
- Auxiliary query conditions: additional query conditions provided, which may affect the generation of SQL statements.
- definition: Information for prompts, this message is very important.

Your main tasks are:

1. Parse user questions:
   - Use natural language processing (NLP) techniques to parse user questions and extract query requirements and conditions.

2. Refer to SQL Q&A pairs:
    - Use the provided SQL Q&A pairs as a reference to understand common query patterns and SQL statement structures.

3. Analyze database structure information:
    - Based on the database structure information, understand the fields and relationships of the table, and build the basic framework of the SQL statement.

4. Check sample data:
    - Analyze the data characteristics based on the first three rows of the table, which helps to determine how to construct query conditions and filter results.

5. Generate SQL statements:
    - Based on user questions, query requirements and conditions, tables involved, and auxiliary query conditions, construct complete SQL statements.

6. Verification and optimization:
    - Check whether the generated SQL statement is logical and optimize it if necessary.

### Input:
- SQL Q&A pairs: a small number of example SQL Q&A pairs.
- Database structure information: including table names, fields, relationships between tables (such as foreign keys, etc.).
- The first three rows of values in the table: sample data.
- User questions: natural language queries or questions.
- Query requirements and conditions: specific query requirements and conditions in user questions.
- Tables involved in SQL statements: tables involved in user questions.
- Auxiliary query conditions: additional query conditions.
- definition: Information for prompts, this message is very important.

### Output:
- Return the result in json format, the format is {"sql": "SQL statement that meets the user's question requirements"}

### Operation steps:
1. Parse user questions: extract query requirements and conditions from the questions.
2. Refer to SQL Q&A pairs: understand common query patterns and SQL statement structures.
3. Analyze database structure information: build the basic framework of the SQL statement.
4. Check sample data: determine query conditions and filter results.
5. Generate SQL statements: construct complete SQL statements.
6. Verification and optimization: ensure the logical correctness of the SQL statement and optimize it.

### Note:
- Ensure that the SQL statement accurately reflects the query requirements and conditions in the user questions.
- Reasonably construct query logic based on database structure and sample data.
- When generating SQL statements, consider all the information provided to ensure the correctness and efficiency of the statements.
- If the user question involves complex query requirements, please consider all requirements and conditions to generate SQL statements.

### The most important thing is to remember:
- definition: Information for prompts, this message is very important.
- In the generated SQL statement, table names and field names need to be enclosed in backticks, such as `table_name`, `column_name`.
- In the generated SQL statement, table names and field names must be correct to ensure the correctness and efficiency of the statement.
'''


@dataclass
class OnlineSchemaResult:
    """Result from online schema linking."""
    db_id: str
    question: str
    evidence: str = ""
    simplified_ddl: str = ""
    ddl_data: str = ""
    foreign_key: str = ""
    tables: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    preliminary_sql: str = ""
    enhanced_schema: str = ""
    # BIRD format
    column_meaning: Dict[str, str] = field(default_factory=dict)
    column_meaning_text: str = ""
    # Spider format (includes type and sample values)
    column_info: Dict[str, Dict] = field(default_factory=dict)
    column_info_text: str = ""
    # Few-shot examples (formatted text matching offline format)
    example: str = ""
    # Latency tracking for each step
    latency: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "db": self.db_id,
            "question": self.question,
            "evidence": self.evidence,
            "simplified_ddl": self.simplified_ddl,
            "ddl_data": self.ddl_data,
            "foreign_key": self.foreign_key,
            "tables": self.tables,
            "columns": self.columns,
            "preliminary_sql": self.preliminary_sql,
            "column_meaning": self.column_meaning,
            "column_info": self.column_info,
            "latency": self.latency,
        }

    def to_ppl_dict(self) -> Dict[str, Any]:
        """
        Convert to PPL-compatible dict for use with build_enhanced_schema_text().

        This enables online schema linking results to use the same formatting
        code path as offline schema linking (ppl_dev.json through ppl_loader_new.py).
        """
        return {
            "db": self.db_id,
            "db_id": self.db_id,
            "question": self.question,
            "evidence": self.evidence,
            "simplified_ddl": self.simplified_ddl,
            "ddl_data": self.ddl_data,
            "foreign_key": self.foreign_key,
            "tables": self.tables,
            "columns": self.columns,
            "column_meaning_text": self.column_meaning_text,
            "column_meaning": self.column_meaning,
            "column_info": self.column_info,
            "example": self.example,  # Few-shot examples (populated by FewShotExampleRetriever)
        }


class OnlineSchemaLinker:
    """
    Online schema linking for BIRD/Spider datasets.

    Performs schema linking on-the-fly for each new question.
    """

    def __init__(self, dataset: str = "bird", llm_config: Optional[LLMConfig] = None, db_root: Optional[str] = None, split: str = "dev", k_shot: int = 3):
        """
        Initialize online schema linker.

        Args:
            dataset: "bird" or "spider"
            llm_config: LLM configuration (uses default if None)
            db_root: Root path to databases (auto-detected if None)
            split: Dataset split (dev, train, etc.)
            k_shot: Number of few-shot examples to retrieve (default 3)
        """
        self.dataset = dataset.lower()
        self.split = split
        self.k_shot = k_shot
        self.llm_config = llm_config or LLMConfig()
        self.llm = get_llm_chat_model(self.llm_config)
        self.project_root = Path(__file__).resolve().parent.parent

        # Set database root path
        if db_root:
            self.db_root = Path(db_root)
        else:
            if self.dataset == "bird":
                self.db_root = self.project_root / "Data" / "BIRD" / split / f"{split}_databases"
            else:
                if split == "test":
                    self.db_root = self.project_root / "Data" / "spider_data" / "test_database"
                else:
                    self.db_root = self.project_root / "Data" / "spider_data" / "database"

        # Load column meanings (for BIRD dataset)
        self.column_meaning_index = self.load_column_meanings()
        # Load Spider samples (for Spider dataset)
        self.spider_samples_index = self.load_spider_samples()

        # Initialize few-shot example retriever (loaded once, reused for all questions)
        self.few_shot_retriever = None
        if k_shot > 0:
            self._init_few_shot_retriever()

    def _init_few_shot_retriever(self):
        """Initialize the few-shot example retriever (lazy load)."""
        try:
            from few_shot_retriever import FewShotExampleRetriever
            self.few_shot_retriever = FewShotExampleRetriever(
                dataset=self.dataset,
                k_shot=self.k_shot
            )
        except ImportError:
            # Try relative import
            try:
                from .few_shot_retriever import FewShotExampleRetriever
                self.few_shot_retriever = FewShotExampleRetriever(
                    dataset=self.dataset,
                    k_shot=self.k_shot
                )
            except ImportError as e:
                print(f"[WARN] Could not load FewShotExampleRetriever: {e}")
                print("[WARN] Few-shot examples will not be available")
                self.few_shot_retriever = None
        except Exception as e:
            print(f"[WARN] Error initializing FewShotExampleRetriever: {e}")
            self.few_shot_retriever = None

    def get_few_shot_examples(self, question: str) -> str:
        """
        Get formatted few-shot examples for a question.

        Args:
            question: The question to find similar examples for

        Returns:
            Formatted string with k similar examples, or empty string if unavailable
        """
        if self.few_shot_retriever is None:
            return ""
        try:
            return self.few_shot_retriever.get_formatted_examples(question)
        except Exception as e:
            print(f"[WARN] Error getting few-shot examples: {e}")
            return ""

    def load_spider_samples(self) -> Dict[tuple, Dict]:
        """
        Load spider_dev_samples.json and build an index for fast lookup.

        Returns:
            Dict mapping (db_id, normalized_question) -> sample record
        """
        if self.dataset != "spider":
            return {}

        # Try to find spider_dev_samples.json
        samples_path = self.project_root / "Data" / "spider_data" / "spider_dev_samples.json"

        if not samples_path.exists():
            print(f"[WARN] spider_dev_samples.json not found at {samples_path}")
            return {}

        try:
            with open(samples_path, 'r', encoding='utf-8') as f:
                samples_data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load spider_dev_samples.json: {e}")
            return {}

        # Build index: {(db_id, normalized_question): sample_record}
        index = {}
        for item in samples_data:
            db_id = item.get('db_id', '')
            question = self.normalize_question(item.get('question', ''))
            key = (db_id, question)
            index[key] = item

        print(f"Loaded Spider samples for {len(index)} questions")
        return index

    def normalize_question(self, question: str) -> str:
        """Normalize question for matching."""
        return ' '.join(question.lower().split())

    def get_spider_sample(self, db_id: str, question: str) -> Optional[Dict]:
        """Get Spider sample by db_id and question."""
        key = (db_id, self.normalize_question(question))
        return self.spider_samples_index.get(key)

    def load_column_meanings(self) -> Dict[str, Dict[str, str]]:
        """
        Load column_meaning.json and build an index for fast lookup.

        Returns:
            Dict mapping db_id -> {table.column: description}
        """
        if self.dataset != "bird":
            return {}

        # Try to find column_meaning.json
        column_meaning_path = self.project_root / "Data" / "BIRD" / self.split / "column_meaning.json"

        if not column_meaning_path.exists():
            print(f"[WARN] column_meaning.json not found at {column_meaning_path}")
            return {}

        try:
            with open(column_meaning_path, 'r', encoding='utf-8') as f:
                column_meaning_data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load column_meaning.json: {e}")
            return {}

        # Build index: {db_id: {table.column: description}}
        index = {}
        for key, description in column_meaning_data.items():
            parts = key.split('|')
            if len(parts) == 3:
                db_id, table, column = parts
                if db_id not in index:
                    index[db_id] = {}
                index[db_id][f"{table}.{column}"] = description

        print(f"Loaded column meanings for {len(index)} databases")
        return index

    def get_column_meanings_for_db(self, db_id: str) -> Dict[str, str]:
        """Get all column meanings for a database."""
        return self.column_meaning_index.get(db_id, {})

    def get_relevant_column_meanings(self, db_id: str, columns: List[str]) -> Dict[str, str]:
        """
        Get column meanings relevant to the schema-linked columns.

        Args:
            db_id: Database ID
            columns: List of schema-linked columns (format: table.`column`)

        Returns:
            Dict mapping table.column -> description for relevant columns
        """
        if db_id not in self.column_meaning_index:
            return {}

        db_meanings = self.column_meaning_index[db_id]
        relevant_meanings = {}

        for col in columns:
            # Handle formats like: table.`column` or `table`.`column`
            normalized = col.replace('`', '').lower()

            # Try to find a match in the meanings
            for meaning_key, description in db_meanings.items():
                meaning_key_lower = meaning_key.lower()
                if normalized == meaning_key_lower or normalized.endswith('.' + meaning_key_lower.split('.')[-1]):
                    relevant_meanings[meaning_key] = description
                    break

        return relevant_meanings

    def format_column_meaning_text(self, column_meaning: Dict[str, str]) -> str:
        """
        Format column meanings into readable text for schema.
        Groups by table for better organization.
        """
        if not column_meaning:
            return ""

        lines = ["### The meaning of every column:"]
        lines.append("#")

        # Group by table
        tables = {}
        for col_key, description in column_meaning.items():
            parts = col_key.split('.')
            if len(parts) == 2:
                table, col = parts
            else:
                table, col = 'unknown', col_key

            if table not in tables:
                tables[table] = []

            # Clean up description - remove leading # if present
            desc = description.strip()
            if desc.startswith('#'):
                desc = desc[1:].strip()

            # Truncate very long descriptions
            if len(desc) > 200:
                desc = desc[:200] + "..."

            tables[table].append(f"# {col}: {desc}")

        for table, cols in sorted(tables.items()):
            lines.append(f"# [{table}]")
            lines.extend(cols)

        lines.append("#")
        return '\n'.join(lines)

    def format_column_info_text(self, column_info: Dict[str, Dict]) -> str:
        """
        Format column_info (Spider format) into readable text.
        Shows column types and sample values.

        Args:
            column_info: Dict mapping table.column -> {type, sample_values}

        Returns:
            Formatted string showing column types and sample values
        """
        if not column_info:
            return ""

        lines = ["### Column Info (type and sample values):"]

        # Group by table
        tables = {}
        for col_key, info in column_info.items():
            parts = col_key.split('.')
            if len(parts) == 2:
                table, col = parts
            else:
                table, col = 'unknown', col_key

            if table not in tables:
                tables[table] = []

            col_type = info.get('type', 'unknown')
            samples = info.get('sample_values', [])
            sample_str = ', '.join(str(s) for s in samples[:5])  # Limit to 5 samples

            tables[table].append(f"#   - {col} ({col_type}): [{sample_str}]")

        for table, cols in sorted(tables.items()):
            lines.append(f"# {table}:")
            lines.extend(cols)

        return '\n'.join(lines)

    def connect_to_db(self, db_id: str) -> sqlite3.Connection:
        db_path = self.db_root / db_id/ f"{db_id}.sqlite"
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        return sqlite3.connect(str(db_path))
    
    def get_all_table_names(self, db_id: str) -> List[str]:
        conn = self.connect_to_db(db_id)
        cursor = conn.cursor()
        cursor.execute("SELECT name From sqlite_master WHERE type='table' AND name != 'sqlite_sequence';")
        table_names = cursor.fetchall()
        conn.close()
        return [name[0] for name in table_names]
    
    def get_all_column_names(self, db_id: str, table_name: str) -> List[str]:
        conn = self.connect_to_db(db_id)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        table_info = cursor.fetchall()
        column_names = [column[1] for column in table_info]
        conn.close()
        return column_names

    def get_table_infos(self, db_id: str) -> str:
        """Get simplified DDL for all tables."""
        table_list = self.get_all_table_names(db_id)
        table_str = '#\n# '
        for table in table_list:
            column_list = self.get_all_column_names(db_id, table)
            column_list = ['`' + column + '`' for column in column_list]
            columns_str = f'{table}(' + ', '.join(column_list) + ')'
            table_str += columns_str + '\n# '
        return table_str

    def get_foreign_key_info(self, db_id: str, table_name: str) -> List[Tuple]:
        conn = self.connect_to_db(db_id)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA foreign_key_list('{table_name}');")
        foreign_key_info = cursor.fetchall()
        conn.close()
        return foreign_key_info

    def get_foreign_key_infos(self, db_id: str) -> str:
        """Get all foreign key information."""
        table_list = self.get_all_table_names(db_id)
        foreign_str = '#\n# '
        for table in table_list:
            foreign_lists = self.get_foreign_key_info(db_id, table)
            for foreign in foreign_lists:
                foreign_one = f'{table}({foreign[3]}) references {foreign[2]}({foreign[4]})'
                foreign_str += foreign_one + '\n# '
        return foreign_str

    def get_sample_data(self, db_id: str) -> str:
        """Get first 3 rows of data from each table."""
        conn = self.connect_to_db(db_id)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cur.fetchall()

        simplified_ddl_data = []
        for table in tables:
            table_name = table[0]
            if table_name == 'sqlite_sequence':
                continue
            cur.execute(f"select * from `{table_name}`")
            col_name_list = [desc[0] for desc in cur.description]

            db_data_all = []
            for i in range(3):
                row = cur.fetchone()
                if row:
                    db_data_all.append(row)

            if not db_data_all:
                continue

            test = ""
            for idx, column_data in enumerate(col_name_list):
                try:
                    values = []
                    for row in db_data_all:
                        if idx < len(row):
                            values.append(str(row[idx]))
                    if values:
                        test += f"`{column_data}`[{','.join(values)}],"
                except:
                    pass
            if test:
                simplified_ddl_data.append(f"{table_name}({test[:-1]})")

        conn.close()
        ddls_data = "# " + ";\n# ".join(simplified_ddl_data) + ";\n" if simplified_ddl_data else ""
        return ddls_data

    def get_db_schema(self, db_id: str) -> List[str]:
        """Get full schema (table.column) for database."""
        conn = self.connect_to_db(db_id)
        cursor = conn.cursor()
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()

        schema = []
        for table in tables:
            table_name = table[0]
            if table_name == 'sqlite_sequence':
                continue
            columns = cursor.execute(f"PRAGMA table_info('{table_name}');").fetchall()
            for column in columns:
                schema.append(f"{table_name}.{column[1]}")

        conn.close()
        return schema

    def call_llm_json(self, system_prompt: str, user_prompt: str, required_keys: Optional[List[str]] = None, allow_sql_fallback: bool = False) -> Dict[str, Any]:
        """
        Call LLM and parse JSON response robustly.

        Notes:
        - Mirrors offline Schema-Linking behavior by first attempting API-level
          JSON mode (`response_format={"type":"json_object"}`), then falling
          back to normal calls when backend/provider doesn't support it.
        - For SQL generation calls, can fall back to extracting plain SQL text
          when model fails to emit JSON.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_err = None
        last_content = ""
        for attempt in range(3):
            try:
                retry_messages = list(messages)
                if attempt > 0:
                    retry_messages.append({
                        "role": "user",
                        "content": 'Return ONLY a valid JSON object. No markdown fences, no explanations.',
                    })

                # Offline RSL-SQL uses OpenAI JSON mode.
                # Try the same first; fall back gracefully if unsupported.
                try:
                    response = self.llm.invoke(
                        retry_messages,
                        response_format={"type": "json_object"},
                    )
                except Exception:
                    response = self.llm.invoke(retry_messages)
                content = get_response_text(response).strip()
                last_content = content
                if not content:
                    raise ValueError("Empty model response")

                parsed = json_file(content, aggressive_mode=False)
                if not isinstance(parsed, dict):
                    raise ValueError("Parsed response is not a JSON object")

                # baseline.parse.json_file returns parse_error marker on hard failures
                if parsed.get("parse_error"):
                    raise ValueError(str(parsed.get("parse_error")))

                if required_keys:
                    missing = [k for k in required_keys if k not in parsed]
                    if missing:
                        if allow_sql_fallback:
                            sql = extract_sql(content).strip()
                            if sql:
                                return {"sql": sql}
                        raise ValueError(f"Missing required keys: {missing}")

                if allow_sql_fallback:
                    sql_val = parsed.get("sql")
                    if isinstance(sql_val, str) and sql_val.strip():
                        parsed["sql"] = sql_val.strip()
                        return parsed
                    sql = extract_sql(content).strip()
                    if sql:
                        return {"sql": sql}

                return parsed
            except Exception as e:
                last_err = e
                # If SQL generation returned plain SQL instead of JSON, salvage it.
                if allow_sql_fallback and last_content:
                    sql = extract_sql(last_content).strip()
                    if sql:
                        return {"sql": sql}
                if attempt == 2:
                    print(f"LLM JSON parse failed after 3 attempts: {e}")
                    return {}
                continue

        if last_err is not None:
            print(f"LLM JSON parse failed after 3 attempts: {last_err}")
        return {}

    def _clip_columns_per_table(self, columns: List[str], max_per_table: int = 15) -> List[str]:
        """
        Enforce maximum columns per table to prevent LLM over-selection.

        The TABLE_AUG_INSTRUCTION asks for "top 15 columns most relevant to the
        question for each table", but weaker models may ignore this and return
        all columns.  This clips programmatically as a safety net.
        """
        table_counts: Dict[str, int] = {}
        clipped: List[str] = []

        for col in columns:
            # Parse table name from "table.`column`" or "table.column"
            parts = col.split('.')
            if len(parts) >= 2:
                table = parts[0].strip().replace('`', '').lower()
            else:
                table = '__unknown__'

            table_counts[table] = table_counts.get(table, 0) + 1
            if table_counts[table] <= max_per_table:
                clipped.append(col)

        if len(clipped) < len(columns):
            dropped = len(columns) - len(clipped)
            print(f"[WARN] Column guardrail: clipped {dropped} columns (max {max_per_table}/table)")

        return clipped

    def table_column_selection(self, table_info: str, question: str, evidence: str = "") -> Dict[str, List[str]]:
        """Use LLM to identify relevant tables and columns."""
        prompt = table_info.strip()
        if evidence:
            prompt += '\n\n### definition: ' + evidence
        prompt += "\n### Question: " + question

        result = self.call_llm_json(
            TABLE_AUG_INSTRUCTION,
            prompt,
            required_keys=["tables", "columns"],
        )

        tables = result.get("tables", [])
        columns = result.get("columns", [])

        # --- Validate: keep only string items, drop malformed dicts/ints ---
        tables = [t for t in tables if isinstance(t, str) and t.strip()]
        columns = [c for c in columns if isinstance(c, str) and '.' in c]

        # --- Guardrail: enforce max 15 columns per table ---
        columns = self._clip_columns_per_table(columns, max_per_table=15)

        return {
            "tables": tables,
            "columns": columns,
        }

    def generate_preliminary_sql(self, table_info: str, table_column: Dict[str, List[str]], question: str, evidence: str = "", example: str = "") -> str:
        """Generate preliminary SQL for schema extraction."""
        enhanced_info = table_info
        enhanced_info += f'\n### tables: {table_column["tables"]}\n'
        enhanced_info += f'### columns: {table_column["columns"]}\n'

        prompt = ""
        if example:
            prompt = example.strip() + "\n\n"
        prompt += "### Return ONLY valid JSON in this exact format: {\"sql\": \"<sqlite SQL query>\"}. No markdown, no explanation.\n"
        prompt += "### The SQL must minimize execution time while ensuring correctness.\n"
        prompt += enhanced_info.strip()
        if evidence:
            prompt += '\n\n### definition: ' + evidence
        prompt += "\n### Question: " + question

        result = self.call_llm_json(
            SQL_GENERATION_INSTRUCTION,
            prompt,
            required_keys=["sql"],
            allow_sql_fallback=True,
        )
        sql = result.get("sql", "")
        return sql.replace('\n', ' ') if sql else ""

    # -----------------------------------------------------------------
    # Schema extraction from SQL  (sqlglot AST → substring fallback)
    # -----------------------------------------------------------------

    def extract_schema_from_sql(self, sql: str, db_id: str) -> Tuple[List[str], List[str]]:
        """
        Extract tables and columns actually referenced in a SQL statement.

        Uses sqlglot (same library as offline util.py::extract_tables_and_columns)
        to parse the SQL into an AST and pull out real table/column references.
        Falls back to substring matching only when sqlglot is unavailable or
        the SQL cannot be parsed (e.g. syntax errors from a weak LLM).
        """
        if HAS_SQLGLOT:
            try:
                return self._extract_schema_sqlglot(sql, db_id)
            except Exception as e:
                print(f"[WARN] sqlglot parse failed ({e}), falling back to substring matching")
        return self._extract_schema_substring(sql, db_id)

    def _extract_schema_sqlglot(self, sql: str, db_id: str) -> Tuple[List[str], List[str]]:
        """
        AST-based extraction (matches offline util.py::extract_tables_and_columns).

        Advantages over substring matching:
        - "school" won't match just because "schools" appears as a table name
        - "street" won't match just because "mailstreet" is referenced
        - Only columns that are *actually referenced* in the SQL are returned
        """
        parsed = sqlglot.parse_one(sql, read="sqlite")

        # Collect referenced identifiers from the AST
        sql_table_names = {
            t.name.lower()
            for t in parsed.find_all(sqlglot.exp.Table)
        }
        sql_column_names = {
            c.alias_or_name.lower()
            for c in parsed.find_all(sqlglot.exp.Column)
        }

        # SELECT * means every column of the referenced tables
        has_star = bool(list(parsed.find_all(sqlglot.exp.Star)))

        # Match against actual DB schema
        db_schema = self.get_db_schema(db_id)
        pred_columns_set: set = set()

        for item in db_schema:
            parts = item.split('.')
            if len(parts) != 2:
                continue
            table, column = parts
            if table.lower() == 'sqlite_sequence':
                continue

            # SELECT * → include all columns of referenced tables
            if has_star and table.lower() in sql_table_names:
                pred_columns_set.add(item.lower())
                continue

            # Normal match: column name found in AST-extracted columns
            if column.lower() in sql_column_names:
                pred_columns_set.add(item.lower())

        pred_columns = list(pred_columns_set)

        # Format columns with backticks
        pred_columns = [item.replace('.', '.`') + '`' for item in pred_columns]

        # Extract unique tables
        tables = list(set(item.split('.')[0] for item in pred_columns))

        return tables, pred_columns

    def _extract_schema_substring(self, sql: str, db_id: str) -> Tuple[List[str], List[str]]:
        """Fallback: substring matching (same as offline bid_schema_linking.py)."""
        db_schema = self.get_db_schema(db_id)
        db_schema_lower = [item.lower() for item in db_schema]

        sql_lower = sql.lower()
        pred_columns = []

        for item in db_schema_lower:
            table = item.split('.')[0]
            if table == 'sqlite_sequence':
                continue
            column = item.split('.')[1]
            if column in sql_lower:
                pred_columns.append(item)

        pred_columns = [item.replace('.', '.`') + '`' for item in pred_columns]
        tables = list(set(item.split('.')[0] for item in pred_columns))

        return tables, pred_columns

    # -----------------------------------------------------------------
    # Schema extraction from evidence  (substring – unchanged from offline)
    # -----------------------------------------------------------------

    def extract_schema_from_evidence(self, evidence: str, db_id: str) -> Tuple[List[str], List[str]]:
        """Extract tables and columns from evidence by matching against schema."""
        if not evidence:
            return [], []

        db_schema = self.get_db_schema(db_id)
        db_schema_lower = [item.lower() for item in db_schema]

        evidence_lower = evidence.lower()
        pred_columns = []

        for item in db_schema_lower:
            table = item.split('.')[0]
            if table == 'sqlite_sequence':
                continue
            column = item.split('.')[1]
            if column in evidence_lower:
                pred_columns.append(item)

        # Format columns with backticks
        pred_columns = [item.replace('.', '.`') + '`' for item in pred_columns]

        # Extract unique tables
        tables = list(set([item.split('.')[0] for item in pred_columns]))

        return tables, pred_columns

    def merge_and_filter_schema(self, llm_result: Dict[str, List[str]], sql_tables: List[str], sql_columns: List[str], hint_tables: List[str], hint_columns: List[str], db_id: str) -> Tuple[List[str], List[str]]:
        """Merge schema from all sources and filter against actual database."""
        # Merge all tables and columns
        # Helper to safely convert to string
        def safe_str(item):
            if isinstance(item, str):
                return item
            elif isinstance(item, dict):
                # LLM sometimes returns dicts - try to extract meaningful value
                return str(item.get('name', item.get('column', item.get('table', str(item)))))
            else:
                return str(item)

        all_tables = (
            llm_result.get("tables", []) +
            sql_tables +
            hint_tables
        )
        all_columns = (
            llm_result.get("columns", []) +
            sql_columns +
            hint_columns
        )

        # Normalize to lowercase and deduplicate (handle non-string items)
        all_tables = list(set([safe_str(t).lower() for t in all_tables if t]))
        all_columns = list(set([safe_str(c).lower() for c in all_columns if c]))

        # Filter against actual database schema
        db_schema = self.get_db_schema(db_id)
        db_schema_clean = [obj.replace('`', '') for obj in db_schema]

        filtered_columns = []
        for obj in db_schema_clean:
            obj_lower = obj.lower()
            # Check if column (with or without backticks) is in our list
            obj_formatted = obj_lower.replace('.', '.`') + '`'
            if obj_lower in all_columns or obj_formatted in all_columns:
                if obj_lower not in [c.lower().replace('`', '') for c in filtered_columns]:
                    filtered_columns.append(obj)

        # Format with backticks
        filtered_columns = [item.replace('.', '.`') + '`' for item in filtered_columns]

        # Extract tables from filtered columns
        filtered_tables = list(set([item.split('.')[0] for item in filtered_columns]))

        # Also include tables from all_tables that exist in DB
        db_tables = self.get_all_table_names(db_id)
        db_tables_lower = [t.lower() for t in db_tables]
        for t in all_tables:
            if t.lower() in db_tables_lower and t not in filtered_tables:
                # Get proper case from DB
                for db_t in db_tables:
                    if db_t.lower() == t.lower():
                        filtered_tables.append(db_t)
                        break

        # --- Final safety-net: clip columns after merge ---
        # This catches cases where a weak LLM returned too many columns AND
        # the SQL/evidence extraction added more on top.  Use a slightly
        # higher limit (20) than the LLM guardrail (15) to preserve the
        # extra columns legitimately found via SQL/evidence extraction.
        filtered_columns = self._clip_columns_per_table(filtered_columns, max_per_table=20)

        # Recompute tables from the (possibly clipped) column list
        filtered_tables = list(set([item.split('.')[0] for item in filtered_columns]))
        # Re-add tables that were in all_tables but had no columns
        for t in all_tables:
            if t.lower() in db_tables_lower:
                already = [ft.lower() for ft in filtered_tables]
                if t.lower() not in already:
                    for db_t in db_tables:
                        if db_t.lower() == t.lower():
                            filtered_tables.append(db_t)
                            break

        # Add key/FK columns for linked tables to reduce downstream join failures.
        filtered_columns = self._augment_with_key_and_fk_columns(
            db_id=db_id,
            tables=filtered_tables,
            columns=filtered_columns,
        )

        # Keep schema concise after augmentation.
        filtered_columns = self._clip_columns_per_table(filtered_columns, max_per_table=25)

        # Recompute tables from final columns (plus tables explicitly linked by name)
        filtered_tables = list(set([item.split('.')[0] for item in filtered_columns]))
        for t in all_tables:
            if t.lower() in db_tables_lower:
                already = [ft.lower() for ft in filtered_tables]
                if t.lower() not in already:
                    for db_t in db_tables:
                        if db_t.lower() == t.lower():
                            filtered_tables.append(db_t)
                            break

        return filtered_tables, filtered_columns

    def _augment_with_key_and_fk_columns(
        self,
        db_id: str,
        tables: List[str],
        columns: List[str],
    ) -> List[str]:
        """
        Add primary-key and foreign-key columns for selected tables.
        This provides join-critical columns even when preliminary SQL generation fails.
        """
        if not tables:
            return columns

        db_schema = self.get_db_schema(db_id)
        schema_map = {item.lower(): item for item in db_schema if "." in item}

        def fmt_col(table_name: str, col_name: str) -> Optional[str]:
            key = f"{table_name}.{col_name}".lower()
            proper = schema_map.get(key)
            if not proper:
                return None
            return proper.replace(".", ".`") + "`"

        result_cols = list(columns or [])
        existing = set(c.lower().replace("`", "") for c in result_cols)
        tables_lc = {t.lower() for t in tables}

        conn = self.connect_to_db(db_id)
        cur = conn.cursor()
        try:
            for table in tables:
                table_q = table.replace("'", "''")

                # Add primary key columns
                for row in cur.execute(f"PRAGMA table_info('{table_q}')").fetchall():
                    # row: cid, name, type, notnull, dflt_value, pk
                    col_name = row[1]
                    is_pk = row[5]
                    if is_pk:
                        col = fmt_col(table, col_name)
                        if col:
                            norm = col.lower().replace("`", "")
                            if norm not in existing:
                                result_cols.append(col)
                                existing.add(norm)

                # Add FK source + target columns when both sides are in selected tables
                for fk in cur.execute(f"PRAGMA foreign_key_list('{table_q}')").fetchall():
                    # fk: id, seq, table, from, to, on_update, on_delete, match
                    ref_table = str(fk[2])
                    from_col = str(fk[3])
                    to_col = str(fk[4])

                    if table.lower() in tables_lc:
                        col = fmt_col(table, from_col)
                        if col:
                            norm = col.lower().replace("`", "")
                            if norm not in existing:
                                result_cols.append(col)
                                existing.add(norm)

                    if ref_table.lower() in tables_lc:
                        col = fmt_col(ref_table, to_col)
                        if col:
                            norm = col.lower().replace("`", "")
                            if norm not in existing:
                                result_cols.append(col)
                                existing.add(norm)
        finally:
            conn.close()

        return result_cols

    # =========================================================================
    # SIMPLIFIED SCHEMA METHODS (matching offline RSL-SQL Step 2)
    # =========================================================================

    def build_table_columns_dict(self, tables: List[str], columns: List[str]) -> Dict[str, List[str]]:
        """
        Build a mapping from table name to list of column names.

        Args:
            tables: List of table names
            columns: List of columns in format "table.`column`"

        Returns:
            Dict mapping table -> [column1, column2, ...]
        """
        table_columns = {table: [] for table in tables}

        for col in columns:
            # Parse "table.`column`" format
            parts = col.split('.')
            if len(parts) == 2:
                table = parts[0].strip()
                column = parts[1].strip()
                # Find matching table (case-insensitive)
                for t in tables:
                    if t.lower() == table.lower():
                        if column not in table_columns[t]:
                            table_columns[t].append(column)
                        break

        return table_columns

    def create_simplified_ddl(self, tables: List[str], columns: List[str]) -> str:
        """
        Create simplified DDL containing only linked tables and columns.
        Replicates simplified_schema.py::simplified() DDL format.

        Args:
            tables: List of linked table names
            columns: List of linked columns in format "table.`column`"

        Returns:
            Simplified DDL string in format: #\n# table(col1, col2)\n# table2(col1)...
        """
        simple_ddl = "#\n# "

        for table in tables:
            simple_ddl += table + "("
            column_list = []

            for column in columns:
                parts = column.split('.')
                if len(parts) == 2:
                    col_table = parts[0].strip()
                    col_name = parts[1].strip()
                    if col_table.lower() == table.lower():
                        column_list.append(col_name)

            if column_list:
                simple_ddl += ",".join(column_list)

            simple_ddl += ")\n# "

        return simple_ddl.strip()

    def get_simplified_sample_data(self, db_id: str, tables: List[str], table_columns: Dict[str, List[str]]) -> str:
        """
        Get sample data for only linked tables and columns.
        Replicates util.py::simple_throw_row_data().

        Args:
            db_id: Database ID
            tables: List of linked table names
            table_columns: Dict mapping table -> list of column names

        Returns:
            Sample data string in format: # table(col1[v1,v2,v3],col2[v1,v2,v3]);
        """
        conn = self.connect_to_db(db_id)
        cur = conn.cursor()

        simplified_ddl_data = []

        for table in tables:
            col_name_list = table_columns.get(table, [])
            if not col_name_list:
                continue

            # Clean column names (remove backticks for query)
            clean_cols = [c.replace('`', '') for c in col_name_list]
            column_str = ",".join([f"`{c}`" for c in clean_cols])

            try:
                cur.execute(f"SELECT {column_str} FROM `{table}` LIMIT 3")
                db_data_all = cur.fetchall()

                if not db_data_all:
                    continue

                test = ""
                for idx, column_data in enumerate(col_name_list):
                    try:
                        values = []
                        for row in db_data_all:
                            if idx < len(row):
                                values.append(str(row[idx]))
                        if values:
                            test += f"{column_data}[{','.join(values)}],"
                    except:
                        pass

                if test:
                    simplified_ddl_data.append(f"{table}({test[:-1]})")
            except Exception as e:
                # Skip tables that cause errors
                continue

        conn.close()

        if simplified_ddl_data:
            return "#\n# " + ";\n# ".join(simplified_ddl_data) + ";\n# "
        return "#\n# "

    def get_simplified_foreign_keys(self, db_id: str, tables: List[str]) -> str:
        """
        Get foreign key relationships only between linked tables.
        Replicates simplified_schema.py::simplified() FK filtering.

        Args:
            db_id: Database ID
            tables: List of linked table names

        Returns:
            Filtered FK string containing only relationships between linked tables
        """
        # Get all FK info first
        full_fk = self.get_foreign_key_infos(db_id)

        # Normalize table names for comparison
        tables_lower = [t.lower() for t in tables]

        # Filter FKs to only those between linked tables
        filtered_lines = ["#"]

        for line in full_fk.split("\n"):
            line = line.strip()
            if not line or line == "#":
                continue

            try:
                # Parse line like: "# table1(col) references table2(col)"
                if "references" in line:
                    # Extract table names
                    line_content = line.lstrip("# ").strip()
                    table1 = line_content.split("(")[0].strip()
                    table2 = line_content.split("references")[1].strip().split("(")[0].strip()

                    # Check if both tables are in our linked tables
                    if table1.lower() in tables_lower and table2.lower() in tables_lower:
                        filtered_lines.append("# " + line_content)
            except:
                continue

        filtered_lines.append("# ")
        return "\n".join(filtered_lines)

    def get_filtered_column_meanings(self, db_id: str, tables: List[str], columns: List[str]) -> str:
        """
        Get column meanings for only linked columns.
        Replicates simplified_schema.py::explanation_collection().

        Args:
            db_id: Database ID
            tables: List of linked table names
            columns: List of linked columns in format "table.`column`"

        Returns:
            Column meanings string in format: # table.column: meaning
        """
        if db_id not in self.column_meaning_index:
            return ""

        db_meanings = self.column_meaning_index[db_id]

        # Normalize columns for matching
        columns_normalized = [obj.replace('`', '').lower() for obj in columns]
        columns_formatted = [obj.replace('.', '.`') + '`' for obj in columns_normalized]
        tables_lower = [t.lower() for t in tables]

        explanation = ""

        for key, meaning in db_meanings.items():
            # key is in format "table.column"
            parts = key.split('.')
            if len(parts) != 2:
                continue

            table_name = parts[0].lower()
            column_name = parts[1].lower()

            # Check if table is in linked tables
            if table_name not in tables_lower:
                continue

            # Check if column is in linked columns
            col_check = f"{table_name}.`{column_name}`"
            if col_check in columns_formatted:
                explanation += f"# {key}: {meaning}\n"

        return explanation

    def build_enhanced_schema(self, result: OnlineSchemaResult) -> str:
        """
        Build enhanced schema text for agents.
        Matches offline table_info_construct() format from step_2_information_augmentation.py.
        """
        parts = [
            "### Sqlite SQL tables, with their properties:",
            result.simplified_ddl if result.simplified_ddl else "#\n# ",
            "\n### Here are some data information about database references.",
            result.ddl_data if result.ddl_data else "#\n# ",
            "\n### Foreign key information of Sqlite SQL tables, used for table joins:",
            result.foreign_key if result.foreign_key else "#\n# ",
            "\n### The meaning of every column:",
            "#",
            result.column_meaning_text.strip() if result.column_meaning_text else "",
            "#",
        ]

        # Column info with types and sample values (Spider dataset format)
        if result.column_info_text:
            parts.append("\n" + result.column_info_text)

        # Evidence/Definition (BIRD) - matches offline format
        if result.evidence:
            parts.append("\n### definition: " + result.evidence)

        return "\n".join(parts)

    def process_question(self, question: str, db_id: str, evidence: str = "", example: str = "", generate_preliminary: bool = True) -> OnlineSchemaResult:
        """
        Process a new question with online schema linking.

        Follows the offline RSL-SQL pipeline:
        - Phase 1 (Steps 1-5): Use FULL schema for LLM selection and preliminary SQL
        - Phase 2 (Steps 6-8): Create SIMPLIFIED schema with only linked tables/columns

        Args:
            question: Natural language question
            db_id: Database ID
            evidence: Domain knowledge/evidence (BIRD dataset)
            example: Few-shot SQL examples
            generate_preliminary: Whether to generate preliminary SQL for extraction

        Returns:
            OnlineSchemaResult with all schema information including step latencies
        """
        total_start = time.perf_counter()
        latency = {}

        result = OnlineSchemaResult(
            db_id=db_id,
            question=question,
            evidence=evidence,
        )

        # =====================================================================
        # PHASE 1: Use FULL schema for LLM selection (matching offline Step 1)
        # =====================================================================

        # Step 1: Get FULL database structure for LLM selection
        step_start = time.perf_counter()
        full_ddl = self.get_table_infos(db_id)
        full_sample_data = self.get_sample_data(db_id)
        full_foreign_key = self.get_foreign_key_infos(db_id)
        latency["db_structure_sec"] = time.perf_counter() - step_start

        # Step 2: Build FULL table info for LLM
        table_info = (
            '### Sqlite SQL tables, with their properties:\n' + full_ddl +
            '\n### Here are some data information about database references.\n' + full_sample_data +
            '\n### Foreign key information of Sqlite SQL tables, used for table joins:\n' + full_foreign_key
        )

        # Step 3: LLM-based table/column selection (using FULL schema)
        step_start = time.perf_counter()
        llm_selection = self.table_column_selection(table_info, question, evidence)
        latency["table_column_selection_sec"] = time.perf_counter() - step_start

        # Step 4: Generate preliminary SQL and extract schema from it
        sql_tables, sql_columns = [], []
        if generate_preliminary:
            step_start = time.perf_counter()
            result.preliminary_sql = self.generate_preliminary_sql(
                table_info, llm_selection, question, evidence, example
            )
            latency["preliminary_sql_gen_sec"] = time.perf_counter() - step_start

            if result.preliminary_sql:
                step_start = time.perf_counter()
                sql_tables, sql_columns = self.extract_schema_from_sql(
                    result.preliminary_sql, db_id
                )
                latency["extract_from_sql_sec"] = time.perf_counter() - step_start

        # Step 5: Extract schema from evidence (BIRD)
        step_start = time.perf_counter()
        hint_tables, hint_columns = self.extract_schema_from_evidence(evidence, db_id)
        latency["extract_from_evidence_sec"] = time.perf_counter() - step_start

        # Step 5b: Merge and filter all schema sources to get linked tables/columns
        step_start = time.perf_counter()
        result.tables, result.columns = self.merge_and_filter_schema(
            llm_selection,
            sql_tables,
            sql_columns,
            hint_tables,
            hint_columns,
            db_id
        )
        latency["merge_filter_sec"] = time.perf_counter() - step_start

        # =====================================================================
        # PHASE 2: Create SIMPLIFIED schema (matching offline Step 2)
        # =====================================================================

        # Step 6: Create SIMPLIFIED schema with only linked tables/columns
        step_start = time.perf_counter()
        table_columns_dict = self.build_table_columns_dict(result.tables, result.columns)
        result.simplified_ddl = self.create_simplified_ddl(result.tables, result.columns)
        result.ddl_data = self.get_simplified_sample_data(db_id, result.tables, table_columns_dict)
        result.foreign_key = self.get_simplified_foreign_keys(db_id, result.tables)
        latency["simplified_schema_sec"] = time.perf_counter() - step_start

        # Step 7: Get FILTERED column meanings (only for linked columns)
        step_start = time.perf_counter()
        if self.dataset == "bird":
            # BIRD: Get column meanings only for linked columns
            result.column_meaning = self.get_column_meanings_for_db(db_id)
            result.column_meaning_text = self.get_filtered_column_meanings(db_id, result.tables, result.columns)
        elif self.dataset == "spider":
            # Spider: Get column_meaning and column_info from spider_dev_samples.json
            spider_sample = self.get_spider_sample(db_id, question)
            if spider_sample:
                # Column meanings
                result.column_meaning = spider_sample.get('column_meaning', {})
                result.column_meaning_text = self.format_column_meaning_text(result.column_meaning)
                # Column info (type + sample values)
                result.column_info = spider_sample.get('column_info', {})
                result.column_info_text = self.format_column_info_text(result.column_info)
        latency["column_meaning_sec"] = time.perf_counter() - step_start

        # Step 7b: Get few-shot examples (if retriever is available)
        step_start = time.perf_counter()
        if self.few_shot_retriever is not None:
            result.example = self.get_few_shot_examples(question)
        latency["few_shot_examples_sec"] = time.perf_counter() - step_start

        # Step 8: Build enhanced schema text (using SIMPLIFIED components)
        step_start = time.perf_counter()
        result.enhanced_schema = self.build_enhanced_schema(result)
        latency["build_schema_sec"] = time.perf_counter() - step_start

        # Total time
        latency["schema_linking_total_sec"] = time.perf_counter() - total_start
        result.latency = latency

        return result

    def get_schema_text(self, result: OnlineSchemaResult) -> str:
        """Get the enhanced schema text for agents."""
        return result.enhanced_schema


# ============================================================================
# Convenience functions for integration with secondloop
# ============================================================================

def create_online_linker(
    dataset: str = "bird",
    use_ollama: bool = False,
    use_ollama_server: bool = False,
    ollama_model: str = "ticlazau/qwen2.5-coder-7b-instruct",
    ollama_host: str = "localhost",
    ollama_port: int = 11435,
    llm_config: Optional[LLMConfig] = None,
    k_shot: int = 3,
    split: str = "dev",
) -> OnlineSchemaLinker:
    """
    Create an OnlineSchemaLinker with appropriate LLM config.

    Args:
        dataset: "bird" or "spider"
        use_ollama: Use local Ollama (port 11434)
        use_ollama_server: Use server Ollama (port 11435)
        ollama_model: Model name for Ollama
        ollama_host: Host for server Ollama
        ollama_port: Port for server Ollama
        llm_config: Custom LLM config (overrides Ollama settings)
        k_shot: Number of few-shot examples to retrieve (default 3, 0 to disable)

    Returns:
        Configured OnlineSchemaLinker
    """
    if llm_config:
        config = llm_config
    elif use_ollama_server:
        config = get_ollama_server_config(
            model=ollama_model,
            host=ollama_host,
            port=ollama_port,
        )
    elif use_ollama:
        config = get_ollama_config(model=ollama_model)
    else:
        config = LLMConfig()

    return OnlineSchemaLinker(dataset=dataset, llm_config=config, k_shot=k_shot, split=split)


def process_single_question(
    question: str,
    db_id: str,
    dataset: str = "bird",
    evidence: str = "",
    use_ollama_server: bool = False,
    ollama_model: str = "ticlazau/qwen2.5-coder-7b-instruct",
    ollama_port: int = 11435,
    k_shot: int = 3,
) -> OnlineSchemaResult:
    """
    Convenience function to process a single question.

    Args:
        question: Natural language question
        db_id: Database ID
        dataset: "bird" or "spider"
        evidence: Domain knowledge (BIRD)
        use_ollama_server: Use server Ollama
        ollama_model: Model for Ollama
        ollama_port: Port for server Ollama
        k_shot: Number of few-shot examples (default 3, 0 to disable)

    Returns:
        OnlineSchemaResult with all schema information
    """
    linker = create_online_linker(
        dataset=dataset,
        use_ollama_server=use_ollama_server,
        ollama_model=ollama_model,
        ollama_port=ollama_port,
        k_shot=k_shot,
    )

    return linker.process_question(
        question=question,
        db_id=db_id,
        evidence=evidence,
    )


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Online Schema Linking")
    parser.add_argument("question", help="Natural language question")
    parser.add_argument("db_id", help="Database ID")
    parser.add_argument("--dataset", choices=["bird", "spider"], default="bird")
    parser.add_argument("--evidence", type=str, default="", help="Evidence/definition (BIRD)")
    parser.add_argument("--use_ollama_server", action="store_true", help="Use Ollama server (port 11435)")
    parser.add_argument("--ollama_model", type=str, default="ticlazau/qwen2.5-coder-7b-instruct")
    parser.add_argument("--ollama_port", type=int, default=11435)
    parser.add_argument("--k_shot", type=int, default=3, help="Number of few-shot examples (0 to disable)")

    args = parser.parse_args()

    print(f"Processing question: {args.question}")
    print(f"Database: {args.db_id}")
    print(f"Dataset: {args.dataset}")
    print(f"Few-shot examples: {args.k_shot}")
    print("-" * 50)

    result = process_single_question(
        question=args.question,
        db_id=args.db_id,
        dataset=args.dataset,
        evidence=args.evidence,
        use_ollama_server=args.use_ollama_server,
        ollama_model=args.ollama_model,
        ollama_port=args.ollama_port,
        k_shot=args.k_shot,
    )

    print("\n=== Schema Linking Results ===")
    print(f"Tables: {result.tables}")
    print(f"Columns: {result.columns}")
    print(f"\n=== Preliminary SQL ===")
    print(result.preliminary_sql)
    print(f"\n=== Few-Shot Examples ===")
    if result.example:
        print(result.example[:500] + "..." if len(result.example) > 500 else result.example)
    else:
        print("(none)")
    print(f"\n=== Enhanced Schema ===")
    print(result.enhanced_schema)
