
from __future__ import annotations

import difflib
import math
import re
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from fastmcp import FastMCP
from pydantic import BaseModel, Field, model_validator


ClientFactory = Callable[[], Any]

MAX_INDEXED_VALUES_PER_COLUMN = 50
MAX_GROUP_SUMMARY_RESULTS = 3
MAX_GROUP_ANALYSIS_RESULTS = 10


class AskSheetQuestionArgs(BaseModel):
    """Arguments for the ask_sheet_question tool."""

    question: str = Field(..., description="Natural language question about the sheet data.")
    sheetId: Optional[str] = Field(None, description="Smartsheet sheet ID to analyze.")
    sheetUrl: Optional[str] = Field(None, description="Smartsheet sheet URL (if the ID is unknown).")
    rowLimit: int = Field(200, ge=1, le=5000, description="Maximum number of rows to retrieve for analysis.")
    filterText: Optional[str] = Field(
        None,
        description="Optional case-insensitive text filter applied to rows before analysis.",
    )

    @model_validator(mode="after")
    def _ensure_reference(cls, model: "AskSheetQuestionArgs") -> "AskSheetQuestionArgs":
        if not model.sheetId and not model.sheetUrl:
            raise ValueError("Either sheetId or sheetUrl must be provided.")
        return model


def register_question_tools(mcp: FastMCP, client_factory: ClientFactory) -> None:
    """Register question-oriented tools on the provided MCP server."""

    @mcp.tool()
    async def ask_sheet_question(args: AskSheetQuestionArgs) -> Dict[str, Any]:
        """Answer lightweight analytical questions about a Smartsheet sheet."""

        sheet = await _fetch_sheet(client_factory, args.sheetId, args.sheetUrl, args.rowLimit)
        sheet_title = sheet.get("name")
        sheet_label = sheet_title or "the sheet"
        sheet_display = f"'{sheet_title}'" if sheet_title else "the sheet"
        column_lookup = {
            column.get("id"): column.get("title") or str(column.get("id"))
            for column in sheet.get("columns", [])
        }
        column_titles = [title for title in column_lookup.values() if title]

        rows_context: List[Dict[str, Any]] = []
        numeric_columns: Dict[str, List[float]] = defaultdict(list)
        text_index: Dict[str, Dict[str, str]] = {}

        filter_text = args.filterText.lower() if args.filterText else None

        for row in sheet.get("rows", []):
            values: Dict[str, Any] = {}
            numeric_map: Dict[str, float] = {}
            for cell in row.get("cells", []):
                column_id = cell.get("columnId")
                column_title = column_lookup.get(column_id)
                if not column_title:
                    continue
                raw_value = _extract_cell_value(cell)
                values[column_title] = raw_value

                numeric_value = _coerce_to_number(raw_value)
                if numeric_value is not None:
                    numeric_map[column_title] = numeric_value
                    numeric_columns[column_title].append(numeric_value)
                elif isinstance(raw_value, str):
                    lower = raw_value.strip().lower()
                    if lower and len(text_index.setdefault(column_title, {})) < MAX_INDEXED_VALUES_PER_COLUMN:
                        text_index[column_title][lower] = raw_value

            if filter_text:
                haystack = " ".join(str(v).lower() for v in values.values() if v is not None)
                if filter_text not in haystack:
                    continue

            rows_context.append({"values": values, "numeric": numeric_map})
            if len(rows_context) >= args.rowLimit:
                break

        numeric_columns = {column: values for column, values in numeric_columns.items() if values}

        question_lower = args.question.lower().strip()
        detected_filters = _detect_filters(question_lower, text_index, column_titles)
        filtered_rows = _apply_filters(rows_context, detected_filters)
        group_column = _detect_group_column(question_lower, column_titles)

        analysis_notes: List[str] = []
        if detected_filters and not filtered_rows:
            analysis_notes.append("Detected filter values but no rows matched; using all retrieved rows instead.")
            working_rows = rows_context
            applied_filters: List[Tuple[str, str]] = []
        else:
            working_rows = filtered_rows if detected_filters else rows_context
            applied_filters = detected_filters

        answer: Optional[str] = None
        group_summary: Optional[Dict[str, Any]] = None

        if not working_rows:
            answer = f"No rows were available to analyze for {sheet_display}."
        else:
            if answer is None and _contains(question_lower, ("sum", "total", "add up")):
                column = _guess_column(question_lower, numeric_columns.keys())
                if column:
                    if group_column:
                        summary = _summarize_group_numeric(working_rows, group_column, column, "sum")
                        if summary:
                            answer = _format_group_answer(summary, sheet_display)
                            group_summary = summary
                        else:
                            analysis_notes.append(
                                f"Column '{column}' did not have numeric values for grouping by {group_column}."
                            )
                    if answer is None:
                        values = _collect_numeric(working_rows, column)
                        if values:
                            total = sum(values)
                            answer = (
                                f"The total of {column} in {sheet_display} is {_format_number_text(total)} across "
                                f"{len(values)} rows{_format_filter_suffix(applied_filters)}."
                            )
                        else:
                            analysis_notes.append(f"Column '{column}' did not have numeric values after filtering.")
                else:
                    analysis_notes.append("Could not determine which column to sum.")

            if answer is None and _contains(question_lower, ("average", "avg", "mean")):
                column = _guess_column(question_lower, numeric_columns.keys())
                if column:
                    if group_column:
                        summary = _summarize_group_numeric(working_rows, group_column, column, "average")
                        if summary:
                            answer = _format_group_answer(summary, sheet_display)
                            group_summary = summary
                        else:
                            analysis_notes.append(
                                f"Column '{column}' did not have numeric values for grouping by {group_column}."
                            )
                    if answer is None:
                        values = _collect_numeric(working_rows, column)
                        if values:
                            avg = sum(values) / len(values)
                            answer = (
                                f"The average {column} in {sheet_display} is {_format_number_text(avg)} using "
                                f"{len(values)} values{_format_filter_suffix(applied_filters)}."
                            )
                        else:
                            analysis_notes.append(f"Column '{column}' did not have numeric values after filtering.")
                else:
                    analysis_notes.append("Could not determine which column to average.")

            if answer is None and _contains(question_lower, ("max", "highest", "largest", "top", "biggest")):
                column = _guess_column(question_lower, numeric_columns.keys())
                if column:
                    values = _collect_numeric(working_rows, column)
                    if values:
                        max_value = max(values)
                        row_preview = _find_row_with_value(working_rows, column, max_value)
                        preview_text = _format_row_preview(row_preview)
                        answer = (
                            f"The highest {column} in {sheet_display} is {_format_number_text(max_value)}"
                            f"{_format_filter_suffix(applied_filters)}."
                        )
                        if preview_text:
                            answer += f" First matching row: {preview_text}."
                    else:
                        analysis_notes.append(f"Column '{column}' did not have numeric values after filtering.")
                else:
                    analysis_notes.append("Could not determine which column to inspect for the maximum value.")

            if answer is None and _contains(question_lower, ("min", "lowest", "smallest", "least")):
                column = _guess_column(question_lower, numeric_columns.keys())
                if column:
                    values = _collect_numeric(working_rows, column)
                    if values:
                        min_value = min(values)
                        row_preview = _find_row_with_value(working_rows, column, min_value)
                        preview_text = _format_row_preview(row_preview)
                        answer = (
                            f"The lowest {column} in {sheet_display} is {_format_number_text(min_value)}"
                            f"{_format_filter_suffix(applied_filters)}."
                        )
                        if preview_text:
                            answer += f" First matching row: {preview_text}."
                    else:
                        analysis_notes.append(f"Column '{column}' did not have numeric values after filtering.")
                else:
                    analysis_notes.append("Could not determine which column to inspect for the minimum value.")

            if answer is None and _contains(question_lower, ("count", "how many", "number of")):
                if group_column:
                    summary = _summarize_group_counts(working_rows, group_column)
                    if summary:
                        answer = _format_group_answer(summary, sheet_display)
                        group_summary = summary
                    else:
                        analysis_notes.append(f"No values found for grouping by {group_column}.")
                if answer is None:
                    if applied_filters:
                        answer = (
                            f"{len(working_rows)} rows match "
                            f"{_describe_filters(applied_filters)}{_format_filter_suffix(applied_filters)}."
                        )
                    else:
                        column = _guess_column(question_lower, column_titles)
                        if column:
                            count = sum(1 for row in working_rows if _has_value(row["values"].get(column)))
                            answer = (
                                f"{count} rows in {sheet_display} have a value for {column}."
                            )
                        else:
                            answer = (
                                f"{sheet_label} has {len(working_rows)} rows in the analyzed window."
                            )

        if answer is None:
            answer = (
                f"{sheet_label} has {len(working_rows)} rows (of {len(rows_context)} retrieved) "
                f"across {len(column_titles)} columns."
            )

        analysis: Dict[str, Any] = {
            "sheet": {"id": sheet.get("id"), "name": sheet_title},
            "columns": column_titles,
            "rows_retrieved": len(rows_context),
            "rows_used": len(working_rows),
            "filters_detected": _serialize_filters(detected_filters),
            "filters_applied": _serialize_filters(applied_filters),
        }
        if args.filterText:
            analysis["filter_text"] = args.filterText
        if group_summary:
            analysis["group_summary"] = group_summary

        numeric_overview = {
            column: {
                "count": len(values),
                "sum": _format_number(sum(values)),
                "average": _format_number(sum(values) / len(values) if values else 0.0),
                "min": _format_number(min(values)),
                "max": _format_number(max(values)),
            }
            for column, values in list(numeric_columns.items())[:8]
        }
        if numeric_overview:
            analysis["numeric_overview"] = numeric_overview
        if analysis_notes:
            analysis["notes"] = analysis_notes

        sample_rows = [row["values"] for row in working_rows[: min(5, len(working_rows))]]

        return {
            "answer": answer,
            "sheet": {"id": sheet.get("id"), "name": sheet_title},
            "analysis": analysis,
            "sampleRows": sample_rows,
        }


async def _fetch_sheet(
    client_factory: ClientFactory, sheet_id: Optional[str], sheet_url: Optional[str], row_limit: int
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"pageSize": row_limit}
    client = client_factory()
    if sheet_id:
        return await client.request("GET", f"/sheets/{sheet_id}", params=params)
    if sheet_url:
        match = re.search(r"/sheets/([^?/]+)", sheet_url)
        if not match:
            raise ValueError("Could not parse sheetId from sheetUrl.")
        token = match.group(1)
        return await client.request("GET", f"/sheets/{token}", params=params)
    raise ValueError("Missing sheet reference.")


def _extract_cell_value(cell: Dict[str, Any]) -> Any:
    if "displayValue" in cell and cell["displayValue"] is not None:
        return cell["displayValue"]
    return cell.get("value")


def _coerce_to_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("%"):
            try:
                return float(text[:-1].replace(",", "").strip()) / 100.0
            except ValueError:
                return None
        text = text.replace(",", "")
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _detect_filters(
    question_lower: str,
    text_index: Dict[str, Dict[str, str]],
    column_titles: Sequence[str],
) -> List[Tuple[str, str]]:
    filters: List[Tuple[str, str]] = []
    used_columns: set[str] = set()

    for column, value in _detect_explicit_filters(question_lower, column_titles, text_index):
        filters.append((column, value))
        used_columns.add(column)
        if len(filters) >= 3:
            return filters

    for column, values in text_index.items():
        if column in used_columns:
            continue
        for value_lower, original in values.items():
            if len(value_lower) < 2:
                continue
            pattern = re.compile(r"\b" + re.escape(value_lower) + r"\b")
            if pattern.search(question_lower):
                filters.append((column, original))
                break
        if len(filters) >= 3:
            break
    return filters


def _detect_explicit_filters(
    question_lower: str,
    column_titles: Sequence[str],
    text_index: Dict[str, Dict[str, str]],
) -> List[Tuple[str, str]]:
    explicit_filters: List[Tuple[str, str]] = []
    for column in column_titles:
        column_lower = column.lower()
        escaped = re.escape(column_lower)
        patterns = [
            rf"""{escaped}\s*(?:=|equals|is)\s*['"]?([^,;:'"\n]+)""",
            rf"""{escaped}\s*:\s*['"]?([^,;:'"\n]+)""",
            rf"""(?:where|with)\s+{escaped}\s*(?:=|equals|is)?\s*['"]?([^,;:'"\n]+)""",
        ]
        for pattern in patterns:
            match = re.search(pattern, question_lower)
            if not match:
                continue
            raw_value = _clean_extracted_value(match.group(1))
            if not raw_value:
                continue
            explicit_filters.append((column, _match_in_index(column, raw_value, text_index)))
            break
    return explicit_filters


def _clean_extracted_value(value: str) -> str:
    cleaned = value.strip(" \t'\"")
    cleaned = re.sub(r"(?:and|or|but|by).*", "", cleaned).strip()
    return cleaned


def _match_in_index(column: str, candidate: str, text_index: Dict[str, Dict[str, str]]) -> str:
    values = text_index.get(column)
    candidate_lower = candidate.lower().strip()
    if not values:
        return candidate.strip()
    if candidate_lower in values:
        return values[candidate_lower]
    best_match: Optional[str] = None
    best_ratio = 0.0
    for value_lower, original in values.items():
        ratio = difflib.SequenceMatcher(None, value_lower, candidate_lower).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = original
    if best_match and best_ratio >= 0.75:
        return best_match
    return candidate.strip()


def _detect_group_column(question_lower: str, column_titles: Sequence[str]) -> Optional[str]:
    for title in column_titles:
        lowered = title.lower()
        patterns = [
            f" by {lowered}",
            f" per {lowered}",
            f" by the {lowered}",
            f" per the {lowered}",
            f" grouped by {lowered}",
            f" for each {lowered}",
        ]
        if any(pattern in question_lower for pattern in patterns):
            return title
    match = re.search(r"(?:by|per|each|every)\s+([a-z0-9 _-]+)", question_lower)
    if match:
        candidate = match.group(1).strip()
        return _guess_column(candidate, column_titles)
    return None


def _apply_filters(rows: Sequence[Dict[str, Dict[str, Any]]], filters: Sequence[Tuple[str, str]]):
    if not filters:
        return list(rows)
    filtered: List[Dict[str, Dict[str, Any]]] = []
    for row in rows:
        values = row.get("values", {})
        include = True
        for column, expected in filters:
            value = values.get(column)
            if value is None:
                include = False
                break
            if str(value).strip().lower() != expected.strip().lower():
                include = False
                break
        if include:
            filtered.append(row)
    return filtered


def _collect_numeric(rows: Sequence[Dict[str, Dict[str, Any]]], column: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        numeric_map = row.get("numeric", {})
        if column in numeric_map:
            values.append(numeric_map[column])
    return values


def _find_row_with_value(
    rows: Sequence[Dict[str, Dict[str, Any]]], column: str, target: float
) -> Optional[Dict[str, Any]]:
    for row in rows:
        numeric_map = row.get("numeric", {})
        if column not in numeric_map:
            continue
        if math.isclose(numeric_map[column], target, rel_tol=1e-9, abs_tol=1e-9):
            return row.get("values")
    return None


def _format_row_preview(row: Optional[Dict[str, Any]]) -> str:
    if not row:
        return ""
    items = []
    for column, value in row.items():
        if value is None:
            continue
        items.append(f"{column}: {value}")
        if len(items) >= 3:
            break
    return ", ".join(items)


def _guess_column(question_lower: str, titles: Sequence[str]) -> Optional[str]:
    if not titles:
        return None
    normalized = {title.lower(): title for title in titles if title}
    for lower_title, original in normalized.items():
        if lower_title and lower_title in question_lower:
            return original
    tokens = re.findall(r"[a-z0-9]+", question_lower)
    best_match: Optional[str] = None
    best_score = 0.0
    for lower_title, original in normalized.items():
        ratio = difflib.SequenceMatcher(None, lower_title, question_lower).ratio()
        token_hits = 0
        for token in tokens:
            if token and token in lower_title:
                token_hits += 1
        token_score = token_hits / max(len(lower_title.split()), 1)
        score = max(ratio, token_score)
        if score > best_score:
            best_score = score
            best_match = original
    if best_score >= 0.45:
        return best_match
    return None


def _contains(question_lower: str, keywords: Sequence[str]) -> bool:
    return any(keyword in question_lower for keyword in keywords)


def _summarize_group_numeric(
    rows: Sequence[Dict[str, Dict[str, Any]]],
    group_column: str,
    metric_column: str,
    aggregation: str,
) -> Optional[Dict[str, Any]]:
    groups: Dict[str, Dict[str, float]] = {}
    for row in rows:
        values = row.get("values", {})
        numeric_map = row.get("numeric", {})
        group_value = values.get(group_column)
        metric_value = numeric_map.get(metric_column)
        if metric_value is None:
            continue
        normalized_group = _normalize_group_value(group_value)
        bucket = groups.setdefault(normalized_group, {"total": 0.0, "count": 0})
        bucket["total"] += metric_value
        bucket["count"] += 1
    if not groups:
        return None

    results: List[Dict[str, Any]] = []
    for group_value, bucket in groups.items():
        count = bucket["count"]
        total = bucket["total"]
        if aggregation == "sum":
            value = total
        elif aggregation == "average":
            value = total / count if count else 0.0
        else:
            raise ValueError(f"Unsupported aggregation '{aggregation}'")
        results.append(
            {
                "group": group_value,
                "value": _format_number(value),
                "count": count,
            }
        )

    results.sort(key=lambda item: item["value"], reverse=True)
    total_rows = sum(item["count"] for item in results)
    truncated = results[:MAX_GROUP_ANALYSIS_RESULTS]
    return {
        "aggregation": aggregation,
        "groupColumn": group_column,
        "metricColumn": metric_column,
        "rowsConsidered": total_rows,
        "totalGroups": len(results),
        "results": truncated,
    }


def _summarize_group_counts(
    rows: Sequence[Dict[str, Dict[str, Any]]], group_column: str
) -> Optional[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in rows:
        group_value = _normalize_group_value(row.get("values", {}).get(group_column))
        counter[group_value] += 1
    if not counter:
        return None
    total_rows = sum(counter.values())
    results = [
        {
            "group": group,
            "value": count,
            "count": count,
        }
        for group, count in counter.most_common(MAX_GROUP_ANALYSIS_RESULTS)
    ]
    return {
        "aggregation": "count",
        "groupColumn": group_column,
        "metricColumn": None,
        "rowsConsidered": total_rows,
        "totalGroups": len(counter),
        "results": results,
    }


def _normalize_group_value(value: Any) -> str:
    if value is None:
        return "(blank)"
    text = str(value).strip()
    return text if text else "(blank)"


def _format_group_answer(summary: Dict[str, Any], sheet_display: str) -> str:
    aggregation = summary["aggregation"]
    group_column = summary["groupColumn"]
    metric_column = summary.get("metricColumn")
    results = summary.get("results", [])
    if not results:
        return f"No grouped results were available for {sheet_display}."

    prefix: str
    if aggregation == "sum":
        prefix = f"Totals for {metric_column} grouped by {group_column} in {sheet_display}: "
    elif aggregation == "average":
        prefix = f"Average {metric_column} by {group_column} in {sheet_display}: "
    else:
        prefix = f"Counts by {group_column} in {sheet_display}: "

    top = results[:MAX_GROUP_SUMMARY_RESULTS]
    formatted = ", ".join(
        f"{item['group']} ({_format_metric_value(item['value'])})" for item in top
    )
    if len(results) > MAX_GROUP_SUMMARY_RESULTS:
        formatted += ", …"
    return prefix + formatted


def _format_metric_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return _format_number_text(float(value))
    return str(value)


def _format_filter_suffix(filters: Sequence[Tuple[str, str]]) -> str:
    if not filters:
        return ""
    parts = [f"{column} = '{value}'" for column, value in filters]
    return f" (filtered by {', '.join(parts)})"


def _describe_filters(filters: Sequence[Tuple[str, str]]) -> str:
    parts = [f"{column} = '{value}'" for column, value in filters]
    return " and ".join(parts)


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _format_number(value: float) -> Any:
    if math.isfinite(value):
        if abs(value - round(value)) < 1e-9:
            return int(round(value))
        return round(value, 4)
    return value


def _format_number_text(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.2f}"


def _serialize_filters(filters: Sequence[Tuple[str, str]]) -> List[Dict[str, str]]:
    return [{"column": column, "value": value} for column, value in filters]


__all__ = ["register_question_tools", "AskSheetQuestionArgs"]
