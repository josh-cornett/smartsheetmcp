
import asyncio
import sys
import types

import pytest

# Provide a lightweight fastmcp stub so question_tools can import FastMCP.
_stub_module = types.ModuleType("fastmcp")


class _FastMCPStub:
    def __init__(self, *args, **kwargs):
        pass

    def tool(self, *args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


_stub_module.FastMCP = _FastMCPStub
sys.modules.setdefault("fastmcp", _stub_module)

from question_tools import AskSheetQuestionArgs, register_question_tools


class FakeMCP:
    def __init__(self) -> None:
        self.tools = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return decorator


class FakeClient:
    def __init__(self, sheet):
        self.sheet = sheet

    async def request(self, method: str, endpoint: str, params=None, json=None):
        return self.sheet


def build_sheet(name: str, columns: list[str], rows: list[list]):
    column_defs = [{"id": idx + 1, "title": title} for idx, title in enumerate(columns)]
    smartsheet_rows = []
    for idx, row_values in enumerate(rows):
        cells = []
        for col_idx, value in enumerate(row_values):
            cells.append({"columnId": col_idx + 1, "value": value})
        smartsheet_rows.append({"id": idx + 1, "cells": cells})
    return {
        "id": 123,
        "name": name,
        "columns": column_defs,
        "rows": smartsheet_rows,
    }


@pytest.fixture
def ask_tool():
    def _factory(sheet):
        fake_client = FakeClient(sheet)
        fake_mcp = FakeMCP()
        register_question_tools(fake_mcp, lambda: fake_client)
        return fake_mcp.tools["ask_sheet_question"]

    return _factory


def invoke(tool, question: str):
    return asyncio.run(tool(AskSheetQuestionArgs(question=question, sheetId="123")))


def test_grouped_sum_with_explicit_filter(ask_tool):
    sheet = build_sheet(
        "Pipeline",
        ["Region", "Amount", "Status"],
        [
            ["North", 100, "Closed"],
            ["South", 80, "Closed"],
            ["North", 150, "Closed"],
            ["South", 20, "Open"],
        ],
    )
    tool = ask_tool(sheet)
    result = invoke(tool, "What is the total amount by region where status = closed?")

    group_summary = result["analysis"]["group_summary"]
    assert group_summary["aggregation"] == "sum"
    assert group_summary["groupColumn"] == "Region"
    assert group_summary["metricColumn"] == "Amount"

    grouped_values = {item["group"]: item["value"] for item in group_summary["results"]}
    assert grouped_values["North"] == 250
    assert grouped_values["South"] == 80

    assert {"column": "Status", "value": "Closed"} in result["analysis"]["filters_applied"]


def test_grouped_count_response(ask_tool):
    sheet = build_sheet(
        "Tasks",
        ["Task", "Owner"],
        [
            ["Task 1", "Alice"],
            ["Task 2", "Bob"],
            ["Task 3", "Alice"],
            ["Task 4", "Cara"],
        ],
    )
    tool = ask_tool(sheet)
    result = invoke(tool, "How many tasks by owner?")

    group_summary = result["analysis"]["group_summary"]
    assert group_summary["aggregation"] == "count"
    counts = {item["group"]: item["value"] for item in group_summary["results"]}
    assert counts == {"Alice": 2, "Bob": 1, "Cara": 1}


def test_grouped_average_with_colon_filter(ask_tool):
    sheet = build_sheet(
        "Performance",
        ["Owner", "Score", "Status"],
        [
            ["Alice", 80, "Open"],
            ["Alice", 100, "Closed"],
            ["Bob", 90, "Open"],
            ["Bob", 70, "Open"],
        ],
    )
    tool = ask_tool(sheet)
    result = invoke(tool, "What is the average score per owner with status: open?")

    group_summary = result["analysis"].get("group_summary")
    assert group_summary is not None
    assert group_summary["aggregation"] == "average"
    assert group_summary["groupColumn"] == "Owner"
    assert group_summary["metricColumn"] == "Score"

    averages = {item["group"]: item["value"] for item in group_summary["results"]}
    assert averages["Alice"] == 80
    assert averages["Bob"] == 80

    assert {"column": "Status", "value": "Open"} in result["analysis"]["filters_applied"]
