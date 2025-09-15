"""
FastMCP Cloud entrypoint for Smartsheet MCP (Python port)

This implements parity with the Node/TypeScript Smartsheet MCP server using
FastMCP (Python). Configure Fast MCP Cloud Entrypoint as `server.py:mcp` and
provide the same environment variables:

- SMARTSHEET_API_KEY (required)
- SMARTSHEET_ENDPOINT (default https://api.smartsheet.com/2.0)
- ALLOW_DELETE_TOOLS (optional, 'true' enables delete tools)

Dependencies are declared in requirements.txt.
"""

from __future__ import annotations

import os
from typing import Optional, Any, Dict

from fastmcp import FastMCP
from pydantic import BaseModel, Field
import httpx
import asyncio
import re
from question_tools import register_question_tools


class SmartsheetClient:
    def __init__(self, token: str, base_url: str) -> None:
        self.token = token
        self.base_url = base_url.rstrip("/")

    async def request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> Any:
        attempt = 0
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "smar-mcp-python/1.0.0",
        }
        last_exc: Optional[Exception] = None
        while attempt <= max_retries:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                    resp = await client.request(method, url, headers=headers, params=params, json=json)
                if resp.status_code == 429 and attempt < max_retries:
                    retry_after = resp.headers.get("retry-after")
                    delay = (float(retry_after) if retry_after else 1.0) + (2 ** attempt) * 0.5
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                # Non-retriable or retries exhausted
                last_exc = e
                break
            except Exception as e:  # network/timeout etc.
                last_exc = e
                if attempt >= max_retries:
                    break
                await asyncio.sleep(0.5 * (2 ** attempt))
                attempt += 1
        if last_exc:
            raise last_exc
        raise RuntimeError("Unknown Smartsheet request failure")


# Environment configuration
SMARTSHEET_API_KEY = os.getenv("SMARTSHEET_API_KEY")
SMARTSHEET_ENDPOINT = (os.getenv("SMARTSHEET_ENDPOINT") or "https://api.smartsheet.com/2.0").rstrip("/")


def _client() -> SmartsheetClient:
    if not SMARTSHEET_API_KEY:
        raise RuntimeError("SMARTSHEET_API_KEY is not set")
    return SmartsheetClient(SMARTSHEET_API_KEY, SMARTSHEET_ENDPOINT)


# Create FastMCP server instance (discovered by FastMCP Cloud)
mcp = FastMCP(
    name="smartsheet",
    version="1.0.0",
    description="Smartsheet tools via FastMCP (Python)"
)

register_question_tools(mcp, _client)


# -------------------
# Tools
# -------------------


class GetSheetArgs(BaseModel):
    sheetId: str = Field(..., description="Smartsheet sheet ID")
    include: Optional[str] = Field(None, description="Comma-separated flags: format,formulas,etc.")
    pageSize: Optional[int] = None
    page: Optional[int] = None


@mcp.tool()
async def get_sheet(args: GetSheetArgs) -> dict:
    params: Dict[str, Any] = {}
    if args.include:
        params["include"] = args.include
    if args.pageSize is not None:
        params["pageSize"] = args.pageSize
    if args.page is not None:
        params["page"] = args.page
    return await _client().request("GET", f"/sheets/{args.sheetId}", params=params)


class GetSheetVersionArgs(BaseModel):
    sheetId: str = Field(..., description="Smartsheet sheet ID")


@mcp.tool()
async def get_sheet_version(args: GetSheetVersionArgs) -> dict:
    data = await _client().request("GET", f"/sheets/{args.sheetId}/version")
    return data


class GetSheetByUrlArgs(BaseModel):
    url: str
    include: Optional[str] = None
    pageSize: Optional[int] = None
    page: Optional[int] = None


@mcp.tool()
async def get_sheet_by_url(args: GetSheetByUrlArgs) -> dict:
    m = re.search(r"/sheets/([^?/]+)", args.url)
    if not m:
        raise ValueError("Invalid sheet URL")
    token = m.group(1)
    params: Dict[str, Any] = {}
    if args.include:
        params["include"] = args.include
    if args.pageSize is not None:
        params["pageSize"] = args.pageSize
    if args.page is not None:
        params["page"] = args.page
    return await _client().request("GET", f"/sheets/{token}", params=params)


class AddOrUpdateRowCell(BaseModel):
    columnId: int
    value: Optional[str] = None
    formula: Optional[str] = None
    

class AddRow(BaseModel):
    toBottom: Optional[bool] = None
    toTop: Optional[bool] = None
    parentId: Optional[int] = None
    siblingId: Optional[int] = None
    cells: list[AddOrUpdateRowCell]


class AddRowsArgs(BaseModel):
    sheetId: str
    rows: list[AddRow]


@mcp.tool()
async def add_rows(args: AddRowsArgs) -> dict:
    """Add rows to a sheet."""
    payload = [row.model_dump(exclude_none=True) for row in args.rows]
    return await _client().request("POST", f"/sheets/{args.sheetId}/rows", json=payload)


class UpdateRow(BaseModel):
    id: int
    cells: list[AddOrUpdateRowCell]


class UpdateRowsArgs(BaseModel):
    sheetId: str
    rows: list[UpdateRow]


@mcp.tool()
async def update_rows(args: UpdateRowsArgs) -> dict:
    """Update rows in a sheet (values, formatting, formulas)."""
    payload = [row.model_dump(exclude_none=True) for row in args.rows]
    return await _client().request("PUT", f"/sheets/{args.sheetId}/rows", json=payload)


class DeleteRowsArgs(BaseModel):
    sheetId: str
    rowIds: list[int]
    ignoreRowsNotFound: Optional[bool] = False


@mcp.tool()
async def delete_rows(args: DeleteRowsArgs) -> dict:
    """Delete rows from a sheet (requires ALLOW_DELETE_TOOLS=true)."""
    allow_delete = os.getenv("ALLOW_DELETE_TOOLS", "false").lower() == "true"
    if not allow_delete:
        raise RuntimeError("Delete tools are disabled. Set ALLOW_DELETE_TOOLS=true to enable.")

    ids = ",".join(map(str, args.rowIds))
    params = {"ids": ids}
    if args.ignoreRowsNotFound:
        params["ignoreRowsNotFound"] = "true"
    return await _client().request("DELETE", f"/sheets/{args.sheetId}/rows", params=params)


@mcp.tool()
async def server_info() -> dict:
    """Return server info for troubleshooting."""
    return {
        "name": "smartsheet",
        "transport": "fastmcp-python",
        "endpoint": SMARTSHEET_ENDPOINT,
        "delete_enabled": os.getenv("ALLOW_DELETE_TOOLS", "false").lower() == "true",
    }


# -------------------
# Additional Tools for parity
# -------------------


class GetCellHistoryArgs(BaseModel):
    sheetId: str
    rowId: str
    columnId: str
    include: Optional[str] = None
    pageSize: Optional[int] = None
    page: Optional[int] = None


@mcp.tool()
async def get_cell_history(args: GetCellHistoryArgs) -> dict:
    params: Dict[str, Any] = {}
    if args.include:
        params["include"] = args.include
    if args.pageSize is not None:
        params["pageSize"] = args.pageSize
    if args.page is not None:
        params["page"] = args.page
    return await _client().request(
        "GET",
        f"/sheets/{args.sheetId}/rows/{args.rowId}/columns/{args.columnId}/history",
        params=params,
    )


class GetRowArgs(BaseModel):
    sheetId: str
    rowId: str
    include: Optional[str] = None
    exclude: Optional[str] = None


@mcp.tool()
async def get_row(args: GetRowArgs) -> dict:
    params: Dict[str, Any] = {}
    if args.include:
        params["include"] = args.include
    if args.exclude:
        params["exclude"] = args.exclude
    return await _client().request("GET", f"/sheets/{args.sheetId}/rows/{args.rowId}", params=params)


class GetSheetLocationArgs(BaseModel):
    sheetId: str


@mcp.tool()
async def get_sheet_location(args: GetSheetLocationArgs) -> dict:
    sheet = await _client().request("GET", f"/sheets/{args.sheetId}")
    return {
        "folderId": sheet.get("parentId"),
        "folderType": sheet.get("parentType"),
        "workspaceId": sheet.get("workspaceId"),
    }


class CopySheetArgs(BaseModel):
    sheetId: str
    destinationName: str
    destinationFolderId: Optional[str] = None
    workspaceId: Optional[str] = None


@mcp.tool()
async def copy_sheet(args: CopySheetArgs) -> dict:
    data: Dict[str, Any] = {"newName": args.destinationName}
    if args.destinationFolderId:
        data.update({"destinationType": "folder", "destinationId": args.destinationFolderId})
    elif args.workspaceId:
        data.update({"destinationType": "workspace", "destinationId": args.workspaceId})
    else:
        data.update({"destinationType": "home"})
    return await _client().request("POST", f"/sheets/{args.sheetId}/copy", json=data)


class CreateSheetColumn(BaseModel):
    title: str
    type: str
    primary: Optional[bool] = None


class CreateSheetArgs(BaseModel):
    name: str
    columns: list[CreateSheetColumn]
    folderId: Optional[str] = None


@mcp.tool()
async def create_sheet(args: CreateSheetArgs) -> dict:
    payload = {"name": args.name, "columns": [c.model_dump(exclude_none=True) for c in args.columns]}
    endpoint = "/sheets" if not args.folderId else f"/folders/{args.folderId}/sheets"
    return await _client().request("POST", endpoint, json=payload)


class CreateUpdateRequestRecipient(BaseModel):
    email: str


class CreateUpdateRequestArgs(BaseModel):
    sheetId: str
    rowIds: Optional[list[int]] = None
    columnIds: Optional[list[int]] = None
    includeAttachments: Optional[bool] = None
    includeDiscussions: Optional[bool] = None
    message: Optional[str] = None
    subject: Optional[str] = None
    ccMe: Optional[bool] = None
    sendTo: list[CreateUpdateRequestRecipient]


@mcp.tool()
async def create_update_request(args: CreateUpdateRequestArgs) -> dict:
    payload = {
        "rowIds": args.rowIds,
        "columnIds": args.columnIds,
        "includeAttachments": args.includeAttachments,
        "includeDiscussions": args.includeDiscussions,
        "message": args.message,
        "subject": args.subject,
        "ccMe": args.ccMe,
        "sendTo": [r.model_dump() for r in args.sendTo],
    }
    # Strip None values
    payload = {k: v for k, v in payload.items() if v is not None}
    return await _client().request("POST", f"/sheets/{args.sheetId}/updaterequests", json=payload)


# Search tools
class SearchSheetsArgs(BaseModel):
    query: str


@mcp.tool()
async def search_sheets(args: SearchSheetsArgs) -> dict:
    return await _client().request("GET", "/search", params={"query": args.query, "scopes": "sheetNames,cellData,summaryFields"})


class SearchInSheetArgs(BaseModel):
    sheetId: str
    query: str


@mcp.tool()
async def search_in_sheet(args: SearchInSheetArgs) -> dict:
    return await _client().request("GET", f"/search/sheets/{args.sheetId}", params={"query": args.query})


class SearchInSheetByUrlArgs(BaseModel):
    url: str
    query: str


@mcp.tool()
async def search_in_sheet_by_url(args: SearchInSheetByUrlArgs) -> dict:
    m = re.search(r"/sheets/([^?/]+)", args.url)
    if not m:
        raise ValueError("Invalid sheet URL")
    token = m.group(1)
    # Resolve token to numeric ID by fetching sheet, then search
    sheet = await _client().request("GET", f"/sheets/{token}")
    return await _client().request("GET", f"/search/sheets/{sheet['id']}", params={"query": args.query})


class WhatAssignedBySheetIdArgs(BaseModel):
    sheetId: str


@mcp.tool()
async def what_am_i_assigned_to_by_sheet_id(args: WhatAssignedBySheetIdArgs) -> dict:
    user = await _client().request("GET", "/users/me")
    return await _client().request("GET", f"/search/sheets/{args.sheetId}", params={"query": user.get("email", "")})


class WhatAssignedBySheetUrlArgs(BaseModel):
    url: str


@mcp.tool()
async def what_am_i_assigned_to_by_sheet_url(args: WhatAssignedBySheetUrlArgs) -> dict:
    user = await _client().request("GET", "/users/me")
    m = re.search(r"/sheets/([^?/]+)", args.url)
    if not m:
        raise ValueError("Invalid sheet URL")
    token = m.group(1)
    sheet = await _client().request("GET", f"/sheets/{token}")
    return await _client().request("GET", f"/search/sheets/{sheet['id']}", params={"query": user.get("email", "")})


class SearchSimpleArgs(BaseModel):
    query: str


@mcp.tool()
async def search_folders(args: SearchSimpleArgs) -> dict:
    return await _client().request("GET", "/search", params={"query": args.query, "scopes": "folderNames"})


@mcp.tool()
async def search_workspaces(args: SearchSimpleArgs) -> dict:
    return await _client().request("GET", "/search", params={"query": args.query, "scopes": "workspaceNames"})


@mcp.tool()
async def search_reports(args: SearchSimpleArgs) -> dict:
    return await _client().request("GET", "/search", params={"query": args.query, "scopes": "reportNames"})


@mcp.tool()
async def search_dashboards(args: SearchSimpleArgs) -> dict:
    return await _client().request("GET", "/search", params={"query": args.query, "scopes": "sightNames"})


# User tools
class GetUserArgs(BaseModel):
    userId: str


@mcp.tool()
async def get_current_user() -> dict:
    return await _client().request("GET", "/users/me")


@mcp.tool()
async def get_user(args: GetUserArgs) -> dict:
    return await _client().request("GET", f"/users/{args.userId}")


@mcp.tool()
async def list_users() -> dict:
    return await _client().request("GET", "/users")


# Workspace tools
@mcp.tool()
async def get_workspaces() -> dict:
    return await _client().request("GET", "/workspaces")


class GetWorkspaceArgs(BaseModel):
    workspaceId: str


@mcp.tool()
async def get_workspace(args: GetWorkspaceArgs) -> dict:
    return await _client().request("GET", f"/workspaces/{args.workspaceId}")


class CreateWorkspaceArgs(BaseModel):
    workspaceName: str


@mcp.tool()
async def create_workspace(args: CreateWorkspaceArgs) -> dict:
    return await _client().request("POST", "/workspaces", json={"name": args.workspaceName})


# Folder tools
class GetFolderArgs(BaseModel):
    folderId: str


@mcp.tool()
async def get_folder(args: GetFolderArgs) -> dict:
    return await _client().request("GET", f"/folders/{args.folderId}")


class CreateFolderArgs(BaseModel):
    folderId: str
    folderName: str


@mcp.tool()
async def create_folder(args: CreateFolderArgs) -> dict:
    return await _client().request("POST", f"/folders/{args.folderId}/folders", json={"name": args.folderName})


class CreateWorkspaceFolderArgs(BaseModel):
    workspaceId: str
    folderName: str


@mcp.tool()
async def create_workspace_folder(args: CreateWorkspaceFolderArgs) -> dict:
    return await _client().request("POST", f"/workspaces/{args.workspaceId}/folders", json={"name": args.folderName})


# Discussion tools
class GetDiscussionsBySheetArgs(BaseModel):
    sheetId: str
    include: Optional[str] = None
    pageSize: Optional[int] = None
    page: Optional[int] = None
    includeAll: Optional[bool] = None


@mcp.tool()
async def get_discussions_by_sheet_id(args: GetDiscussionsBySheetArgs) -> dict:
    params: Dict[str, Any] = {}
    if args.include:
        params["include"] = args.include
    if args.pageSize is not None:
        params["pageSize"] = args.pageSize
    if args.page is not None:
        params["page"] = args.page
    if args.includeAll is not None:
        params["includeAll"] = str(args.includeAll).lower()
    return await _client().request("GET", f"/sheets/{args.sheetId}/discussions", params=params)


class GetDiscussionsByRowArgs(BaseModel):
    sheetId: str
    rowId: str
    include: Optional[str] = None
    pageSize: Optional[int] = None
    page: Optional[int] = None
    includeAll: Optional[bool] = None


@mcp.tool()
async def get_discussions_by_row_id(args: GetDiscussionsByRowArgs) -> dict:
    params: Dict[str, Any] = {}
    if args.include:
        params["include"] = args.include
    if args.pageSize is not None:
        params["pageSize"] = args.pageSize
    if args.page is not None:
        params["page"] = args.page
    if args.includeAll is not None:
        params["includeAll"] = str(args.includeAll).lower()
    return await _client().request("GET", f"/sheets/{args.sheetId}/rows/{args.rowId}/discussions", params=params)


class CreateSheetDiscussionArgs(BaseModel):
    sheetId: str
    commentText: str


@mcp.tool()
async def create_sheet_discussion(args: CreateSheetDiscussionArgs) -> dict:
    return await _client().request("POST", f"/sheets/{args.sheetId}/discussions", json={"comment": {"text": args.commentText}})


class CreateRowDiscussionArgs(BaseModel):
    sheetId: str
    rowId: str
    commentText: str


@mcp.tool()
async def create_row_discussion(args: CreateRowDiscussionArgs) -> dict:
    return await _client().request(
        "POST", f"/sheets/{args.sheetId}/rows/{args.rowId}/discussions", json={"comment": {"text": args.commentText}}
    )
