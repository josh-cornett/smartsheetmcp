#!/usr/bin/env node

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import express from "express";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import { z } from "zod";
import { SmartsheetAPI } from "./apis/smartsheet-api.js";
import { config } from "dotenv";
import { getDiscussionTools } from "./tools/smartsheet-discussion-tools.js";
import { getFolderTools } from "./tools/smartsheet-folder-tools.js";
import { getSearchTools } from "./tools/smartsheet-search-tools.js";
import { getSheetTools } from "./tools/smartsheet-sheet-tools.js";
import { getUpdateRequestTools } from "./tools/smartsheet-update-request-tools.js";
import { getUserTools } from "./tools/smartsheet-user-tools.js";
import { getWorkspaceTools } from "./tools/smartsheet-workspace-tools.js";

// Load environment variables
config();

// Control whether deletion operations are enabled
const allowDeleteTools = process.env.ALLOW_DELETE_TOOLS === 'true';
console.info(`Delete operations are ${allowDeleteTools ? 'enabled' : 'disabled'}`);
  
// Initialize the MCP server
const server = new McpServer({
  name: "smartsheet",
  version: "1.0.0",
});

// Initialize the direct API client
const api = new SmartsheetAPI(process.env.SMARTSHEET_API_KEY, process.env.SMARTSHEET_ENDPOINT);

// Tool: Discussion tools
getDiscussionTools(server, api);

// Tool: Folder tools
getFolderTools(server, api);

// Tool: Search tools
getSearchTools(server, api);

// Tool: Sheet tools
getSheetTools(server, api, allowDeleteTools);

// Tool: Update Request tools
getUpdateRequestTools(server, api);

// Tool: User tools
getUserTools(server, api);

// Tool: Workspace tools
getWorkspaceTools(server, api); 

// Start the server with either stdio or SSE transport
async function main() {
  const transportMode = (process.env.MCP_TRANSPORT || "stdio").toLowerCase();

  if (transportMode === "sse") {
    const port = Number(process.env.PORT || 3000);
    const ssePath = process.env.MCP_SSE_PATH || "/sse";
    const messagesPath = process.env.MCP_MESSAGES_PATH || "/messages";
    const authToken = process.env.MCP_AUTH_TOKEN;

    const app = express();
    app.use(express.json({ limit: "1mb" }));

    // Simple bearer auth middleware if token is configured
    const requireAuth = (req: express.Request, res: express.Response, next: express.NextFunction) => {
      if (!authToken) return next();
      const auth = req.headers["authorization"] || "";
      const valid = typeof auth === "string" && auth.startsWith("Bearer ") && auth.slice(7) === authToken;
      if (!valid) return res.status(401).send("Unauthorized");
      next();
    };

    // Keep transports per-session for legacy SSE
    const transports: Record<string, SSEServerTransport> = {};

    // SSE endpoint: establishes the event stream
    app.get(ssePath, requireAuth, async (req, res) => {
      const transport = new SSEServerTransport(messagesPath, res);
      transports[transport.sessionId] = transport;

      res.on("close", () => {
        delete transports[transport.sessionId];
      });

      await server.connect(transport);
    });

    // Messages endpoint: clients POST messages here
    app.post(messagesPath, requireAuth, async (req, res) => {
      const sessionId = (req.query.sessionId as string) || "";
      const transport = transports[sessionId];
      if (!transport) return res.status(400).send("No transport found for sessionId");
      await transport.handlePostMessage(req, res, req.body);
    });

    app.listen(port, () => {
      console.info(`Smartsheet MCP Server running over SSE on port ${port} (sse: ${ssePath}, messages: ${messagesPath})`);
    });
    return;
  }

  // Default: stdio transport
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.info("Smartsheet MCP Server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error in main()", { error });
  process.exit(1);
});
