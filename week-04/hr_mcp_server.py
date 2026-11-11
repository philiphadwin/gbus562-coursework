"""MCP server for Hartwell Foods HR: policy search and leave balances.

Tools:
  search_policy(query)       The 4 HR-document sections closest to the query, with source files.
  get_leave_balance(email)   An employee's vacation, sick and personal days left (HR staff mode).
  get_leave_balance()        In --as-employee mode: only that one employee's balance, no inputs.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --as-employee dana.whitfield@hartwellfoods.example
    .venv\\Scripts\\python.exe week-04\\hr_mcp_server.py --rebuild     # re-index the HR docs, then exit

The server talks MCP over stdio, so an AI app (Claude Code, or an agent) starts it; running it by
hand just waits for input. The Chroma index lives in week-04/chroma_db (ignored by Git) and is
rebuilt automatically when a file in hartwell-hr/hr-docs changes.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

import hr_search

HERE = Path(__file__).resolve().parent
EMPLOYEES_CSV = HERE / "hartwell-hr" / "employees.csv"
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)

log = logging.getLogger("hr_mcp_server")


def find_employee(email):
    """Return the employees.csv row for this email (case-insensitive), or None.
    The file is read on every call so edits show up without restarting the server."""
    wanted = (email or "").strip().lower()
    if not wanted:
        return None
    with EMPLOYEES_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["email"].strip().lower() == wanted:
                return row
    return None


def balance_for(email):
    row = find_employee(email)
    if row is None:
        raise ToolError(
            f"No employee record matches '{email}'. Do not state, estimate, or guess any leave "
            "balance for this person. Say that the balance could not be found and that a person "
            "from HR will check it."
        )
    return {
        "email": row["email"],
        "name": row["name"],
        "department": row["department"],
        "manager": row["manager"] or None,
        "vacation_days_left": int(row["vacation_days_left"]),
        "sick_days_left": int(row["sick_days_left"]),
        "personal_days_left": int(row["personal_days_left"]),
    }


def build_server(collection, as_employee=None):
    mcp = MCPServer(
        "hartwell-hr",
        instructions=(
            "Hartwell Foods HR data. Use search_policy for questions about HR policy and cite the "
            "source file of each fact you use. Use get_leave_balance for leave balances. Only "
            "state policies and numbers that these tools return."
        ),
    )

    @mcp.tool(annotations=READ_ONLY)
    def search_policy(query: str) -> dict:
        """Search Hartwell Foods' HR policy documents (vacation and time off, sick days, jury duty,
        bereavement, parental and family leave, remote work, benefits enrollment, and reporting
        workplace concerns).

        Returns the 4 document sections closest to the query, best match first. Each has the
        source file name, the document title and section, a similarity score from 0 to 1, and the
        section text. Cite the source file for every policy fact you use, e.g.
        [leave-policy.html]. Low similarity scores mean the documents may not cover the question;
        if no section actually answers it, say so rather than guessing.

        Args:
            query: What to look up, in plain words, e.g. "carry over unused vacation days".
        """
        query = query.strip()
        if not query:
            raise ToolError("The search query is empty. Give a few words describing the policy topic.")
        return {"query": query, "results": hr_search.search(collection, query)}

    if as_employee:
        @mcp.tool(name="get_leave_balance", annotations=READ_ONLY)
        def get_own_leave_balance() -> dict:
            """Get the leave balance of the employee this session is for: vacation, sick and
            personal days left, plus their name, department and manager.

            Takes no inputs. It can only return this one employee's balance; there is no way to
            look up anyone else's.
            """
            return balance_for(as_employee)
    else:
        @mcp.tool(annotations=READ_ONLY)
        def get_leave_balance(email: str) -> dict:
            """Look up an employee's leave balance by their work email (for HR staff): vacation,
            sick and personal days left, plus their name, department and manager.

            Args:
                email: The employee's work email address, e.g. dana.whitfield@hartwellfoods.example.
            """
            return balance_for(email)

    return mcp


def main():
    parser = argparse.ArgumentParser(description="MCP server for Hartwell Foods HR data.")
    parser.add_argument("--as-employee", metavar="EMAIL",
                        help="serve only this employee's leave balance; get_leave_balance takes no inputs")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the HR document index, then exit")
    args = parser.parse_args()

    # stdout carries the MCP protocol, so all logging goes to stderr.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(name)s: %(message)s")

    if args.rebuild:
        collection = hr_search.get_collection(rebuild=True)
        log.info("Rebuilt the index: %d chunks.", collection.count())
        return

    if args.as_employee:
        args.as_employee = args.as_employee.strip()
        if find_employee(args.as_employee) is None:
            log.warning("--as-employee %s matches no record in employees.csv; "
                        "get_leave_balance will return an error.", args.as_employee)
        log.info("Employee mode: get_leave_balance serves only %s", args.as_employee)

    # Build or open the index, and load the embedding model, before accepting requests,
    # so the first search isn't slow enough to time out.
    collection = hr_search.get_collection()
    hr_search.search(collection, "warm up")

    build_server(collection, args.as_employee).run("stdio")


if __name__ == "__main__":
    main()
