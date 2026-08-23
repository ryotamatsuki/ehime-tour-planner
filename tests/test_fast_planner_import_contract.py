from __future__ import annotations

import ast
from pathlib import Path


def test_fast_planner_does_not_import_new_planner_helpers():
    """Keep hot-reload compatibility with older in-memory planner modules.

    Streamlit can reload workflow.fast_planner while workflow.planner remains
    cached in sys.modules from the previous deploy. The fast planner may depend
    on the long-lived base classes, but must not import newly-added constants or
    private helpers from workflow.planner at module import time.
    """
    path = Path("workflow/fast_planner.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "workflow.planner":
            imported.update(alias.name for alias in node.names)

    assert imported == {"PlannerState", "PlannerWorkflow"}
