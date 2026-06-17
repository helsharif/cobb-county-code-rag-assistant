"""Export the Option 5 LightRAG knowledge graph to an interactive HTML view.

The full LightRAG graph can be large, so this exporter writes a filtered subset
by default. Use --match to inspect a topic neighborhood or --limit to change
the top-degree sample size.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import networkx as nx


DEFAULT_GRAPH = Path(
    "vectorstore/rag_anything_lightrag_option5/lightrag/graph_chunk_entity_relation.graphml"
)
DEFAULT_OUTPUT = Path("vectorstore/rag_anything_lightrag_option5/option5_kg_view.html")

TYPE_COLORS = {
    "document": "#4e79a7",
    "section": "#59a14f",
    "table": "#f28e2b",
    "concept": "#e15759",
    "entity": "#b07aa1",
    "default": "#767676",
}


def _as_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _match_nodes(graph: nx.Graph, needle: str) -> set[str]:
    needle = needle.casefold()
    matches: set[str] = set()
    for node, data in graph.nodes(data=True):
        haystack = " ".join(
            [
                _as_text(node),
                _as_text(data.get("entity_id")),
                _as_text(data.get("entity_type")),
                _as_text(data.get("description")),
                _as_text(data.get("file_path")),
            ]
        ).casefold()
        if needle in haystack:
            matches.add(node)
    return matches


def _expand_neighborhood(graph: nx.Graph, seeds: set[str], depth: int) -> set[str]:
    selected = set(seeds)
    frontier = set(seeds)
    for _ in range(max(depth, 0)):
        next_frontier: set[str] = set()
        for node in frontier:
            next_frontier.update(graph.neighbors(node))
        next_frontier -= selected
        selected.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return selected


def _select_subgraph(graph: nx.Graph, limit: int, match: str | None, depth: int) -> nx.Graph:
    if match:
        selected = _expand_neighborhood(graph, _match_nodes(graph, match), depth)
        if len(selected) > limit:
            ranked = sorted(selected, key=lambda n: graph.degree(n), reverse=True)
            selected = set(ranked[:limit])
    else:
        ranked = sorted(graph.nodes, key=lambda n: graph.degree(n), reverse=True)
        selected = set(ranked[:limit])
    return graph.subgraph(selected).copy()


def _node_payload(node: str, data: dict, degree: int) -> dict:
    entity_type = _as_text(data.get("entity_type") or "default").lower()
    label = _as_text(data.get("entity_id") or node)
    description = _as_text(data.get("description"))
    file_path = _as_text(data.get("file_path"))
    title_parts = [
        f"<b>{html.escape(label)}</b>",
        f"Type: {html.escape(entity_type)}",
        f"Degree: {degree}",
    ]
    if file_path:
        title_parts.append(f"File: {html.escape(file_path)}")
    if description:
        title_parts.append(html.escape(description))
    return {
        "id": node,
        "label": label[:80],
        "title": "<br>".join(title_parts),
        "group": entity_type,
        "value": max(4, min(40, degree)),
        "color": TYPE_COLORS.get(entity_type, TYPE_COLORS["default"]),
    }


def _edge_payload(source: str, target: str, data: dict) -> dict:
    keywords = _as_text(data.get("keywords"))
    description = _as_text(data.get("description"))
    weight_raw = data.get("weight", 1)
    try:
        weight = float(weight_raw)
    except (TypeError, ValueError):
        weight = 1.0
    title = "<br>".join(
        html.escape(part)
        for part in [keywords, description]
        if part
    )
    payload = {
        "from": source,
        "to": target,
        "value": max(1, min(10, weight)),
        "title": title,
    }
    if keywords:
        payload["label"] = keywords[:40]
    return payload


def _html_document(nodes: list[dict], edges: list[dict], source: Path, graph: nx.Graph, subgraph: nx.Graph) -> str:
    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)
    source_text = html.escape(str(source))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Option 5 LightRAG Knowledge Graph</title>
  <script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
  <style>
    html, body {{ height: 100%; margin: 0; font-family: Segoe UI, Arial, sans-serif; }}
    body {{ display: grid; grid-template-rows: auto 1fr; color: #1f2933; background: #f7f8fa; }}
    header {{ padding: 12px 16px; border-bottom: 1px solid #d8dee6; background: #ffffff; }}
    h1 {{ margin: 0 0 6px; font-size: 18px; font-weight: 650; }}
    p {{ margin: 0; font-size: 13px; color: #52606d; }}
    #graph {{ width: 100%; height: 100%; background: #ffffff; }}
  </style>
</head>
<body>
  <header>
    <h1>Option 5 LightRAG Knowledge Graph</h1>
    <p>Showing {subgraph.number_of_nodes():,} of {graph.number_of_nodes():,} nodes and {subgraph.number_of_edges():,} of {graph.number_of_edges():,} edges from {source_text}. Drag nodes, scroll to zoom, hover for metadata.</p>
  </header>
  <div id="graph"></div>
  <script>
    const nodes = new vis.DataSet({nodes_json});
    const edges = new vis.DataSet({edges_json});
    const container = document.getElementById("graph");
    const data = {{ nodes, edges }};
    const options = {{
      nodes: {{ shape: "dot", font: {{ size: 13, face: "Segoe UI" }} }},
      edges: {{ color: "#a8b2bd", smooth: {{ type: "dynamic" }}, font: {{ size: 9, align: "middle" }} }},
      groups: {{
        document: {{ color: "{TYPE_COLORS['document']}" }},
        section: {{ color: "{TYPE_COLORS['section']}" }},
        table: {{ color: "{TYPE_COLORS['table']}" }},
        concept: {{ color: "{TYPE_COLORS['concept']}" }},
        entity: {{ color: "{TYPE_COLORS['entity']}" }}
      }},
      physics: {{
        stabilization: {{ iterations: 180 }},
        barnesHut: {{ gravitationalConstant: -32000, springLength: 110, springConstant: 0.025 }}
      }},
      interaction: {{ hover: true, tooltipDelay: 120, navigationButtons: true, keyboard: true }}
    }};
    new vis.Network(container, data, options);
  </script>
</body>
</html>
"""


def export_graph(graph_path: Path, output_path: Path, limit: int, match: str | None, depth: int) -> None:
    graph = nx.read_graphml(graph_path)
    subgraph = _select_subgraph(graph, limit=limit, match=match, depth=depth)
    nodes = [
        _node_payload(node, data, degree=subgraph.degree(node))
        for node, data in subgraph.nodes(data=True)
    ]
    edges = [
        _edge_payload(source, target, data)
        for source, target, data in subgraph.edges(data=True)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _html_document(nodes, edges, graph_path, graph, subgraph),
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")
    print(f"Full graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
    print(f"View graph: {subgraph.number_of_nodes():,} nodes, {subgraph.number_of_edges():,} edges")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH, help="Path to the LightRAG GraphML file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="HTML file to write.")
    parser.add_argument("--limit", type=int, default=350, help="Maximum nodes to include in the HTML view.")
    parser.add_argument("--match", help="Only show nodes whose metadata matches this text, plus neighbors.")
    parser.add_argument("--depth", type=int, default=1, help="Neighbor depth when --match is used.")
    args = parser.parse_args()

    export_graph(
        graph_path=args.graph,
        output_path=args.output,
        limit=args.limit,
        match=args.match,
        depth=args.depth,
    )


if __name__ == "__main__":
    main()
