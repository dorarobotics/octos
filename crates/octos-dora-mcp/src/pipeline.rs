//! Lightweight DOT pipeline parser for direct tool execution.
//!
//! Parses DOT files where nodes have `tool=` and `args=` attributes,
//! executes tools in topological order via the bridge channel.
//! No LLM needed — purely deterministic.

use serde_json;
use std::collections::{HashMap, HashSet};

/// A pipeline node with optional direct tool call.
#[derive(Debug, Clone)]
pub struct DotNode {
    pub name: String,
    pub label: String,
    pub tool: Option<String>,
    pub args: Option<serde_json::Value>,
    pub node_type: String,
    pub checkpoint: Option<String>,
    pub deadline_secs: Option<u64>,
}

impl DotNode {
    pub fn is_gate(&self) -> bool {
        self.node_type == "gate" || self.node_type == "safety_gate"
    }
}

/// A parsed DOT pipeline.
#[derive(Debug)]
pub struct DotPipeline {
    pub name: String,
    pub nodes: Vec<DotNode>,
    pub edges: Vec<(String, String)>,
    pub is_cyclic: bool,
    pub max_cycles: u32,
}

impl DotPipeline {
    /// Parse a DOT file.
    pub fn from_file(path: &str) -> eyre::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_string(&text, path)
    }

    /// Parse DOT text.
    pub fn from_string(text: &str, name: &str) -> eyre::Result<Self> {
        let mut pipeline = DotPipeline {
            name: name.to_string(),
            nodes: Vec::new(),
            edges: Vec::new(),
            is_cyclic: false,
            max_cycles: 0,
        };

        // Extract graph name
        if let Some(cap) = regex_find(r"digraph\s+(\w+)\s*\{", text) {
            pipeline.name = cap;
        }

        // Graph-level attributes
        if let Some(attrs) = regex_find(r"graph\s*\[([^\]]*)\]", text) {
            if attrs.contains("cycle=true") || attrs.contains("cycle = true") {
                pipeline.is_cyclic = true;
            }
            if let Some(mc) = regex_find(r"max_cycles\s*=\s*(\d+)", &attrs) {
                pipeline.max_cycles = mc.parse().unwrap_or(0);
            }
        }

        // Parse node definitions: name [attr=val ...]
        let node_re = regex::Regex::new(r"(\w+)\s*\[([^\]]+)\]").unwrap();
        let mut node_map: HashMap<String, usize> = HashMap::new();

        for cap in node_re.captures_iter(text) {
            let node_name = cap[1].to_string();
            let attrs = &cap[2];

            // Skip graph-level
            if ["graph", "node", "edge", "digraph"].contains(&node_name.as_str()) {
                continue;
            }

            let label = parse_attr(attrs, "label").unwrap_or_else(|| node_name.clone());
            let node_type = parse_attr(attrs, "type").unwrap_or_else(|| "codergen".to_string());
            let tool = parse_attr(attrs, "tool");
            let args_str = parse_json_attr(attrs, "args");
            let args = args_str
                .and_then(|s| serde_json::from_str(&s).ok());
            let checkpoint = parse_attr(attrs, "checkpoint");
            let deadline_secs = parse_attr(attrs, "deadline")
                .and_then(|s| s.parse().ok());

            let idx = pipeline.nodes.len();
            node_map.insert(node_name.clone(), idx);
            pipeline.nodes.push(DotNode {
                name: node_name,
                label,
                tool,
                args,
                node_type,
                checkpoint,
                deadline_secs,
            });
        }

        // Parse edges
        for line in text.lines() {
            let line = line.trim().trim_end_matches(';');
            if !line.contains("->") || (line.contains('[') && line.contains(']')) {
                continue;
            }
            let parts: Vec<&str> = line.split("->").collect();
            for pair in parts.windows(2) {
                let src = pair[0].trim().to_string();
                let dst = pair[1].trim().to_string();
                if node_map.contains_key(&src) && node_map.contains_key(&dst) {
                    pipeline.edges.push((src, dst));
                }
            }
        }

        // Sort nodes in topological order
        pipeline.nodes = topological_sort(&pipeline.nodes, &pipeline.edges);

        Ok(pipeline)
    }
}

fn regex_find(pattern: &str, text: &str) -> Option<String> {
    regex::Regex::new(pattern)
        .ok()?
        .captures(text)
        .and_then(|c| c.get(1).map(|m| m.as_str().to_string()))
}

fn parse_attr(attrs: &str, key: &str) -> Option<String> {
    // Quoted: key="value"
    let re = regex::Regex::new(&format!(r#"{}s*=\s*"([^"]*)""#, key)).ok()?;
    if let Some(cap) = re.captures(attrs) {
        return Some(cap[1].to_string());
    }
    // Unquoted: key=value
    let re = regex::Regex::new(&format!(r"{}\s*=\s*(\w+)", key)).ok()?;
    re.captures(attrs).map(|c| c[1].to_string())
}

fn parse_json_attr(attrs: &str, key: &str) -> Option<String> {
    let re = regex::Regex::new(&format!(r#"{}\s*=\s*['"](\{{[^}}]*\}})['"]"#, key)).ok()?;
    re.captures(attrs).map(|c| c[1].replace("\\\"", "\""))
}

fn topological_sort(nodes: &[DotNode], edges: &[(String, String)]) -> Vec<DotNode> {
    let name_to_idx: HashMap<&str, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.name.as_str(), i))
        .collect();

    let mut in_degree: HashMap<&str, usize> = nodes.iter().map(|n| (n.name.as_str(), 0)).collect();
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();

    for (src, dst) in edges {
        *in_degree.entry(dst.as_str()).or_default() += 1;
        adj.entry(src.as_str()).or_default().push(dst.as_str());
    }

    let mut queue: Vec<&str> = in_degree
        .iter()
        .filter(|&(_, &d)| d == 0)
        .map(|(&n, _)| n)
        .collect();
    queue.sort(); // deterministic

    let mut result = Vec::new();
    let mut visited: HashSet<&str> = HashSet::new();

    while let Some(node) = queue.first().cloned() {
        queue.remove(0);
        if visited.contains(node) {
            continue;
        }
        visited.insert(node);
        if let Some(&idx) = name_to_idx.get(node) {
            result.push(nodes[idx].clone());
        }
        if let Some(neighbors) = adj.get(node) {
            for &next in neighbors {
                if let Some(d) = in_degree.get_mut(next) {
                    *d = d.saturating_sub(1);
                    if *d == 0 {
                        queue.push(next);
                    }
                }
            }
        }
    }

    // Add any unvisited nodes
    for n in nodes {
        if !visited.contains(n.name.as_str()) {
            result.push(n.clone());
        }
    }

    result
}
