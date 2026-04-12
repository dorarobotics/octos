# SLAM Navigation Visual Simulation

A minimal visual demo where you can **see** a Hunter SE robot navigate through a MuJoCo warehouse — stripped from the full 16-node `octos_inspection` dataflow to 9 nodes (nav + sim + viz only).

## What It Does

The robot follows a 1065-waypoint path through a simulated warehouse, visiting stations A (y=10m) and B (y=18m). Rerun shows the live 3D scene: robot body, LiDAR pointcloud, planned path, and trajectory trail.

## Architecture: 9-Node Dataflow

```
                    ┌─────────────┐
                    │ mujoco-sim  │ (10ms tick, Hunter SE + warehouse)
                    └──┬──────┬───┘
                       │      │
          ground_truth_pose   └─── pointcloud ──→ ┌────────┐
                       │                          │ rerun  │ (3D viz)
                       ↓                          └────────┘
              ┌─────────────────┐                      ↑ raw_path
              │ road-lane-pub   │←── road_lane ── pub-road (200ms)
              └────────┬────────┘
                       │ cur_pose_all
                       ↓
              ┌─────────────────┐
              │    planning     │←── road_attri ── task-pub-stub (100ms)
              └───┬─────────┬───┘
           raw_path│         │Request
                  ↓         ↓
           ┌──────────┐ ┌──────────┐
           │lat-control│ │lon-control│
           └─────┬────┘ └─────┬────┘
         SteeringCmd     TrqBreCmd
                  ↓         ↓
              ┌─────────────────┐
              │  moveit-skills  │ (nav forwarding → wheel_commands)
              └────────┬────────┘
                       │ wheel_control
                       ↓
                  (back to mujoco-sim)
```

**Removed from octos_inspection** (7 nodes): `iot-gateway`, `cloud-brain`, `planning-scene`, `planner`, `ik-solver`, `trajectory-executor` (arm pipeline).

## Prerequisites

- [dora-rs](https://github.com/dora-rs/dora) (`pip install dora-rs`)
- [dora-nav](https://github.com/dorarobotics/dora-nav) — built C++ nodes
- [dora-moveit2](https://github.com/dorarobotics/dora-moveit2) — MuJoCo Python sim
- MuJoCo (installed via `pip install mujoco`)
- [Rerun](https://rerun.io) (`pip install rerun-sdk`)
- Python packages: `pyarrow`, `numpy`

## Quick Start

### 1. Manual Mode (no LLM — robot follows full path)

```bash
# Set paths to your built dependencies
export DORA_NAV_PATH=/home/demo/Public/dora-nav
export DORA_MOVEIT2_PATH=/home/demo/Public/dora-moveit2

cd /path/to/octos
dora up && dora start examples/slam-nav-sim/dataflow_nav_sim.yaml --attach
# → Rerun opens showing robot driving through warehouse
# → Ctrl+C to stop
```

### 2. Autonomous Mode (with LLM agent)

Uncomment the `robot-edge-a` node in `dataflow_nav_sim.yaml`, then:

```bash
export OPENAI_API_BASE=http://your-vllm-server:4567/v1
export OPENAI_MODEL=robobrain

USER_COMMAND="navigate to station A, check lidar, then return home" \
dora up && dora start examples/slam-nav-sim/dataflow_nav_sim.yaml --attach
# → Robot navigates to A (y=10m), reads LiDAR, returns home (y=0.28m)
```

## What You'll See in Rerun

| Visual | Color | Description |
|--------|-------|-------------|
| Global path | White line | 1065-waypoint route through warehouse |
| Static map | Red points | Pre-built map from `data/map.pcd` |
| Live LiDAR | Green points | Real-time 360° pointcloud |
| Planned path | Blue line | Current path segment from planner |
| Robot body | Yellow box | Hunter SE bounding box |
| Heading | Cyan arrow | Robot forward direction |
| Trail | Orange line | Position history |

## Octos Safety Features

This example demonstrates several octos safety concepts:

- **Safety tiers** (`nav_tool_map.json`): `navigate_to` requires `safe_motion` tier; `get_map`/`read_lidar` only need `observe`
- **Workspace bounds**: Station positions define the safe navigation corridor
- **Stall detection**: 10-second timeout if robot stops moving unexpectedly
- **Pipeline deadlines** (`patrol_mission.dot`): 120s per navigation leg, 300s mission total
- **Invariants**: Post-conditions verify robot reached target station

## Files

| File | Purpose |
|------|---------|
| `dataflow_nav_sim.yaml` | 9-node dora dataflow topology |
| `nodes/moveit_skill_node.py` | Nav skill bridge (SteeringCmd/TrqBreCmd → wheels) |
| `nodes/octos_robot_edge_node.py` | LLM agent (optional, ONE-SHOT mode) |
| `nodes/task_pub_stub.py` | Road attribute publisher |
| `nodes/rerun_viz_node.py` | Rerun 3D visualizer |
| `nodes/octos/` | Vendored octos agent framework |
| `nav_tool_map.json` | 6 tool mappings with safety tiers |
| `patrol_mission.dot` | DOT pipeline with deadlines + invariants |
| `skills/robot/navigation/SKILL.md` | Navigation skill definition |

## Comparison to Full Example

| Feature | slam-nav-sim (this) | octos_inspection (full) |
|---------|-------------------|------------------------|
| Nodes | 9 | 16 |
| Arm control | No | Yes (GEN72 7-DOF) |
| Cloud brain | No | Yes (task decomposition) |
| HTTP gateway | No | Yes (Flask :5000) |
| LLM agent | Optional | Required |
| Navigation | Yes | Yes |
| Rerun 3D | Yes | Yes |

See [`octos_inspection`](https://github.com/dorarobotics/octos_inspection) for the full 16-node version with arm + cloud brain.
