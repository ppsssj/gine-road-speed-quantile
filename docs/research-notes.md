# Research Notes: Road-Speed GNN Iteration

## Purpose and evidence boundary

This note separates paper-verified facts from the follow-up repository history and from limitations found in the current code. It does not present the historical baseline as a finished forecasting system.

- **Paper-verified statements** below come from [the conference paper](./paper/2025-kmms-road-network-design.pdf).
- **Follow-up repository statements** describe the code and research context preserved here; they are not claims that the paper proposed or evaluated those methods.

## Paper-verified system framework

**Paper:** 노재준, 박성진, 신동화, 허주완, 이현주, 이선호, 송유정, 「SUMO 기반 도시 교통 시뮬레이션을 활용한 신규 도로 개설 효과 분석 시스템 설계」, *2025년도 한국멀티미디어학회 추계학술발표대회 논문집 제28권 2호*.

The paper designs an analysis system to evaluate new-road-development effects in Asan City using traffic simulation. Its stated input is OpenStreetMap road-network data with link-level attributes such as connectivity, length, and vehicle maximum speed.

```text
OpenStreetMap road-network data
  → K-Means clustering: search-space reduction
  → GIN: link-flow estimation
  → NSGA-III: multi-objective road-candidate selection
  → SUMO: traffic-impact simulation and analysis
```

The paper is a system/framework design. Its conclusion says that future work is to implement the individual modules and validate the proposed structure through experiments. It does not report GINE, Grid Snap, kNN auxiliary edges, multi-task speed prediction, or a completed experiment for the whole pipeline.

## Research history in this repository

```text
2025 paper: system framework design
  K-Means → GIN → NSGA-III → SUMO
          ↓
Follow-up graph-learning implementation / validation attempt
          ↓
Documented road-graph connectivity and representation issues
          ↓
Grid Snap + kNN auxiliary edges + GINE + edge attributes
  + multi-task classification/regression
          ↓
Later code review
          ↓
Current-speed information leakage identified
          ↓
Leakage-safe temporal-prediction design: future work
```

### Follow-up graph-learning work

The available road data had to be converted into graphs. The repository-level follow-up notes identify endpoint-coordinate alignment, network connectivity, and the use of edge attributes as practical concerns. These observations and the corresponding implementation are subsequent work; the paper does not attribute them to its proposed framework.

### Representation iteration

The historical repository implementation added the following mechanisms:

- **Grid Snap:** merges nearby endpoint coordinates onto a fixed grid to improve graph connectivity.
- **kNN auxiliary edges:** supplements observed road edges where local proximity may help connect the graph.
- **GINE with edge attributes:** makes geometric and road-type edge features available to message passing.
- **Multi-task outputs:** combines four-class speed-quantile classification with continuous speed regression.
- **Training additions:** includes DropEdge, soft labels, focal loss, train-partition normalization, prior bias, a regression hint, and temperature scaling.

These are follow-up design choices, not paper proposals or independently established best practices for this dataset.

## Later code-review findings

### 1. Current-speed information leakage

The most important limitation is that the current speed is both input-derived information and the basis of the target.

| Location | Existing behavior |
| --- | --- |
| Node features | `spd`, `nbr_mean`, and `delta_spd` are computed from the current file's speeds. |
| Edge attributes | `spd_diff` is calculated from current endpoint speeds. |
| Regression target | `y_speed = spd`. |
| Classification target | `y` is obtained by applying same-slot `q25/q50/q75` thresholds to `spd`. |

This makes the baseline a current-state reconstruction/classification formulation. It does not support an interpretation as a model forecasting a later traffic state.

**Future design:** construct inputs from historical states (`t-n … t-1`) and predict a later state (`t`). Any speed-derived feature must be available strictly before the target time.

### 2. Distance-feature aggregation

In `load_graph_from_json`, the historical aggregation performs:

```python
dist_wsum += w
w_sum += w
dist = dist_wsum / w_sum
```

Because both accumulators receive the same distance weight, `dist` can become nearly constant (approximately 1). This can reduce the usefulness of the distance feature. It remains unchanged so that the baseline continues to reproduce its historical behavior.

### 3. Dataset split source of truth

`gine_v7.py` selects input files only with `glob(data_dir/*.json)`, sorts them, and reserves the final four files as test data. It does not load `data/train/` or `data/test/` for its split decision.

The repository currently contains both root-level JSON files and `data/train/` / `data/test/` directories. Their presence should not be taken as evidence that the directories drive the v7 split. Future work should define and version explicit train/validation/test file lists.

### 4. Validation-preprocessing leakage risk

The current code calculates per-slot quantile thresholds and class weights from all `train_paths`, then partitions those graphs into training and validation sets. This allows validation examples to influence those two values. Normalization is subsequently calculated from `tr_graphs` only.

**Future design order:**

1. Choose and freeze train/validation/test splits.
2. Compute thresholds, normalization statistics, and class weights from training data only.
3. Apply those fixed values to validation and test data.

## Preservation boundary

The following are deliberately not changed during this cleanup:

- model architecture;
- feature set and target definitions;
- temporal modeling;
- split logic;
- threshold, normalization, and distance-aggregation logic;
- training execution, checkpoints, or metrics.

`gine_v7.py` remains the historical experimental baseline. The limitations above document why a subsequent experiment should be designed separately rather than silently replacing it.
