# Graph Report - _graphsrc  (2026-07-22)

## Corpus Check
- Corpus is ~7,980 words - fits in a single context window. You may not need a graph.

## Summary
- 78 nodes · 93 edges · 7 communities (6 shown, 1 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- NB0 — EDA & data prep (LiTSDataset, 2.5D loader, perceptual-hash dedup, mask fusion)
- NB3 — Cross-model inference & per-image metrics (DeepLab/SegFormer/YOLO)
- NB2 — SegFormer-B0 training & evaluation (ConfMat, Dice+CE, TTA, tumor-F2)
- NB1 — DeepLabV3 training & evaluation (ConfMat, Dice+CE, TTA, tumor-F2)
- Shared LiTSDataset / 2.5D loader (a)
- Shared LiTSDataset / 2.5D loader (b)
- Shared ConfMat confusion-matrix metric class

## God Nodes (most connected - your core abstractions)
1. `run_eval()` - 6 edges
2. `LiTSDataset` - 5 edges
3. `ConfMat` - 5 edges
4. `run_eval()` - 5 edges
5. `LiTSDataset` - 5 edges
6. `ConfMat` - 5 edges
7. `LiTSDataset` - 4 edges
8. `ConfMat` - 4 edges
9. `prep()` - 4 edges
10. `criterion()` - 3 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Communities (7 total, 1 thin omitted)

### Community 0 - "NB0 — EDA & data prep (LiTSDataset, 2.5D loader, perceptual-hash dedup, mask fusion)"
Cohesion: 0.13
Nodes (4): combined_label(), LiTSDataset, load_25d(), load_bin()

### Community 1 - "NB3 — Cross-model inference & per-image metrics (DeepLab/SegFormer/YOLO)"
Cohesion: 0.15
Nodes (4): pred_dl(), pred_sf(), prep(), _r25()

### Community 2 - "NB2 — SegFormer-B0 training & evaluation (ConfMat, Dice+CE, TTA, tumor-F2)"
Cohesion: 0.20
Nodes (6): ConfMat, criterion(), dice_loss(), forward_logits(), run_eval(), tta_pred()

### Community 3 - "NB1 — DeepLabV3 training & evaluation (ConfMat, Dice+CE, TTA, tumor-F2)"
Cohesion: 0.20
Nodes (4): ConfMat, criterion(), dice_loss(), run_eval()

### Community 4 - "Shared LiTSDataset / 2.5D loader (a)"
Cohesion: 0.33
Nodes (3): LiTSDataset, load_25d(), Dataset

### Community 5 - "Shared LiTSDataset / 2.5D loader (b)"
Cohesion: 0.33
Nodes (3): LiTSDataset, load_25d(), Dataset

## Knowledge Gaps
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LiTSDataset` connect `Shared LiTSDataset / 2.5D loader (b)` to `NB2 — SegFormer-B0 training & evaluation (ConfMat, Dice+CE, TTA, tumor-F2)`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Why does `LiTSDataset` connect `Shared LiTSDataset / 2.5D loader (a)` to `NB1 — DeepLabV3 training & evaluation (ConfMat, Dice+CE, TTA, tumor-F2)`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Why does `ConfMat` connect `Shared ConfMat confusion-matrix metric class` to `NB3 — Cross-model inference & per-image metrics (DeepLab/SegFormer/YOLO)`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._
- **Should `NB0 — EDA & data prep (LiTSDataset, 2.5D loader, perceptual-hash dedup, mask fusion)` be split into smaller, more focused modules?**
  _Cohesion score 0.1323529411764706 - nodes in this community are weakly interconnected._