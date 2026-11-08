# performance 消融结果

|实验|状态|已完成条件与分数（LoCoMo自定义综合指标，非官方F1）|
|---|---|---|
|evidence_selection|completed|full: 54.2067；no_native_bindings: 53.4769；semantic_topk: 44.8227|
|address_representation|completed|anchor_residual: 54.2067；anchor_only: 53.7336；full_float32: 54.6708|

原主表54.1326保持冻结；已完成消融的新完整组54.2067作为组内对照。引用绑定差值区间跨0；SemanticTopK是联合选择对照。
