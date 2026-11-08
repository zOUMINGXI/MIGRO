# storage 消融结果

|实验|状态|已完成条件与分数（LoCoMo自定义综合指标，非官方F1）|
|---|---|---|
|persistence_encoding|completed||

存储字节：{"full_selected_bytes": 869928, "literal_only_bytes": 952995, "uncompressed_bytes": 2381266, "causal_candidate_bytes": 869928}

因果编码相对既有无损候选的额外节省：8.716415091369845%；共享部署权重另计。

地址表示空间（附属对照）：{"cases": 10, "observations": 5882, "bytes": {"anchor_residual": 802550, "anchor_only": 560516, "full_float32": 9036512}, "scope": "standalone serialized address tables including headers; exact contents/native references/public basis unchanged and excluded; not total state archive savings", "generation_results_pending": false}

地址质量与检索一致性见性能线同一实验，不能将地址表节省写成完整档案节省。
