# migration 消融结果

|实验|状态|已完成条件与分数（LoCoMo自定义综合指标，非官方F1）|
|---|---|---|
|host_interface_phi|completed|full: 49.7251；transport_no_calibration: 49.7251；target_tokenizer_direct: 49.8166|
|host_interface_mistral|completed|full: 45.4759；transport_no_calibration: 44.6969；target_tokenizer_direct: 44.7073|

Phi发布接口为step0，完整接口与关闭校准的残差均为零；验证后复用相同预测。Mistral非零校准可检验校准贡献。所有接口条件固定同一批Qwen证据；与历史AB全流程迁移表分开。


## Additional transport composition factorial

|Host|Stage|Condition scores|
|---|---|---|
|qwen4|completed|composed_uncalibrated: 54.2067; piecewise_uncalibrated: 53.4052|
|phi3|completed|composed_uncalibrated: 49.7251; piecewise_uncalibrated: 48.0490|
|mistral|completed|composed_calibrated: 45.4759; composed_uncalibrated: 44.6969; piecewise_calibrated: 39.9767; piecewise_uncalibrated: 39.1175|

Fixed same source packets, frozen checkpoints, native per-host answer policy. Boundary-off uses valid-vocabulary piecewise mapping; source evidence is not corrupted. Qwen/Phi identity calibration gives2unique conditions each; Mistral has4. Fresh controls, common whole-record trimming if needed. The hypothesis concerns the boundary-composition operator, not universal superiority to native text.


## Core State Transfer efficacy

|Host|Stage|Condition scores|
|---|---|---|
|qwen4|completed|state_transfer: 54.2067; bm25_text: 47.5001; recent_window: 26.2580; no_memory: 18.8993|
|phi3|completed|state_transfer: 49.7251; bm25_text: 37.9870; recent_window: 24.8312; no_memory: 30.9455; target_rebuild: 49.6506|
|mistral|completed|state_transfer: 45.4759; bm25_text: 41.5508; recent_window: 24.9603; no_memory: 19.2641; target_rebuild: 45.1922|

Same source observations, frozen weights and complete readout interface. Compare full State Transfer with all-history BM25, bounded recency and no memory. Phi/Mistral additionally use fresh target-native Full-History Rebuilding as a positive reference. Same2000source-token/50output budget; explicit per-arm native-context checks. This tests useful memory and portability relative to simple controls, not optimality.
