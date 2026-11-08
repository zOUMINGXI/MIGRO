# Recorded latency audit

This is the existing November 7–8 log audit, preserved as a dated measurement set. No experiment was rerun. Progress counts describe the audit sample at collection, not current run completion.

[Six-dimension definitions and table](SIX_DIMENSIONS.md) · [Summary CSV](six_dimensions_summary.csv) · [Per-dimension coverage and definitions](six_dimensions_long.csv)

The six dimensions overlap and cannot be added: recorded write/build calls, retrieval/preparation/queue residual, generation, explicitly timed query wall time, observed history-task span, and recorded repair/retry overhead. Missing fields are not zero. Some write values are partial attempts or resumed spans; some generation values are batch allocations. Residuals are not pure retrieval time. Observed spans may include waiting and omit work before restart. A zero retry count does not establish absence of internal retries. These records do not support a hardware-normalized end-to-end speed ranking.
