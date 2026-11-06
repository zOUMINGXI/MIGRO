"""Format-only MAB correction; keep original task metrics and raw predictions."""
import json,re,csv
from pathlib import Path
_LABEL=re.compile(r"(?:answer\s*:\s*)?(?:label\s*:\s*)?([0-9]+)",re.I)
def label(text):
 match=_LABEL.fullmatch(str(text).strip())
 return match.group(1) if match else None
def answer_field(text):
 value=str(text).strip();fence=chr(96)*3
 if value.startswith(fence):
  match=re.fullmatch(fence+r"(?:json)?\s*\n?(.*?)\n?"+fence,value,re.I|re.S)
  if not match:return None
  value=match.group(1).strip()
 if value.startswith(("{","[")):
  try:obj=json.loads(value)
  except json.JSONDecodeError:return None
  return obj["answer"].strip() if isinstance(obj,dict) and isinstance(obj.get("answer"),str) else None
 return value
def extract(text,sub_dataset):
 if sub_dataset.startswith("icl_"):return label(text)
 if sub_dataset=="detective_qa":return answer_field(text)
 return str(text)
def export(out,manifest,labels,predictions,strict):
 from utils.eval_other_utils import post_process
 by_case={m["case_id"]:m for m in manifest};per=[];audit=[]
 for p in predictions:
  sub=p["sub_dataset"]
  if not (sub.startswith("icl_") or sub=="detective_qa"):continue
  parsed=extract(p["prediction"],sub);meta=by_case[p["case_id"]];lab=labels[p["id"]]
  metrics,_=post_process(dict(output=parsed if parsed is not None else ""),lab["answer"],meta["config"])
  value=float(metrics[meta["metric"]]) if parsed is not None else 0.
  per.append(dict(id=p["id"],case_id=p["case_id"],sub_dataset=sub,category=meta["category"],metric=meta["metric"],score=value))
  audit.append(dict(id=p["id"],valid_format=parsed is not None,parsed_answer=parsed,metric=meta["metric"],score=value))
 profiles=[]
 for original in strict["profiles"]:
  row=dict(original);selected=[x for x in per if x["sub_dataset"]==row["sub_dataset"]]
  if selected:
   assert len(selected)==row["n"]
   row["score"]=100*sum(x["score"] for x in selected)/len(selected)
  profiles.append(row)
 categories=[]
 for category in strict["categories"]:
  selected=[p for p in profiles if p["category"]==category["category"]]
  categories.append(dict(category=category["category"],tasks=len(selected),score=sum(x["score"] for x in selected)/len(selected)))
 result=dict(strict,profiles=profiles,categories=categories,
  optional_custom_four_ability_macro=sum(x["score"] for x in categories)/4,
  scoring_variant="format-only correction: whole ICL label and JSON.answer extraction; original task metrics unchanged",
  official_unchanged_metric_reproduction=False,original_task_metrics_retained=True,
  raw_predictions_modified=False,preferred_reporting=True,
  read_budget_role=("historical native128k control" if strict["method"]=="transmem" else "see REPORTING_PROTOCOL.json"),
  original_strict_macro=strict["optional_custom_four_ability_macro"])
 out=Path(out)
 (out/"RESULTS_FORMAT_CORRECTED.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
 (out/"FORMAT_CORRECTION_AUDIT.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2))
 with (out/"results_format_corrected.csv").open("w",newline="") as f:
  writer=csv.DictWriter(f,fieldnames=list(profiles[0]));writer.writeheader();writer.writerows(profiles)
 text=["# "+strict["method"]+" MAB: format-corrected results","",
  "Preferred scoring: extract complete ICL numeric label / valid DetectiveQA JSON.answer, then apply unchanged original per-task metrics. Invalid JSON or ambiguous labels receive zero. The earlier explicit-choice metric remains a separate diagnostic and is not used here. Raw predictions and strict results are retained.",
  "","The four-ability macro is custom, not an official overall score. Auxiliary LLM judge remains disclosed as gpt-5.4-mini.","",
  "|Task|Original metric|n|Score|","|---|---|---:|---:|"]
 for p in profiles:text.append(f'|{p["sub_dataset"]}|{p["metric"]}|{p["n"]}|{p["score"]:.4f}|')
 text+=["",f'Custom four-ability macro: {result["optional_custom_four_ability_macro"]:.6f}.']
 (out/"REPORT_FORMAT_CORRECTED.md").write_text("\n".join(text)+"\n")
 return result
