"""Original MAB task metrics plus disclosed format correction and auxiliary judging."""
from pathlib import Path
import os,sys,json,csv,concurrent.futures,collections,threading,argparse
from types import SimpleNamespace
from .io import predictions as read_predictions, write

def read(p):return json.loads(p.read_text())

def main():
 global D,U
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--method",default="migro")
 parser.add_argument("--work-dir",required=True,type=Path)
 parser.add_argument("--upstream-root",required=True,type=Path)
 parser.add_argument("--output",type=Path)
 parser.add_argument("--judge-model",default="gpt-5.4-mini")
 parser.add_argument("--api-key-env",default="OPENAI_API_KEY")
 parser.add_argument("--base-url")
 cli=parser.parse_args();method=cli.method;D=cli.work_dir.resolve();U=cli.upstream_root.resolve()
 out=cli.output.resolve() if cli.output else D/method
 sys.path.insert(0,str(U))
 from . import judge_client as g
 from utils.eval_other_utils import post_process
 from utils import eval_other_utils as metric_backend
 from .recsys_exact import find_nearest_movie
 metric_backend.find_nearest_movie=find_nearest_movie
 from llm_based_eval.longmem_qa_evaluate import get_anscheck_prompt
 from llm_based_eval import summarization_evaluate as sums
 os.chdir(U);movie=U/"processed_data/Recsys_Redial/entity2id.json";movie.parent.mkdir(parents=True,exist_ok=True)
 if not movie.exists():movie.write_bytes((D/"assets/entity2id.json").read_bytes())
 pred=list(read_predictions(out/"predictions.jsonl").values());manifest=read(D/"MANIFEST.json");expected=sum(x["questions"] for x in manifest)
 if len(pred)!=expected:raise ValueError(f"Incomplete prediction cohort: {len(pred)}/{expected}")
 for field in ("method","host","source_token_cap"):
  values={p[field] for p in pred if field in p}
  if len(values)>1:raise ValueError("Mixed generation settings in predictions: "+field)
 metadata={x["case_id"]:x for x in manifest};labels={lab["id"]:lab for m in manifest for lab in read(D/"cases"/m["case_id"]/"labels.json")}
 if {p["id"] for p in pred}!=set(labels):raise ValueError("Prediction IDs do not match labels")
 expected_cases={lab["id"]:m["case_id"] for m in manifest for lab in read(D/"cases"/m["case_id"]/"labels.json")}
 if any(p["case_id"]!=expected_cases[p["id"]] for p in pred):raise ValueError("Prediction case does not match its question")
 cachepath=out/"judgements.jsonl";cache={}
 if cachepath.exists():
  for line in cachepath.read_text().splitlines():
   row=json.loads(line)
   if row.get("judge_model")==cli.judge_model:cache[(row["id"],row["prediction"])]=row
 args=SimpleNamespace(provider="openai-compatible",model=cli.judge_model,api_key_env=cli.api_key_env,base_url=cli.base_url,temperature=None,reasoning_effort="low",token_limit_field="max_completion_tokens",max_tokens=8192,timeout=180.,retries=2)
 conn=None
 audit_lock=threading.Lock()
 def ask(prompt):
  nonlocal conn
  if conn is None:conn=g.connection(args)
  body=g.request_body(args.provider,prompt,args.model,None,args.max_tokens,reasoning_effort="low")
  # That connection helper belongs to a binary-verdict benchmark. Its fixed
  # response schema conflicts with MAB's native Yes/No and summary rubrics.
  body.pop("response_format",None)
  raw=g.post_json(conn["base_url"].rstrip("/")+"/chat/completions",{"Authorization":"Bearer "+conn["api_key"]},body,args.timeout,args.retries)
  text=raw["choices"][0]["message"]["content"]
  if isinstance(text,list):text="".join(x.get("text","") for x in text)
  return text
 def judge(row):
  lab=labels[row["id"]];sub=lab["sub_dataset"];output=row["prediction"];calls=[]
  if "longmemeval" in sub:
   ans=lab["answer"];ans=ans[0] if isinstance(ans,list) and len(ans)==1 else ans
   task=lab["question_type"];prompt=get_anscheck_prompt(task,lab["raw_question"],ans,output,abstention=str(lab.get("original_question_id","")).endswith("_abs"));raw=ask(prompt);calls.append(raw)
   clean=raw.strip().lower()
   if clean.startswith("yes"):value=1.
   elif clean.startswith("no"):value=0.
   else:raise ValueError("Unparsed LongMemEval judge verdict: "+raw[:500])
  else:
   keypoints=lab["keypoints"];assert keypoints
   answer=lab["answer"];answer=answer[0] if isinstance(answer,list) else answer
   prompts=[sums.fluency_prompt_book.format(text=output.strip()),sums.recall_prompt_book.format(keypoints="\n".join(f"{i+1}. {k}" for i,k in enumerate(keypoints)),summary=output.strip()),sums.precision_prompt_book.format(expert_summary=answer,summary=output.strip())]
   def judge_component(prompt,keys,valid):
    attempts=[]
    for attempt in range(4):
     raw=ask(prompt if attempt==0 else prompt+"\nReturn the final assessment as a JSON object with exactly these numeric keys: "+", ".join(keys)+". Keep the original grading rubric.")
     attempts.append(raw);objects=[];decoder=json.JSONDecoder()
     for i,ch in enumerate(raw):
      if ch!="{":continue
      try:obj,_=decoder.raw_decode(raw[i:])
      except json.JSONDecodeError:continue
      if isinstance(obj,dict) and all(k in obj and isinstance(obj[k],(int,float)) and not isinstance(obj[k],bool) for k in keys) and valid(obj):objects.append(obj)
     if objects:
      if attempt:
       with audit_lock:
        with (out/"judge_format_recoveries.jsonl").open("a") as log:log.write(json.dumps(dict(id=row["id"],component=keys,raw_attempts=attempts),ensure_ascii=False)+"\n")
      return objects[-1],raw
    with audit_lock:
     with (out/"judge_format_failures.jsonl").open("a") as log:log.write(json.dumps(dict(id=row["id"],component=keys,raw_attempts=attempts),ensure_ascii=False)+"\n")
    raise ValueError("Invalid judge response schema after4 attempts: "+str(keys))
   f,fc=judge_component(prompts[0],["fluency"],lambda x:x["fluency"] in (0,1))
   r,rc=judge_component(prompts[1],["recall"],lambda x:0<=x["recall"]<=len(keypoints))
   p,pc=judge_component(prompts[2],["precision","sentence_count"],lambda x:0<=x["precision"]<=x["sentence_count"])
   calls=[fc,rc,pc]
   recall=r["recall"]/len(keypoints);precision=p["precision"]/p["sentence_count"] if p["sentence_count"]>0 else 0.
   value=f["fluency"]*2*recall*precision/(recall+precision) if recall+precision>0 else 0.
  assert 0<=value<=1.000001
  return dict(id=row["id"],prediction=output,score=float(value),judge_model=args.model,official_judge_model_reproduction=False,raw_calls=calls)
 todo=[p for p in pred if metadata[p["case_id"]]["metric"].startswith("llm_") and (p["id"],p["prediction"]) not in cache]
 errors=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(judge,p):p["id"] for p in todo}
  for future in concurrent.futures.as_completed(futures):
   try:row=future.result()
   except Exception as e:
    errors.append(dict(id=futures[future],error=str(e)));continue
   cache[(row["id"],row["prediction"])]=row
   with cachepath.open("a") as f:f.write(json.dumps(row,ensure_ascii=False)+"\n")
   write(out/"SCORE_PROGRESS.json",dict(stage="auxiliary_judging",completed=sum(metadata[p["case_id"]]["metric"].startswith("llm_") and (p["id"],p["prediction"]) in cache for p in pred),cached_entries_total=len(cache),expected=sum(metadata[p["case_id"]]["metric"].startswith("llm_") for p in pred),judge_model=args.model,errors=errors))
 if errors:
  write(out/"SCORE_JUDGE_ERRORS.json",errors)
  raise RuntimeError("Incomplete auxiliary judgments: "+str(len(errors)))
 scored=[]
 for p in pred:
  meta=metadata[p["case_id"]];lab=labels[p["id"]];metric=meta["metric"]
  if metric.startswith("llm_"):value=cache[(p["id"],p["prediction"])]["score"]
  else:metrics,_=post_process(dict(output=p["prediction"]),lab["answer"],meta["config"]);value=float(metrics[metric])
  scored.append(dict(id=p["id"],case_id=p["case_id"],sub_dataset=meta["sub_dataset"],category=meta["category"],metric=metric,score=value))
 write(out/"scored.json",scored)
 profiles=[]
 for sub in sorted({p["sub_dataset"] for p in scored}):
  rows=[p for p in scored if p["sub_dataset"]==sub];profiles.append(dict(sub_dataset=sub,category=rows[0]["category"],metric=rows[0]["metric"],n=len(rows),score=100*sum(x["score"] for x in rows)/len(rows)))
 categories=[]
 for cat in sorted({x["category"] for x in profiles}):
  rows=[x for x in profiles if x["category"]==cat];categories.append(dict(category=cat,tasks=len(rows),score=sum(x["score"] for x in rows)/len(rows)))
 result=dict(completed=True,method=method,benchmark="MemoryAgentBench",contexts=len(manifest),questions=expected,profiles=profiles,categories=categories,optional_custom_four_ability_macro=sum(x["score"] for x in categories)/4,overall_macro_is_custom_not_official=True,auxiliary_judge_model=args.model,official_judge_model_reproduction=False,incremental=True)
 caps={p["source_token_cap"] for p in pred if "source_token_cap" in p}
 result["read_budget_role"]=f"observed-source token cap: {next(iter(caps))}" if caps else "method-native; see generation configuration"
 from .format_corrected_metrics import export as export_format_corrected
 export_format_corrected(out,manifest,labels,pred,result)
 write(out/"RESULTS.json",result);write(out/"summary.json",dict(completed=True,method=method,benchmark="memoryagentbench",questions=expected,contexts=len(manifest),profiles=14,abilities=4,judging_complete=True))
 with (out/"results.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(profiles[0]));w.writeheader();w.writerows(profiles)
 text=["# "+method+" MemoryAgentBench","",f"Fourteen task profiles, four abilities. Original task-specific metrics. Auxiliary judge: {args.model}.","","|Task|Metric|n|Score|","|---|---|---:|---:|"]
 for x in profiles:text.append(f'|{x["sub_dataset"]}|{x["metric"]}|{x["n"]}|{x["score"]:.4f}|')
 (out/"REPORT.md").write_text("\n".join(text)+"\n")
if __name__=="__main__":main()
