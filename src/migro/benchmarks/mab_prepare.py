"""Prepare upstream MAB profiles into separate observations, questions and labels."""
from pathlib import Path
import sys,os,json,time,traceback,codecs,collections,argparse

def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))

def question_field(row, name, index):
 """Match upstream question metadata stored at top level or under metadata."""
 value=row.get(name)
 if value is None:value=row.get("metadata",{}).get(name)
 return value[index] if isinstance(value,list) and index<len(value) else None if isinstance(value,list) else value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-root",required=True,type=Path)
    p.add_argument("--data-dir",required=True,type=Path)
    p.add_argument("--work-dir",required=True,type=Path)
    a=p.parse_args();D=a.work_dir.resolve();U=a.upstream_root.resolve()
    sys.path.insert(0,str(U));os.environ.update(USE_TF="0")
    try:
     import yaml,pyarrow.parquet as pq,nltk,tiktoken
     for name in ["punkt","punkt_tab"]:
      if not nltk.download(name,quiet=True):raise RuntimeError("NLTK resource not available: "+name)
     import conversation_creator as cc
     configs=[]
     for p in (U/"configs/data_conf").rglob("*.yaml"):
      c=yaml.safe_load(p.read_text());s=c["sub_dataset"]
      if (p.name=="Eventqa_full.yaml" or p.name=="Longmemeval_s_star.yaml" or "/Ruler/" in str(p) or "/Test_Time_Learning/" in str(p) or "/Long_Range_Understanding/" in str(p) or p.name in ["Factconsolidation_mh_262k.yaml","Factconsolidation_sh_262k.yaml"]):
       configs.append((str(p.relative_to(U)),c))
     assert len(configs)==14,len(configs)
     datasets={p.name.split("-00000")[0]:pq.read_table(p).to_pylist() for p in a.data_dir.glob("*.parquet")}
     manifest=[];summary=[];enc=tiktoken.encoding_for_model("gpt-4o-mini");oversized=[]
     for conf,cfg in sorted(configs):
      source=cfg["sub_dataset"];selected=[x for x in datasets[cfg["dataset"]] if x["metadata"]["source"]==source][:cfg["max_test_samples"]]
      assert selected,source
      cc.load_eval_data=lambda c:dict(data=selected)
      creator=cc.ConversationCreator(dict(agent_name="rag_migro"),cfg);chunks=creator.get_chunks();qas=creator.get_query_and_answers()
      for index,(row,parts,pairs) in enumerate(zip(selected,chunks,qas)):
       cid=source+"__"+str(index);folder=D/"cases"/cid;repaired=[]
       for i,part in enumerate(parts):
        ids=enc.encode(part,allowed_special={"<|endoftext|>"})
        if len(ids)>cfg["chunk_size"]:
         decoder=codecs.getincrementaldecoder("utf-8")();pieces=[]
         for j in range(0,len(ids),cfg["chunk_size"]):
          text=decoder.decode(enc.decode_bytes(ids[j:j+cfg["chunk_size"]]),final=j+cfg["chunk_size"]>=len(ids))
          if text:pieces.append(text)
         assert "".join(pieces)==part
         repaired.extend(pieces);oversized.append(dict(case_id=cid,official_chunk=i,tokens=len(ids),lossless_pieces=len(pieces)))
        elif part:repaired.append(part)
       observations=[dict(speaker="Document",role="user",body=text,session=cid,date="not specified",dialogue_id=cid+":"+str(i),incremental_turn=i) for i,text in enumerate(repaired)]
       questions=[];labels=[]
       for qi,(query,answer,qaid) in enumerate(pairs):
        qid=cid+":"+str(qi);questions.append(dict(id=qid,case_id=cid,benchmark="memoryagentbench",system=cc.get_template(source,"system","rag_migro"),messages=[dict(role="user",content=query)],retrieval_query=row["questions"][qi] if len(row["questions"])>qi else row["questions"][0]))
        md=row["metadata"];labels.append(dict(id=qid,answer=answer,qa_pair_id=qaid,raw_question=row["questions"][qi] if len(row["questions"])>qi else row["questions"][0],question_type=question_field(row,"question_types",qi),original_question_id=question_field(row,"question_ids",qi),keypoints=md.get("keypoints"),dataset=cfg["dataset"],sub_dataset=source))
       assert len(questions)==len(labels)>0
       write(folder/"observations.json",observations);write(folder/"questions.json",questions);write(folder/"labels.json",labels)
       metric="llm_judge_accuracy" if "longmemeval" in source else "llm_judge_summary_f1" if "infbench_sum" in source else "recsys_recall@5" if "recsys" in source else "exact_match" if "icl_" in source or "detective_" in source else "substring_exact_match"
       item=dict(case_id=cid,category=cfg["dataset"],sub_dataset=source,profile=conf,observations=len(observations),questions=len(questions),context_characters=len(row["context"]),context_o200k_tokens=len(enc.encode(row["context"],allowed_special={"<|endoftext|>"})),generation_max_length=cfg["generation_max_length"],metric=metric,config=cfg)
       manifest.append(item);write(D/"PREP_PROGRESS.json",dict(stage="preparing",cases=len(manifest),last_case=cid,at=time.time()))
      summary.append(dict(sub_dataset=source,category=cfg["dataset"],contexts=len(selected),questions=sum(x["questions"] for x in manifest if x["sub_dataset"]==source)))
     write(D/"MANIFEST.json",manifest);write(D/"PROFILE_SUMMARY.json",summary);write(D/"OVERSIZED_CHUNK_REPAIRS.json",oversized)
     write(D/"PREPARED.json",dict(completed=True,contexts=len(manifest),questions=sum(x["questions"] for x in manifest),observations=sum(x["observations"] for x in manifest),profiles=14,capability_groups=4,overlong_sentence_splits=len(oversized),at=time.time()))
     print(json.dumps(dict(prepared=True,profiles=summary,cases=len(manifest),questions=sum(x["questions"] for x in manifest))),flush=True)
    except Exception as e:
     write(D/"PREP_FAILURE.json",dict(error=str(e),traceback=traceback.format_exc(),at=time.time()));raise

if __name__ == "__main__":
    main()
