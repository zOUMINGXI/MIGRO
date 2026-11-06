"""Losslessly divide official interactions into native MIGRO memory units."""
from pathlib import Path
import json,time,argparse

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--work-dir",required=True,type=Path)
    D=p.parse_args().work_dir.resolve()
    from transformers import AutoTokenizer
    from ..backend.rd_dialogue import observation_text
    from ..backend.paths import asset_path
    tok=AutoTokenizer.from_pretrained(asset_path("research/rd_shared_pages_20261025/deployment/source_tokenizer"),local_files_only=True)
    def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
    manifest=json.loads((D/"MANIFEST.json").read_text());total=0;max_tokens=0
    for row in manifest:
     p=D/"cases"/row["case_id"];original=json.loads((p/"observations.json").read_text());units=[]
     for obs in original:
      text=obs["body"];offsets=tok(text,add_special_tokens=False,return_offsets_mapping=True)["offset_mapping"];ends=[hi for lo,hi in offsets];cuts=[0]
      for i in range(447,len(ends),448):
       if ends[i]>cuts[-1] and ends[i]<len(text):cuts.append(ends[i])
      if cuts[-1]!=len(text):cuts.append(len(text))
      pieces=[]
      for j,(lo,hi) in enumerate(zip(cuts,cuts[1:])):
       part=text[lo:hi]
       if not part:continue
       o=dict(obs,body=part,dialogue_id=obs["dialogue_id"]+"/part:"+str(j),source_turn=obs["dialogue_id"],part_index=j)
       n=len(tok.encode(observation_text(o),add_special_tokens=False));assert n<=512,(row["case_id"],n)
       max_tokens=max(max_tokens,n);pieces.append(part);units.append(o)
      assert "".join(pieces)==text
     write(p/"memory_units.json",units);row["memory_units"]=len(units);row["internal_unit_max_source_tokens"]=512
     total+=len(units);write(D/"UNIT_PREP_PROGRESS.json",dict(contexts_done=manifest.index(row)+1,units_done=total,last_case=row["case_id"],at=time.time()))
    write(D/"MANIFEST.json",manifest)
    protocol=json.loads((D/"PROTOCOL.json").read_text()) if (D/"PROTOCOL.json").exists() else {};protocol.update(internal_memory_units="Each official interaction chunk is losslessly subdivided into <=512 source-codec-token observations for native writer and 2000-token whole-record reader; original turn IDs/order retained",internal_unit_body_target_tokens=448,internal_units=total,maximum_observed_internal_unit_source_tokens=max_tokens,reranker_query_cap=128,reranker_pair_cap=384,full_final_query_preserved=True,official_chunking_and_query_labels_unchanged=True,qualification_before_first_prediction="no pre-fix predictions; old coarse states archived, units-based states rebuilt")
    write(D/"PROTOCOL.json",protocol);write(D/"UNITS_READY.json",dict(completed=True,contexts=len(manifest),original_interaction_chunks=sum(x["observations"] for x in manifest),memory_units=total,max_unit_source_tokens=max_tokens,exact_body_character_reassembly_verified=True,at=time.time()))

if __name__ == "__main__":
    main()
