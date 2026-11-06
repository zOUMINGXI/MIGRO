"""Adapt only current-question token embeddings; delivered evidence stays exact."""
import torch
from .host_bridge import MARKER
def question_mask(bridge,row,after):
 query=row.get("retrieval_query") or row["messages"][-1]["content"]
 start=after.find(query)
 if start<0:raise ValueError("Current question was not found in the rendered suffix")
 finish=start+len(query)
 encoded=bridge.tokenizer(after,add_special_tokens=False,return_offsets_mapping=True)
 ids=encoded["input_ids"];mask=[b>start and a<finish for a,b in encoded["offset_mapping"]]
 if not any(mask):raise ValueError("Question token span is empty")
 return ids,torch.tensor(mask,device=bridge.device,dtype=torch.bool)
def native_question_vectors(bridge,row):
 before,after=bridge.render(row,soft=True).split(MARKER)
 ids,mask=question_mask(bridge,row,after)
 return bridge.host.get_input_embeddings()(torch.tensor(ids,device=bridge.device))[mask]
def build_query_item(bridge,row,z):
 emb=bridge.host.get_input_embeddings()
 before,after=bridge.render(row,soft=True).split(MARKER)
 left=torch.tensor(bridge.ids(before),device=bridge.device);right_ids,mask=question_mask(bridge,row,after)
 right=torch.tensor(right_ids,device=bridge.device)
 values=bridge.connector.base(z.to(bridge.device)).to(emb.weight.dtype)
 tail=emb(right);question=tail[mask]
 adapted=bridge.connector.adapter(question).to(emb.weight.dtype)
 tail=tail.clone();tail[mask]=adapted
 x=torch.cat([emb(left),values,tail])
 labels=torch.full((len(x),),-100,device=bridge.device,dtype=torch.long)
 return x,labels,dict(prompt_tokens=len(left)+len(right),soft_tokens=len(values),answer_tokens=0,sequence_tokens=len(x),adapted_question_tokens=int(mask.sum()),memory_tokens_adapted=0)
