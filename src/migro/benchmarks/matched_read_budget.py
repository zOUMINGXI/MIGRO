"""Native TransMem suffix planning with the same source-token cap as MIGRO."""
class MatchedSourceBudget:
 def __init__(self,official_plan,source_tokenizer_path,cap=2000):
  from tokenizers import Tokenizer
  self.official_plan=official_plan;self.source=Tokenizer.from_file(str(source_tokenizer_path))
  self.source.no_truncation();self.source.no_padding();self.cap=int(cap);self.last={}
 def count_source(self,text):
  return len(self.source.encode(str(text),add_special_tokens=False).ids)
 def __call__(self,tokenizer,context,questions,prompt_builder,max_prompt_tokens=128000,overflow_chunk_tokens=8192):
  if overflow_chunk_tokens is None:raise ValueError("Full history must remain encoded in native overflow memory")
  original=tokenizer.encode(context,add_special_tokens=False)
  overhead=max(int(prompt_builder(tokenizer,"",q,device=None).shape[-1]) for q in questions)
  limit=min(len(original),int(max_prompt_tokens)-overhead)
  if limit<0:raise ValueError("Question alone exceeds native model context")
  def suffix(n):
   return tokenizer.decode(original[-n:] if n else [],skip_special_tokens=False,clean_up_tokenization_spaces=False)
  def count(n):return self.count_source(suffix(n))
  hi=min(self.cap,limit);lo=0
  while hi<limit and count(hi)<=self.cap:
   lo=hi;hi=min(limit,max(hi+1,2*hi))
  if count(hi)<=self.cap:keep=hi
  else:
   while lo+1<hi:
    mid=(lo+hi)//2
    if count(mid)<=self.cap:lo=mid
    else:hi=mid
   keep=lo
  while True:
   window=self.official_plan(tokenizer,context,questions,prompt_builder,
    max_prompt_tokens=overhead+keep,overflow_chunk_tokens=overflow_chunk_tokens)
   actual=self.count_source(window.context)
   if actual<=self.cap:break
   if keep==0:raise RuntimeError("Empty context exceeds source cap")
   keep-=max(1,actual-self.cap)
  self.last=dict(source_token_cap=self.cap,raw_context_source_tokens=actual,
   source_tokenizer="MIGRO deployment/source_tokenizer/tokenizer.json",
   raw_context_host_tokens=window.kept_context_tokens,rendered_prompt_tokens_max=max(window.prompt_lengths),
   same_source_tokenizer_as_migro=True,overflow_memory_retained=True,
   cap_excludes_current_question_and_template=True)
  return window
