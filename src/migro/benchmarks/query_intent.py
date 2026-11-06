"""Question-only retrieval intent; generation prompts remain untouched."""
import re
def retrieval_intent(text,profile):
 value=str(text)
 if "for example" not in value.lower():return value
 if profile=="detective_qa":
  matches=list(re.finditer(r"\bNow\s+Answer\s+the\s+Question\s*:",value,re.I))
  if matches:return value[matches[-1].end():].strip()
 if profile=="infbench_sum_eng_shots2":
  matches=list(re.finditer(r"\bNow\s+summarize\s+the\s+book\b",value,re.I))
  if matches:
   instruction=re.split(r"For\s+example\s*:",value,maxsplit=1,flags=re.I)[0].strip()
   return instruction+"\n\n"+value[matches[-1].start():].strip()
 return value
