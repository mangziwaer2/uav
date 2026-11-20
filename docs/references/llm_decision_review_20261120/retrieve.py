from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, urllib.request
ROOT=Path(__file__).resolve().parent
SOURCES=[
 ("llm_modulo_abs", "https://arxiv.org/abs/2402.01817", "html"),
 ("llm_modulo_pdf", "https://arxiv.org/pdf/2402.01817", "pdf"),
 ("sayplan_pdf", "https://proceedings.mlr.press/v229/rana23a/rana23a.pdf", "pdf"),
 ("lats_abs", "https://arxiv.org/abs/2310.04406", "html"),
 ("lats_pdf", "https://arxiv.org/pdf/2310.04406", "pdf"),
 ("tot_abs", "https://arxiv.org/abs/2305.10601", "html"),
 ("tot_pdf", "https://arxiv.org/pdf/2305.10601", "pdf"),
 ("search_modulo", "https://export.arxiv.org/api/query?search_query=ti%3A%22LLM-Modulo%22&max_results=5", "xml"),
]
def get(x):
 name,url,ext=x
 r={"id":name,"requested_url":url,"retrieved_utc":datetime.now(timezone.utc).isoformat(),"timeout_s":20}
 try:
  req=urllib.request.Request(url,headers={"User-Agent":"ResearchLiteratureReview/1.0"})
  with urllib.request.urlopen(req,timeout=20) as f:
   b=f.read();r.update(status=f.status,resolved_url=f.url,content_type=f.headers.get("Content-Type"))
  path=ROOT/(name+"."+ext);path.write_bytes(b)
  r.update(file=path.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 except Exception as e:r.update(status="failed",error=type(e).__name__+": "+str(e))
 return r
if __name__=="__main__":
 with ThreadPoolExecutor(max_workers=8) as ex:r=list(ex.map(get,SOURCES))
 (ROOT/"retrieval.json").write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")
 for v in r:print(v["id"],v["status"],v.get("bytes",v.get("error")))
