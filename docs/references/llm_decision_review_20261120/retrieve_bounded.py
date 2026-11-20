from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parent
URLS=[("sayplan_pmlr_bounded","https://proceedings.mlr.press/v229/rana23a/rana23a.pdf"),("sayplan_arxiv_bounded","https://arxiv.org/pdf/2307.06135v2")]
def get(x):
 name,url=x;p=ROOT/(name+".pdf");tmp=ROOT/(name+".partial")
 r={"id":name,"requested_url":url,"retrieved_utc":datetime.now(timezone.utc).isoformat(),"network_total_timeout_s":20}
 try:
  f=subprocess.run(["curl","--location","--fail","--silent","--show-error","--max-time","20","--connect-timeout","10","--output",str(tmp),"--write-out","%{http_code}\n%{url_effective}\n%{content_type}",url],capture_output=True,text=True,timeout=22)
  r.update(exit_code=f.returncode,response_metadata=f.stdout,stderr=f.stderr)
  if f.returncode==0 and tmp.read_bytes().startswith(b"%PDF"):
   tmp.rename(p);r.update(status="downloaded",file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
  else:r.update(status="failed")
 except Exception as e:r.update(status="failed",error=type(e).__name__+": "+str(e))
 if tmp.exists():r["discarded_partial_bytes"]=tmp.stat().st_size;tmp.unlink()
 return r
with ThreadPoolExecutor(max_workers=2) as ex:r=list(ex.map(get,URLS))
(ROOT/"retrieval_bounded.json").write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")
for v in r:print(v["id"],v["status"],v.get("bytes",v.get("stderr",v.get("error"))))
