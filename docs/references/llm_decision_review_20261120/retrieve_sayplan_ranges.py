from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib,json,subprocess,re
ROOT=Path(__file__).resolve().parent
URL="https://proceedings.mlr.press/v229/rana23a/rana23a.pdf"
TOTAL=15007428;CHUNK=262144
def get(start):
 end=min(start+CHUNK,TOTAL)-1;p=ROOT/f"sayplan_chunk_{start}.part";h=p.with_suffix(".headers")
 r={"start":start,"end":end,"url":URL,"retrieved_utc":datetime.now(timezone.utc).isoformat(),"total_network_timeout_s":20}
 try:
  f=subprocess.run(["curl","--location","--fail","--silent","--show-error","--max-time","20","--connect-timeout","10","--range",f"{start}-{end}","--dump-header",str(h),"--output",str(p),URL],capture_output=True,text=True,timeout=22)
  header=h.read_text();r.update(exit_code=f.returncode,stderr=f.stderr,headers=header)
  data=p.read_bytes() if p.exists() else b""
  valid=f.returncode==0 and len(data)==end-start+1 and f"bytes {start}-{end}/{TOTAL}" in header
  r.update(status="downloaded" if valid else "failed",bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 except Exception as e:r.update(status="failed",error=type(e).__name__+": "+str(e))
 return r
entries=[]
with ThreadPoolExecutor(max_workers=8) as ex:
 for f in as_completed([ex.submit(get,start) for start in range(0,TOTAL,CHUNK)]):
  r=f.result();entries.append(r)
  (ROOT/"sayplan_ranges_retrieval.json").write_text(json.dumps(sorted(entries,key=lambda x:x["start"]),indent=2)+"\n")
  print(r["start"],r["status"],flush=True)
if all(x["status"]=="downloaded" for x in entries) and len(entries)==len(range(0,TOTAL,CHUNK)):
 etags={re.search(r'(?im)^etag: (.+)$',x["headers"]).group(1).strip() for x in entries}
 if len(etags)!=1:raise RuntimeError("Mixed versions/etags")
 data=b"".join((ROOT/f"sayplan_chunk_{start}.part").read_bytes() for start in range(0,TOTAL,CHUNK))
 assert len(data)==TOTAL and data.startswith(b"%PDF")
 p=ROOT/"sayplan_original.pdf";p.write_bytes(data)
 (ROOT/"sayplan_assembled.json").write_text(json.dumps({"url":URL,"file":p.name,"etag":etags.pop(),"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"assembly":"ordered exact nonoverlapping HTTP 206 ranges; all content-range, lengths and ETags checked"},indent=2)+"\n")
 for start in range(0,TOTAL,CHUNK):
  p=ROOT/f"sayplan_chunk_{start}.part";p.unlink();p.with_suffix(".headers").unlink()
 print("assembled",len(data),flush=True)
else:print("INCOMPLETE; no full original claimed",flush=True)
