import importlib.util,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
p=Path(__file__).with_name("retrieve.py")
spec=importlib.util.spec_from_file_location("retrieval",p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
SOURCES=[("compound_modulo_pdf","https://arxiv.org/pdf/2411.14484v1","pdf"),("compound_modulo_abs","https://arxiv.org/abs/2411.14484v1","html"),("sayplan_arxiv_pdf","https://arxiv.org/pdf/2307.06135","pdf")]
with ThreadPoolExecutor(max_workers=3) as ex:r=list(ex.map(m.get,SOURCES))
(m.ROOT/"retrieval_additional.json").write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")
for v in r:print(v["id"],v["status"],v.get("bytes",v.get("error")))
