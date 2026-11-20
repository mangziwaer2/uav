"""Synthetic mailbox race only: no model call, flight, field, outputs or private data."""
import json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from uav_em.event_demo import FileAdvisor

class Clock:
    def __init__(self,public):self.now=100.;self.public=public;self.stage=0
    def clock(self):return self.now
    def sleep(self,dt):
        self.now+=dt
        r=self.public/'responses'
        if self.stage==0:
            # A normal non-atomic writer has opened the receipt but not finished JSON.
            (r/'00.started.json').write_text('{"agent_id":')
        elif self.stage==1:
            (r/'00.started.json').write_text(json.dumps(dict(agent_id='synthetic',request_id='bound',started_monotonic=100.1)))
            (r/'00.raw.txt').write_text('{}');(r/'00.ready').touch()
        self.stage+=1

with tempfile.TemporaryDirectory(prefix='mailbox_review_') as folder:
    p=Path(folder);(p/'requests').mkdir();(p/'responses').mkdir();clock=Clock(p)
    advisor=FileAdvisor(p,'synthetic',clock.clock,clock.sleep)
    envelope=advisor.ask('inference',dict(request_id='bound',time_seconds=0.))
    assert clock.stage==1 and not envelope['started'] and not envelope['timely']
    assert 'Expecting' in envelope['error']
    result=dict(case='partially_written_valid_start_receipt',counterexample_produced=True,
        writer_finished=False,host_stopped_before_writer_could_complete=True,envelope=envelope,
        actual_llm_calls=0,synthetic=True,scope='possible race, not evidence that a historical request hit it')
    Path(__file__).with_name('evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
