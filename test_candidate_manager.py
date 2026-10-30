import asyncio,threading,time
from manager_app import Manager
class Fake:
 def __init__(self,name):self.name=name;self.state='cpu_ready';self.calls=[]
 def health(self):return {'state':self.state,'alive':True}
 def activate_gpu(self):self.calls.append('activate');self.state='gpu_ready';return {'ok':True}
 def deactivate_gpu(self):self.calls.append('deactivate');self.state='cpu_ready';return {'ok':True}
 def retire_gpu_worker(self):self.calls.append('retire');self.state='cpu_ready';return {'ok':True}

def test_fifo_and_exclusive():
 m=Manager(idle=999);m.workers={p:Fake(p) for p in ['retrieval','stt','tts']};out=[]
 for p in ['retrieval','tts','retrieval','stt','tts','retrieval']:
  out.append(m.submit(p,lambda p=p:p,timeout=10))
 assert out==['retrieval','tts','retrieval','stt','tts','retrieval'];assert m.switches==6;m.close()
def test_idle_timeout_never_deactivates_active_request():
 from threading import Event, Thread
 m=Manager(idle=0);m.workers={p:Fake(p) for p in ['retrieval','tts','stt']}
 entered=Event();release=Event();errors=[]
 def work():
  m.last_used['retrieval']=time.time()-10
  entered.set();release.wait(3);return 'done'
 def submit():
  try:m.submit('retrieval',work,timeout=5)
  except Exception as e:errors.append(e)
 try:
  m.submit('retrieval',lambda:'warmup',timeout=5)
  t=Thread(target=submit);t.start();assert entered.wait(2)
  time.sleep(1.2)
  assert m.workers['retrieval'].state=='gpu_ready'
  assert m.active_profile=='retrieval'
  release.set();t.join(3)
  assert not t.is_alive() and not errors
 finally:
  release.set();m.close()

def test_activation_failure_with_unknown_device_state_keeps_lease():
 m=Manager(idle=999);m.workers={p:Fake(p) for p in ['retrieval','tts','stt']}
 def partial_failure():
  m.workers['tts'].state='error';raise RuntimeError('incomplete CUDA transition')
 m.workers['tts'].activate_gpu=partial_failure
 try:m.submit('tts',lambda:None,timeout=2)
 except RuntimeError:pass
 else:raise AssertionError('failure not surfaced')
 assert m.active_profile=='tts'
 assert m.transition=='error'
 m.close()

def test_activation_rollback():
 m=Manager(idle=999);m.workers={'retrieval':Fake('retrieval'),'tts':Fake('tts'),'stt':Fake('stt')};m.workers['tts'].activate_gpu=lambda: (_ for _ in ()).throw(RuntimeError('boom'))
 try:m.submit('tts',lambda:None,timeout=2)
 except RuntimeError:pass
 else:raise AssertionError('failure not surfaced')
 assert m.active_profile is None;m.close()
if __name__=='__main__':test_fifo_and_exclusive();test_activation_rollback();print('manager unit PASS')
