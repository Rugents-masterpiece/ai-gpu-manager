import os,time,threading,queue,uuid,tempfile,shutil,subprocess,json,base64
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Literal
from adapters import RetrievalWorkerAdapter,TTSWorkerAdapter,STTWorkerAdapter
class Job:
 def __init__(self,profile,fn,deadline):self.id=uuid.uuid4().hex[:12];self.profile=profile;self.fn=fn;self.deadline=deadline;self.enq=time.time();self.done=threading.Event();self.result=None;self.error=None
class Manager:
 def __init__(self,idle=300):
  self.idle=idle;self.profile_idle={'tts':max(1,int(os.getenv('TTS_IDLE_TIMEOUT_SECONDS','60')))};self.q=queue.Queue();self.active_profile=None;self.transition='none';self.state='idle';self.last_used={};self.switches=0;self.events=[];self.workers={};self.running=True;self.accepting=True;self.active_job=None;self.stop_lock=threading.Lock();self.lifecycle_lock=threading.RLock();self.recovery_after={};self.cached_snapshot={};self._cache_snapshot();self.t=threading.Thread(target=self._loop,daemon=True);self.t.start();self.idle_thread=threading.Thread(target=self._idle_loop,daemon=True);self.idle_thread.start()
 def start_workers(self):
  self.workers={'retrieval':RetrievalWorkerAdapter(),'tts':TTSWorkerAdapter(),'stt':STTWorkerAdapter()};self._event('workers_cpu_ready');self.reconcile()
 def reconcile(self):
  owners=[]
  try:
   uuid=os.getenv('GPU_UUID','GPU-66081c19-9aca-8ca3-2cd6-20a4eea6c7d1')
   out=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True,stderr=subprocess.DEVNULL,timeout=5)
   owners=[x.strip() for x in out.splitlines() if x.strip()]
  except Exception as e:self._event('reconciliation_failed',error=repr(e));return {'owners':owners,'error':repr(e)}
  self._event('reconciliation',owners=owners,active_profile=self.active_profile)
  return {'owners':owners,'active_profile':self.active_profile}
 def _event(self,name,**kw):self.events.append({'timestamp':time.time(),'event':name,'active_profile':self.active_profile,**kw});self.events=self.events[-1000:]
 def submit(self,profile,fn,timeout=900):
  with self.stop_lock:
   if not self.accepting:raise RuntimeError('manager is shutting down')
   j=Job(profile,fn,time.time()+timeout);self.q.put(j)
  if not j.done.wait(timeout):raise TimeoutError(j.id)
  if j.error:raise j.error
  return j.result
 def _bad(self,p):
  h=self.workers[p].health()
  return not h.get('alive',True) or h.get('state')=='error' or h.get('fatal_error',False)
 def _recover(self,p):
  with self.lifecycle_lock:
   if time.monotonic()<self.recovery_after.get(p,0):raise RuntimeError('worker recovery backoff')
   self.transition='recovering';self.state='recovering';self._event('worker_recovery_started',profile=p);self._cache_snapshot()
   try:
    # Retirement is separate from replacement: never relinquish a live owner.
    self.workers[p].retire()
    if self.active_profile==p:self.active_profile=None
    self._cache_snapshot();self.workers[p].restart()
    if self._bad(p):raise RuntimeError('replacement worker unhealthy')
    self.transition='none';self.state='idle';self.recovery_after.pop(p,None);self._event('worker_recovered',profile=p)
   except Exception as e:
    self.recovery_after[p]=time.monotonic()+5;self.transition='error';self.state='error';self._event('recovery_failed',profile=p,error=repr(e));raise
 def _activate(self,p):
  with self.lifecycle_lock:
   old=self.active_profile
   if old and self._bad(old):self._recover(old)
   if self._bad(p):self._recover(p)
   if self.active_profile==p and self.workers[p].health().get('state')=='gpu_ready':return
   if self.active_profile:
    old=self.active_profile;self.transition='deactivating';self._event('profile_deactivating',profile=old);self._cache_snapshot()
    try:
     r=self.workers[old].retire_gpu_worker() if old=='stt' else self.workers[old].deactivate_gpu()
     if r.get('ok') is False:raise RuntimeError(r.get('error','offload failed'))
    except Exception:
     self._recover(old)
    else:self.active_profile=None;self._event('profile_deactivated',profile=old,result=r)
   self.transition='activating';self._event('profile_activating',profile=p)
   try:
    # Reserve ownership before activation; an ambiguous failure must be retired.
    self.active_profile=p;self._cache_snapshot();r=self.workers[p].activate_gpu()
    if r.get('ok') is False:raise RuntimeError(r.get('error','activation failed'))
    self.switches+=1;self.transition='none';self._event('profile_active',profile=p,result=r)
   except Exception:
    self._recover(p);raise
 def _loop(self):
  while self.running:
   j=self.q.get()
   if j is None:break
   with self.lifecycle_lock:
    self.active_job=j
    try:
     if not self.accepting:raise RuntimeError('manager shutting down')
     if time.time()>j.deadline:raise TimeoutError(j.id)
     self._activate(j.profile);self.state='busy';self._event('request_started',profile=j.profile,request_id=j.id,queue_depth=self.q.qsize());self._cache_snapshot()
     j.result=j.fn();self.last_used[j.profile]=time.time();self.state='active';self._event('request_finished',profile=j.profile,request_id=j.id)
    except Exception as e:
     j.error=e;self._event('request_failed',profile=j.profile,error=repr(e))
     try:
      owner=self.active_profile
      if owner and self._bad(owner):self._recover(owner)
      elif self._bad(j.profile):self._recover(j.profile)
      elif self.transition!='error':self.state='active' if owner else 'idle'
     except Exception as e2:self.state='error';self.transition='error';self._event('recovery_failed',error=repr(e2))
    finally:
     self.active_job=None;self._cache_snapshot();j.done.set()
 def _idle_tick(self):
  with self.lifecycle_lock:
   if not self.running or not self.accepting or self.active_job is not None or not self.q.empty():return
   try:
    p=self.active_profile
    if not p:return
    if self._bad(p):self._recover(p);return
    if time.time()-self.last_used.get(p,time.time())<=self.profile_idle.get(p,self.idle):return
    self.transition='deactivating';self._event('idle_timeout',profile=p);self._cache_snapshot()
    try:
     r=self.workers[p].retire_gpu_worker() if p=='stt' else self.workers[p].deactivate_gpu()
     if r.get('ok') is False:raise RuntimeError(r.get('error','offload failed'))
    except Exception:self._recover(p)
    else:self.active_profile=None;self.transition='none';self.state='idle';self._event('profile_deactivated',profile=p)
   except Exception as e:self.transition='error';self.state='error';self._event('idle_failed',error=repr(e))
   finally:self._cache_snapshot()
 def _idle_loop(self):
  while self.running:
   time.sleep(1);self._idle_tick()
 def _cache_snapshot(self):
  workers={}
  for k,v in self.workers.items():
   try:workers[k]=v.health()
   except Exception as e:workers[k]={'state':'error','alive':False,'error':repr(e)}
  self.cached_snapshot={'status':'shutting_down' if not self.accepting else ('healthy' if self.running and self.state!='error' and self.transition=='none' else 'degraded'),'active_profile':self.active_profile,'transition':None if self.transition=='none' else self.transition,'queue_depth':self.q.qsize(),'state':self.state,'accepting':self.accepting,'workers':workers,'switch_count':self.switches}
 def snapshot(self):
  import copy
  if self.lifecycle_lock.acquire(blocking=False):
   try:self._cache_snapshot()
   finally:self.lifecycle_lock.release()
  result=copy.deepcopy(self.cached_snapshot);result['queue_depth']=self.q.qsize();result['accepting']=self.accepting
  if not self.accepting:result['status']='shutting_down'
  return result
 def close(self,timeout=120):
  with self.stop_lock:self.accepting=False
  if not self.lifecycle_lock.acquire(timeout=max(0,timeout)):raise TimeoutError('manager shutdown waiting for active job')
  try:
   self.state='shutting_down';self._event('shutdown_started')
   while True:
    try:
     j=self.q.get_nowait()
     if j is not None:j.error=RuntimeError('manager shutting down');j.done.set()
    except queue.Empty:break
   for p,w in self.workers.items():
    w.retire()
    if self.active_profile==p:self.active_profile=None
   self.running=False;self.q.put(None);self._event('shutdown_complete');self._cache_snapshot()
  finally:self.lifecycle_lock.release()
class EmbReq(BaseModel):input:str|list[str];model:str|None=None;input_type:str|None='document'
class RankReq(BaseModel):query:str;documents:list[str];model:str|None=None;top_n:int|None=None
class SpeechReq(BaseModel):input:str;normalize:bool=True;seed:int|None=None;voice:Literal['rudo','melisko','fico']='rudo'
app=FastAPI(title='RTX3050 shadow GPU manager');M=None
@app.on_event('startup')
def startup():
 global M
 M=Manager(idle=int(os.getenv('GPU_IDLE_TIMEOUT_SECONDS','300')));M.start_workers()
@app.on_event('shutdown')
def shutdown():
 if M is not None:
  M.close()
  for w in M.workers.values():w.close()
@app.get('/health')
def health():return M.snapshot()
@app.get('/ready')
def ready():
 s=M.snapshot();tts=s.get('workers',{}).get('tts',{});return {'state':tts.get('state','unavailable'),'can_accept_request':bool(s.get('accepting') and s['status']=='healthy'),'gpu_ready':tts.get('state')=='gpu_ready','active_profile':s['active_profile'],'queue_depth':s['queue_depth'],'workers':s['workers']}
@app.get('/status')
def status():return M.snapshot()
@app.get('/events')
def events():return {'events':M.events[-100:]}
@app.get('/metrics')
def metrics():return {'profile_switch_count':M.switches,'queue_depth':M.q.qsize(),'active_profile':M.active_profile}
@app.post('/v1/embeddings')
def embeddings(r:EmbReq):
 try:
  x=M.submit('retrieval',lambda:M.workers['retrieval'].embed(r.model_dump()));return x['result']
 except Exception as e:raise HTTPException(503,str(e))
@app.post('/v1/rerank')
@app.post('/rerank')
def rerank(r:RankReq):
 try:
  x=M.submit('retrieval',lambda:M.workers['retrieval'].rerank(r.model_dump()));return x['result']
 except Exception as e:raise HTTPException(503,str(e))
@app.post('/v1/audio/transcriptions')
def stt(file:UploadFile=File(...),language:str|None=None,diarize:bool=False,align:bool=True,telephone_mode:bool=False,num_speakers:int|None=None,min_speakers:int|None=None,max_speakers:int|None=None,compute_type:str|None=None):
 path=tempfile.mktemp(prefix='shadow-stt-',suffix=os.path.splitext(file.filename or '.wav')[1]);
 try:
  open(path,'wb').write(file.file.read())
  opts={'language':language or 'sk','diarize':diarize,'align':align,'telephone_mode':telephone_mode}
  if num_speakers is not None:opts['num_speakers']=num_speakers
  if min_speakers is not None:opts['min_speakers']=min_speakers
  if max_speakers is not None:opts['max_speakers']=max_speakers
  if compute_type:opts['compute_type']=compute_type
  return M.submit('stt',lambda:M.workers['stt'].transcribe(path,**opts),timeout=int(os.getenv('STT_JOB_TIMEOUT_SECONDS','1800')))
 except Exception as e:raise HTTPException(503,str(e))
 finally:
  try:os.unlink(path)
  except OSError:pass
@app.post('/v1/audio/speech')
def tts(r:SpeechReq):
 try:
  x=M.submit('tts',lambda:M.workers['tts'].synth(r.input,normalize=r.normalize,seed=r.seed,voice=r.voice));metrics=x.get('metrics',{});M._event('tts_quality',**metrics);return Response(base64.b64decode(x['audio_b64']),media_type='audio/wav',headers={'X-Profile':'tts','X-Voice':x.get('voice',r.voice),'X-Voice-Reference-SHA256':x.get('reference_sha256',''),'X-Inference-Ms':str(x.get('elapsed_ms')),'X-Request-ID':str(x.get('request_id','')),'X-Quality-Warnings':str(metrics.get('forced_eos_count',0)+metrics.get('repetition_count',0))})
 except Exception as e:raise HTTPException(503,str(e))



