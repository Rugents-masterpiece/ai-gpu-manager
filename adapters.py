import os,sys,json,subprocess,threading,time,base64,signal
from typing import Literal
ROOT=os.path.dirname(__file__);PY=sys.executable
STDERR_LOG_LOCK=threading.RLock()
STDERR_LOG_HANDLERS={}
class RPC:
 def __init__(self,script,env=None,python=None):
  import queue,logging,logging.handlers
  self.lock=threading.RLock();self.retire_lock=threading.Lock();self.broken=False;self.reader_error=None;self.lines=queue.Queue(maxsize=256);self.stopped=threading.Event()
  logdir=os.getenv('WORKER_LOG_DIR',os.path.join(ROOT,'worker-logs'));os.makedirs(logdir,exist_ok=True)
  self.log_path=os.path.join(logdir,os.path.basename(script)+'.stderr.log')
  with STDERR_LOG_LOCK:
   if self.log_path not in STDERR_LOG_HANDLERS:
    handler=logging.handlers.RotatingFileHandler(self.log_path,maxBytes=1024*1024,backupCount=3,delay=True);handler.lock=STDERR_LOG_LOCK
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'));STDERR_LOG_HANDLERS[self.log_path]=handler
   self.log_handler=STDERR_LOG_HANDLERS[self.log_path]
  e=os.environ.copy();e.update(env or {});e['PYTHONUNBUFFERED']='1'
  try:self.p=subprocess.Popen([python or PY,os.path.join(ROOT,script)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1,env=e)
  except Exception:raise
  self.reader=threading.Thread(target=self._read_stdout,daemon=True);self.stderr_reader=threading.Thread(target=self._read_stderr,daemon=True);self.reader.start();self.stderr_reader.start()
  try:self._wait_event()
  except Exception:
   self.terminate(force=True);raise
 def _read_stdout(self):
  import queue
  try:
   while not self.stopped.is_set():
    line=self.p.stdout.readline()
    if not line:raise RuntimeError('worker stdout EOF')
    if line.startswith('@@JSON@@'):self.lines.put_nowait(line)
  except Exception as e:
   self.reader_error=e;self.broken=True
   try:self.lines.put_nowait(e)
   except queue.Full:pass
 def _read_stderr(self):
  import logging
  try:
   # Drain until EOF, including the final traceback after confirmed child exit.
   while True:
    text=self.p.stderr.readline(4096)
    if not text:break
    self.log_handler.handle(logging.LogRecord('worker',logging.ERROR,self.log_path,0,'pid=%s %s',(self.p.pid,text.rstrip('\n')),None))
  except (OSError,ValueError):pass
  # Registry-owned handlers are shared by generations; only logging shutdown closes them.
 def _response(self,end):
  import queue
  while True:
   if isinstance(getattr(self,'reader_error',None),queue.Full):raise RuntimeError('worker protocol response queue overflow')
   if time.monotonic()>=end:raise TimeoutError('worker RPC deadline exceeded')
   try:line=self.lines.get(timeout=min(.05,max(0,end-time.monotonic())))
   except queue.Empty:
    error=getattr(self,'reader_error',None)
    if error is not None:raise error
    continue
   if isinstance(line,Exception):raise line
   if line.startswith('@@JSON@@'):
    result=json.loads(line[8:])
    if not isinstance(result,dict):raise RuntimeError('ambiguous worker protocol')
    return result
 def _wait_event(self):
  self.initial=self._response(time.monotonic()+float(os.getenv('WORKER_STARTUP_TIMEOUT_SECONDS','900')))
  if not (self.initial.get('ok') or self.initial.get('event')):raise RuntimeError(self.initial)
 def call(self,obj,timeout=900):
  import queue
  end=time.monotonic()+timeout
  if not self.lock.acquire(timeout=max(0,timeout)):raise TimeoutError('worker RPC lock deadline')
  try:
   if self.broken or self.p.poll() is not None:raise RuntimeError('worker generation unavailable')
   while True:
    if time.monotonic()>=end:raise TimeoutError('worker RPC deadline exceeded')
    try:pending=self.lines.get_nowait()
    except queue.Empty:break
    if isinstance(pending,Exception):raise pending
    if pending.startswith('@@JSON@@'):raise RuntimeError('unsolicited worker response')
    # Ordinary worker progress printed before the request is not a reply.
   sent=queue.Queue(maxsize=1)
   def write():
    try:self.p.stdin.write('@@JSON@@'+json.dumps(obj)+'\n');self.p.stdin.flush();sent.put(None)
    except Exception as e:sent.put(e)
   threading.Thread(target=write,daemon=True).start()
   try:error=sent.get(timeout=max(0,end-time.monotonic()))
   except queue.Empty:raise TimeoutError('worker RPC write timeout')
   if error:raise error
   result=self._response(end)
   if type(result.get('ok')) is not bool or result.get('event'):raise RuntimeError('ambiguous worker response')
   if result.get('fatal_error'):self.broken=True
   return result
  except Exception:
   self.terminate(force=True);raise
  finally:self.lock.release()
 def terminate(self,force=False):
  # No cooperative RPC: even an apparently healthy process may own poisoned CUDA.
  with self.retire_lock:
   self.broken=True
   if self.p.poll() is None:
    self.p.terminate()
    try:self.p.wait(timeout=5)
    except subprocess.TimeoutExpired:self.p.kill();self.p.wait(timeout=5)
   if self.p.poll() is None:raise RuntimeError('owned worker exit unconfirmed')
   self.stopped.set()
   def close_streams():
    # TextIO.close can wait on a reader lock; never spend the exit deadline here.
    for stream,reader in ((self.p.stdin,None),(self.p.stdout,self.reader),(self.p.stderr,self.stderr_reader)):
     if reader:
      reader.join(timeout=1)
      if reader.is_alive():continue
     try:stream.close()
     except (OSError,ValueError):pass
   threading.Thread(target=close_streams,daemon=True).start()
class RetrievalWorkerAdapter:
 profile='retrieval'
 def __init__(self):self.w=RPC('retrieval_worker.py');self.state='cpu_ready'
 def health(self):return {'state':('error' if getattr(self.w,'broken',False) else self.state),'alive':self.w.p.poll() is None}
 def activate_gpu(self):
  r=self.w.call({'op':'activate'},int(os.getenv('RETRIEVAL_ACTIVATE_TIMEOUT_SECONDS','60')))
  if not r.get('ok'):self.state=r.get('state','cpu_ready');raise RuntimeError(r.get('error','retrieval activation failed'))
  self.state=r['state'];return r
 def deactivate_gpu(self):r=self.w.call({'op':'deactivate'},int(os.getenv('RETRIEVAL_DEACTIVATE_TIMEOUT_SECONDS','60')));self.state=r['state'];return r
 def embed(self,r):return self.w.call({'op':'embed',**r})
 def rerank(self,r):return self.w.call({'op':'rerank',**r})
 def retire(self):self.w.terminate(force=True)
 def close(self):self.w.terminate()
 def restart(self):self.close();self.w=RPC('retrieval_worker.py');self.state='cpu_ready'
class TTSWorkerAdapter:
 profile='tts'
 def __init__(self):
  self.python=os.getenv('CHATTERBOX_PYTHON','/home/rugent/chatterbox-sk/.venv/bin/python');self.app=os.getenv('CHATTERBOX_APP','/home/rugent/chatterbox-sk/app');self.w=RPC('chatterbox_worker.py',{'CHATTERBOX_APP':self.app,'CHATTERBOX_PROJECT':'/home/rugent/chatterbox-sk','CUDA_VISIBLE_DEVICES':os.getenv('CUDA_VISIBLE_DEVICES','GPU-66081c19-9aca-8ca3-2cd6-20a4eea6c7d1'),'GPU_UUID':os.getenv('GPU_UUID','GPU-66081c19-9aca-8ca3-2cd6-20a4eea6c7d1')},python=self.python);self.state='cpu_ready';self.last=getattr(self.w,'initial',{})
 def health(self):return {**self.last,'state':('error' if getattr(self.w,'broken',False) else self.state),'alive':self.w.p.poll() is None,'engine':'chatterbox-slovak','python':self.python}
 def activate_gpu(self):
  r=self.w.call({'op':'activate'},int(os.getenv('TTS_ACTIVATE_TIMEOUT_SECONDS','60')))
  if not r.get('ok'):self.state=r.get('state','cpu_ready');self.last=r;raise RuntimeError(r.get('error','tts activation failed'))
  self.state=r['state'];self.last=r;return r
 def deactivate_gpu(self):
  r=self.w.call({'op':'deactivate'},int(os.getenv('TTS_DEACTIVATE_TIMEOUT_SECONDS','60')))
  if not r.get('ok'):self.state=r.get('state','error');self.last=r;raise RuntimeError(r.get('error','tts deactivation failed'))
  self.state=r['state'];self.last=r;return r
 def synth(self,text,normalize=True,seed=None,voice:Literal['rudo','melisko','fico']='rudo'):
  if voice not in ('rudo','melisko','fico'):raise ValueError(f'unsupported voice: {voice!r}')
  r=self.w.call({'op':'synth','input':text,'normalize':bool(normalize),'seed':seed,'voice':voice});
  if not r.get('ok'):self.state=r.get('state',self.state);self.last={**self.last,**r};raise RuntimeError(r.get('error','tts worker error'))
  for key in ('error','error_type','diagnostics'):self.last.pop(key,None)
  self.last.update(ok=True,fatal_error=False,last_metrics=r.get('metrics',{}))
  if 'request_id' in r:self.last['request_id']=r['request_id']
  return r
 def retire(self):self.w.terminate(force=True)
 def close(self):self.w.terminate(force=bool(self.last.get('fatal_error') or self.state=='error'))
 def restart(self):
  self.close();self.w=RPC('chatterbox_worker.py',{'CHATTERBOX_APP':self.app,'CHATTERBOX_PROJECT':'/home/rugent/chatterbox-sk','CUDA_VISIBLE_DEVICES':os.getenv('CUDA_VISIBLE_DEVICES','GPU-66081c19-9aca-8ca3-2cd6-20a4eea6c7d1'),'GPU_UUID':os.getenv('GPU_UUID','GPU-66081c19-9aca-8ca3-2cd6-20a4eea6c7d1')},python=self.python);self.state='cpu_ready';self.last=getattr(self.w,'initial',{})
class STTWorkerAdapter:
 profile='stt'
 def __init__(self):
  self.generation=0;self.lock=threading.Lock();self.active=None;self.current=None;self.standby=None;self._spawn_current()
 def _new(self):self.generation+=1;return RPC(os.getenv('STT_WORKER_SCRIPT','whisperx_worker.py'),{'STT_MODEL_DIR':os.getenv('STT_MODEL_DIR','/home/rugent/models/huggingface'),'HF_HOME':os.getenv('HF_HOME','/home/rugent/.cache/huggingface')},python=os.getenv('STT_PYTHON','/home/rugent/ai-stack/stt-whisperx/.venv/bin/python'))
 def _spawn_current(self):self.current=self._new()
 def health(self):
  w=self.active or self.current
  return {'state':('error' if getattr(w,'broken',False) else ('gpu_ready' if self.active else 'cpu_ready')),'alive':bool(w and w.p.poll() is None),'generation':self.generation,'cpu_ready':bool(self.current and self.current.p.poll() is None),'standby_alive':bool(self.standby and self.standby.p.poll() is None)}
 def activate_gpu(self):
  with self.lock:
   if self.active:return {'state':'gpu_ready','generation':self.generation}
   new_standby=self._new()
   try:
    self.active=self.current;self.current=None
    r=self.active.call({'op':'activate'},int(os.getenv('STT_ACTIVATE_TIMEOUT_SECONDS','90')))
    self.standby=new_standby;return r
   except Exception:
    new_standby.terminate();self.current=self.active;self.active=None;raise
 def transcribe(self,path,**kw):
  if not self.active:raise RuntimeError('STT worker is not GPU active')
  return self.active.call({'op':'transcribe','path':path,**kw},int(os.getenv('STT_TRANSCRIBE_TIMEOUT_SECONDS','1800')))
 def retire_gpu_worker(self):
  with self.lock:
   if not self.active:return {'state':'cpu_ready','generation':self.generation}
   old=self.active
   try:old.call({'op':'retire'},int(os.getenv('STT_RETIRE_TIMEOUT_SECONDS','90')))
   finally:
    try:old.p.wait(90)
    except subprocess.TimeoutExpired:old.p.kill();old.p.wait(5)
   if old.p.poll() is None:raise RuntimeError('owned STT worker exit unconfirmed')
   self.active=None;self.current=self.standby;self.standby=None
   if not self.current or self.current.p.poll() is not None:self._spawn_current()
   return {'state':'cpu_ready','generation':self.generation}
 def retire(self):
  for w in (self.active,self.current,self.standby):
   if w:w.terminate(force=True)
 def close(self):
  for w in (self.active,self.current,self.standby):
   if w:w.terminate()
 def restart(self):
  self.close();self.active=None;self.current=None;self.standby=None;self._spawn_current()




