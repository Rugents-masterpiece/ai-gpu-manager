import os,time,tempfile,torch
from nemo.collections.asr.models import ASRModel
from protocol import send,recv
MODEL=os.getenv('PARAKEET_MODEL','/home/rugent/models/huggingface/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo');GPU=torch.device('cuda:0')
def main():
 t=time.perf_counter();m=ASRModel.restore_from(restore_path=MODEL,map_location='cpu');m.eval();state='cpu_ready';send({'ok':True,'event':'cpu_ready','load_ms':(time.perf_counter()-t)*1000,'vram':torch.cuda.memory_allocated() if torch.cuda.is_available() else 0})
 while True:
  r=recv();
  if r is None:break
  try:
   op=r.get('op')
   if op=='state':send({'ok':True,'state':state})
   elif op=='activate':state='gpu_loading';t=time.perf_counter();m.cuda();torch.cuda.synchronize();state='gpu_ready';send({'ok':True,'state':state,'elapsed_ms':(time.perf_counter()-t)*1000})
   elif op=='transcribe':
    with torch.inference_mode():out=m.transcribe([r['path']],batch_size=1)
    item=out[0];send({'ok':True,'text':getattr(item,'text',str(item))})
   elif op=='retire':send({'ok':True,'state':'retiring'});break
   else:send({'ok':False,'error':'unknown_op'})
  except Exception as e:send({'ok':False,'error':repr(e),'state':state})
if __name__=='__main__':main()
