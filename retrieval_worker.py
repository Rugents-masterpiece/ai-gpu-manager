import os,time,logging,torch,numpy as np
from transformers import AutoModel,AutoModelForCausalLM,AutoTokenizer
from protocol import send,recv
logging.basicConfig(level=os.getenv('LOG_LEVEL','INFO'))
GPU=torch.device('cuda:0'); ROOT='/home/rugent'; EMB=os.getenv('EMB_MODEL',ROOT+'/models/huggingface/Qwen3-Embedding-0.6B'); RER=os.getenv('RER_MODEL',ROOT+'/models/huggingface/Qwen3-Reranker-0.6B')
INSTR=os.getenv('QUERY_INSTRUCTION','Given a Slovak user query, retrieve relevant technical documentation, source code, ERP documentation and software engineering knowledge.')
def pool(h,m):
 left=bool(m[:,-1].sum()==m.shape[0]);
 if left:return h[:,-1]
 n=m.sum(1)-1;return h[torch.arange(h.shape[0],device=h.device),n]
def main():
 t=time.perf_counter(); et=AutoTokenizer.from_pretrained(EMB,local_files_only=True,padding_side='left'); rt=AutoTokenizer.from_pretrained(RER,local_files_only=True,padding_side='left'); em=AutoModel.from_pretrained(EMB,local_files_only=True,torch_dtype=torch.float16).eval(); rm=AutoModelForCausalLM.from_pretrained(RER,local_files_only=True,torch_dtype=torch.float16).eval(); state='cpu_ready';send({'ok':True,'event':'cpu_ready','load_ms':(time.perf_counter()-t)*1000,'vram':torch.cuda.memory_allocated() if torch.cuda.is_available() else 0})
 while True:
  r=recv()
  if r is None:break
  op=r.get('op')
  try:
   if op=='state':send({'ok':True,'state':state})
   elif op=='activate':
    state='gpu_loading';t=time.perf_counter()
    try:
     em.to(GPU);rm.to(GPU);torch.cuda.synchronize();state='gpu_ready';send({'ok':True,'state':state,'elapsed_ms':(time.perf_counter()-t)*1000})
    except Exception as e:
     try:em.to('cpu');rm.to('cpu');torch.cuda.empty_cache()
     finally:state='cpu_ready'
     send({'ok':False,'state':state,'error':repr(e)})
   elif op=='deactivate':
    state='offloading';t=time.perf_counter();em.to('cpu');rm.to('cpu');torch.cuda.synchronize();torch.cuda.empty_cache();state='cpu_ready';send({'ok':True,'state':state,'elapsed_ms':(time.perf_counter()-t)*1000})
   elif op=='embed':
    xs=r['input'] if isinstance(r['input'],list) else [r['input']];
    if r.get('input_type')=='query':xs=[f'Instruct: {INSTR}\nQuery:{x}' for x in xs]
    b=et(xs,padding=True,truncation=True,max_length=4096,return_tensors='pt');b={k:v.to(GPU) for k,v in b.items()}
    with torch.inference_mode():a=torch.nn.functional.normalize(pool(em(**b).last_hidden_state,b['attention_mask']),p=2,dim=1).float().cpu().numpy()
    send({'ok':True,'result':{'object':'list','data':[{'object':'embedding','embedding':x.tolist(),'index':i} for i,x in enumerate(a)],'model':'qwen3-embedding-0.6b'}})
   elif op=='rerank':
    q=r['query'];docs=r['documents'];pairs=['<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no."<|im_end|>\n<|im_start|>user\n<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: '+q+'\n<Document>: '+d+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' for d in docs]; scores=[]
    with torch.inference_mode():
     for i in range(0,len(pairs),2):
      b=rt(pairs[i:i+2],padding=True,truncation=True,max_length=2048,return_tensors='pt');b={k:v.to(GPU) for k,v in b.items()}; logits=rm(**b).logits[:,-1,:]; ids=[rt.convert_tokens_to_ids('no'),rt.convert_tokens_to_ids('yes')];scores+=torch.log_softmax(logits[:,ids],1)[:,1].exp().float().cpu().tolist()
    order=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True);send({'ok':True,'result':{'model':'qwen3-reranker-0.6b','results':[{'index':i,'relevance_score':float(scores[i])} for i in order]}})
   elif op=='shutdown':send({'ok':True});break
   else:send({'ok':False,'error':'unknown_op'})
  except Exception as e: state='error';send({'ok':False,'error':repr(e),'state':state})
if __name__=='__main__':main()
