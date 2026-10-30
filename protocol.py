import json,sys
PREFIX='@@JSON@@'
def send(obj):
    sys.stdout.write(PREFIX+json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n');sys.stdout.flush()
def recv():
    for line in sys.stdin:
        if line.startswith(PREFIX): return json.loads(line[len(PREFIX):])
    return None
