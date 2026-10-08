import json,time,sys
from pathlib import Path
import torch,transformers
from transformers import AutoModelForImageTextToText,AutoProcessor
from PIL import Image
root=sys.argv[1]
image_path=sys.argv[2]
output_path=sys.argv[3]
processor=AutoProcessor.from_pretrained(root,local_files_only=True)
model=AutoModelForImageTextToText.from_pretrained(root,dtype=torch.float32,device_map='cpu',local_files_only=True,attn_implementation='eager')
model.eval()
records=[]
for mode in ['plain','grounding']:
 content=[{'type':'image','image':Image.open(image_path).convert('RGB')}]
 if mode=='grounding':content.append({'type':'text','text':'grounding'})
 inputs=processor.apply_chat_template([{'role':'user','content':content}],tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=True,enable_thinking=False)
 start=time.monotonic()
 with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=64,do_sample=False,use_cache=True)
 generated=output[0,inputs['input_ids'].shape[1]:]
 text=processor.decode(generated,skip_special_tokens=True)
 row={'model':'lightonai/LightOnOCR-3-0.8B','revision':'be8cee5d200b80218cb2865a5deeec1fe6e25f52','mode':mode,'device':'cpu','dtype':'float32','attention':'eager','torch':torch.__version__,'transformers':transformers.__version__,'seconds':time.monotonic()-start,'tokensGenerated':len(generated),'imageGridTHW':inputs['image_grid_thw'].tolist(),'text':text}
 print(json.dumps(row),flush=True);records.append(row)
Path(output_path).write_text(json.dumps(records,indent=2)+'\n')
