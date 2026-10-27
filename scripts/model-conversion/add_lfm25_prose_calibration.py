#!/usr/bin/env python3
"""Add pinned WikiText train/test windows without mixing calibration and heldout.

Generated passages stay in the experiment storage, not the public source tree.
The dataset's CC BY-SA/GFDL notices and immutable revision are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main(source, authored, output, provenance, data_root):
    from huggingface_hub import HfApi,hf_hub_download
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    repo='Salesforce/wikitext';revision='b08601e04326c79dfdd32d625aee71d232d685c3'
    info=HfApi().dataset_info(repo,revision=revision,files_metadata=True)
    assert info.sha==revision
    tokenizer=AutoTokenizer.from_pretrained(str(source),local_files_only=True,trust_remote_code=False)
    suite=json.loads(authored.read_text());suite['description']='Authored chat/code/multilingual diagnostic plus pinned WikiText prose; calibration and heldout remain disjoint. Not a broad capability benchmark.'
    suite['calibration_version']=3
    data_root.mkdir(parents=True,exist_ok=True)
    records=[]
    for split,dataset_split,count in [('calibration','train',128),('heldout','test',64)]:
        filename=f'wikitext-2-raw-v1/{dataset_split}-00000-of-00001.parquet'
        file=Path(hf_hub_download(repo,filename,repo_type='dataset',revision=revision,local_dir=data_root))
        with file.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        metadata=next(r for r in info.siblings if r.rfilename==filename)
        if metadata.lfs is not None:assert digest==metadata.lfs.sha256
        text='\n\n'.join(line for line in pq.read_table(file).column('text').to_pylist() if line.strip())
        tokens=tokenizer.encode(text,add_special_tokens=False)
        windows=len(tokens)//512
        selected=np.random.default_rng(20261015+(dataset_split=='test')).choice(windows,size=count,replace=False)
        records.append(dict(split=dataset_split,file=filename,bytes=file.stat().st_size,sha256=digest,windows=windows,selected_windows=selected.tolist()))
        for i,window in enumerate(selected):
            start=int(window)*512
            prefix=tokenizer.decode(tokens[start:start+448],clean_up_tokenization_spaces=False)
            continuation=tokenizer.decode(tokens[start+448:start+512],clean_up_tokenization_spaces=False)
            suite['cases'].append(dict(id=f'v3-wikitext-{split}-{i:03d}',split=split,language='en',task='prose-continuation',
                messages=[dict(role='user',content='Continue this passage in the same style.\n\n'+prefix)],
                completion='I will continue the passage in the same style.\n</think>\n'+continuation,
                dataset=dict(repository=repo,revision=revision,split=dataset_split,window=int(window))))
    readme=Path(hf_hub_download(repo,'README.md',repo_type='dataset',revision=revision,local_dir=data_root))
    with readme.open('rb') as f:readme_sha=hashlib.file_digest(f,'sha256').hexdigest()
    prov=dict(schema_version=1,repository=repo,revision=revision,url='https://huggingface.co/datasets/'+repo,
              license_notice='CC BY-SA 3.0 and GFDL; see pinned dataset README',readme_sha256=readme_sha,seed=20261015,
              calibration_split='train',heldout_split='test',window_tokens=512,continuation_tokens=64,files=records)
    suite['prose_provenance']=prov
    ids=[r['id'] for r in suite['cases']];assert len(ids)==len(set(ids))
    output.write_text(json.dumps(suite,indent=2,ensure_ascii=False)+'\n');provenance.write_text(json.dumps(prov,indent=2)+'\n')
    print(json.dumps(dict(revision=revision,cases=len(ids),calibration=sum(r['split']=='calibration' for r in suite['cases']),heldout=sum(r['split']=='heldout' for r in suite['cases'])),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','authored','output','provenance','data-root']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.source,a.authored,a.output,a.provenance,a.data_root)
