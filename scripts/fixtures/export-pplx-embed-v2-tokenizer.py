#!/usr/bin/env python3
"""Export real 9B tokenizer NFC/added-token fixtures; no model weights loaded.

Requires Transformers 5.4.0 and a local pinned tokenizer/config directory.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model-conversion'))
from qualify_pplx_embed_v2_mlx import prepare
from convert_pplx_embed_v2_mlx import MODELS


def export(args):
    import transformers
    from transformers import AutoTokenizer, Qwen3_5Config
    assert transformers.__version__ == '5.4.0'
    config=json.loads((args.source/'config.json').read_text())
    config['model_type']='qwen3_5'
    config.pop('auto_map',None)
    tokenizer=AutoTokenizer.from_pretrained(args.source,config=Qwen3_5Config(**config),trust_remote_code=False)
    chunks=['cafe\u0301','','\u1100\u1161','na\u0308ive 👩🏽‍💻','[Q] <|chunk_sep|>']
    ids,spans,_=prepare(tokenizer,chunks,'document',True,set())
    queries=[{'text':text,'ids':prepare(tokenizer,[text],'query',True,set())[0]} for text in chunks]
    result={'source_revision':MODELS[args.kind][1],'transformers':transformers.__version__,
            'chunks':chunks,'document_ids':ids,'spans':spans,'queries':queries}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=MODELS,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    export(parser.parse_args())
