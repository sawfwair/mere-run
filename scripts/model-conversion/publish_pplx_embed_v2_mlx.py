#!/usr/bin/env python3
"""Stage privately, verify every file hash/size, then release a pinned artifact."""
import argparse
import json
from pathlib import Path
from convert_pplx_embed_v2_mlx import ARTIFACTS, MODELS, checksums, digest, write_json


def verify(api, repository, revision, files):
    from huggingface_hub import hf_hub_download
    info = api.model_info(repository, revision=revision, files_metadata=True)
    remote = {p.rfilename:p for p in info.siblings}
    for name, expected in files.items():
        row = remote[name]
        if row.size != expected['bytes']: raise ValueError('Remote size mismatch: '+name)
        sha = row.lfs.sha256 if row.lfs else digest(Path(hf_hub_download(repository,name,revision=info.sha,token=api.token)))
        if sha != expected['sha256']: raise ValueError('Remote hash mismatch: '+name)
    return info.sha


def verify_local_manifest(root):
    for line in (root/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1)
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Invalid checksum path')
        if digest(root/name)!=expected: raise ValueError('Local artifact hash mismatch: '+name)


def publish(args):
    root = args.artifact
    verify_local_manifest(root)
    conversion = json.loads((root/'PPLX_CONVERSION.json').read_text())
    report = json.loads((root/'PPLX_QUALIFICATION.json').read_text())
    if (conversion['source_revision'] != MODELS[args.kind][1] or conversion['kind'] != args.kind
        or report['conversion_sha256'] != digest(root/'PPLX_CONVERSION.json')
        or report['config_sha256'] != digest(root/'config.json')
        or report['index_sha256'] != digest(root/'model.safetensors.index.json')
        or not report.get('passes_diagnostic_gates',report.get('passes_heldout_gates',False)) or not all(report['gates'].values())):
        raise ValueError('Pinned conversion or diagnostic evidence failed validation')
    suffix='late-9b' if args.kind=='late' else 'context-9b-preview'
    bits=json.loads((root/'config.json').read_text())['quantization']['bits']
    precision_suffix='8bit' if bits==8 else 'mixed-4bit'
    write_json(root/'mererun_model.json', {'schemaVersion':3,'id':'text-embed-pplx-v2-'+suffix+'-'+precision_suffix,
        'engine':'pplx-embed-v2','family':'embed','tier':'latest','variant':'standard','precision':'int8' if bits==8 else 'int4',
        'supports':['text_embedding','multimodal_embedding'] if args.kind=='late' else ['text_embedding'],
        'components':{'tokenizer':{'type':'local','path':'.'},'text_encoder':{'type':'local','path':'.'}},
        'upstreamRepoId':MODELS[args.kind][0]+'@'+MODELS[args.kind][1],'createdAt':'2026-10-09T00:00:00Z'})
    import shutil
    shutil.copyfile(__file__, root/Path(__file__).name)
    if args.apple_evidence:
        apple=json.loads(args.apple_evidence.read_text())
        if (not apple['passes'] or apple['kind']!=args.kind
            or apple['reference_sha256']!=digest(root/'PPLX_REFERENCE_VECTORS.json')):
            raise ValueError('Apple evidence does not match the measured artifact')
        public={k:v for k,v in apple.items() if k not in ['artifact_path','machine_memory_bytes','results']}
        public['scope']=('Full-checkpoint native query/chunk parity.' if args.kind=='context' else
                         'Full-checkpoint native text parity and one synthetic image smoke.')+ \
                        ' Maximum context and general retrieval quality unqualified.'
        public['results']=[{k:v for k,v in row.items() if k!='timing_and_memory'} for row in apple['results']]
        import re
        for original,row in zip(apple['results'],public['results']):
            for label,key in [('maximum resident set size','maximum_resident_bytes'),('peak memory footprint','peak_memory_footprint_bytes')]:
                match=re.search(r'(\d+)\s+'+label,original['timing_and_memory'])
                if match: row[key]=int(match.group(1))
        write_json(root/'PPLX_APPLE_QUALIFICATION.json',public)
        card=root/'README.md'
        text=card.read_text()
        if '## Apple validation' not in text:
            card.write_text(text+'\n## Apple validation\n\n'
                'Full-checkpoint native CLI parity passed. '
                'See PPLX_APPLE_QUALIFICATION.json for measured process memory, timings and scope. '
                'This covers the included short text/chunk suite and, for late, one synthetic image; '
                'maximum context and general retrieval quality remain unqualified.\n')
    files = checksums(root)
    files['SHA256SUMS']={'bytes':(root/'SHA256SUMS').stat().st_size,'sha256':digest(root/'SHA256SUMS')}
    if args.prepare_only:
        print(json.dumps({'prepared':args.kind,'files':len(files)}))
        return
    if not args.apple_evidence or not args.token_file:
        raise ValueError('Publication requires local Apple evidence and a local token file')
    from huggingface_hub import HfApi
    api = HfApi(token=args.token_file.read_text().strip())
    repository=ARTIFACTS[args.kind]
    if conversion['artifact_repository'] != repository: raise ValueError('Artifact profile does not match publication destination')
    api.create_repo(repository,repo_type='model',private=True,exist_ok=True)
    if not api.model_info(repository).private: raise ValueError('Publication requires private staging')
    api.upload_large_folder(repo_id=repository,repo_type='model',folder_path=root,
                            allow_patterns=list(files),num_workers=4,print_report=True,print_report_every=60)
    revision=verify(api,repository,'main',files)
    api.update_repo_settings(repository,repo_type='model',private=False)
    public=HfApi(token=False)
    assert not public.model_info(repository,revision=revision).private
    verify(public,repository,revision,files)
    receipt={'repository':repository,'revision':revision,'public':True,'verified_files':files}
    if args.receipt: write_json(args.receipt,receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='verified_files'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=ARTIFACTS,required=True)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--token-file',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--apple-evidence',type=Path)
    parser.add_argument('--receipt',type=Path)
    publish(parser.parse_args())
