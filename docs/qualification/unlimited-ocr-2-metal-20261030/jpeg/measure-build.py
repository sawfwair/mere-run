import concurrent.futures, json, pathlib, subprocess, tempfile, time
root=pathlib.Path.cwd(); codec=root/'Sources/CMereRunJPEG'; sources=[codec/'decode.c',*sorted((codec/'compile').glob('*.c'))]
scratch=pathlib.Path(tempfile.mkdtemp(prefix='mererun-jpeg-build-'))
common=['clang','-O0','-g','-I'+str(codec/'include'),'-I'+str(codec/'libjpeg'),'-I'+str(codec/'libjpeg/src')]
objects=[scratch/(str(i)+'.o') for i in range(len(sources))]
def run(cmd): subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
def compile_one(pair):
 source,obj=pair; run(common+['-c',str(source),'-o',str(obj)])
def link(): run(['clang',*(str(o) for o in objects),'-I'+str(codec/'include'),str(root/'Tests/NativeJPEG/decoder.c'),'-Wl,-dead_strip','-o',str(scratch/'decoder')])
clean=[]; incremental=[]
for repeat in range(3):
 start=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(compile_one,zip(sources,objects)))
 link(); clean.append(time.monotonic()-start)
 start=time.monotonic(); compile_one((sources[0],objects[0])); link(); incremental.append(time.monotonic()-start)
(scratch/'empty.c').write_text('int main(void) { return 0; }\n')
run(['clang','-O0','-g',str(scratch/'empty.c'),'-o',str(scratch/'empty')])
result=dict(scope='Private C decoder only; not a clean Swift package benchmark', baseline='Empty C executable with the same compiler/configuration', configuration='Apple clang -O0 -g; 8 compile workers; dead-stripped link', repetitions=3,cleanSeconds=clean,incrementalSeconds=incremental,incrementalEdit='Recompile decode.c and relink, preserving upstream objects',emptyExecutableBytes=(scratch/'empty').stat().st_size,decoderExecutableBytes=(scratch/'decoder').stat().st_size,cliExecutableBytes=(root/'.build/debug/mere.run').stat().st_size,compiler=subprocess.check_output(['clang','--version'],text=True),baseCommit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),retainedArtifacts='OS compiler/cache and Swift/MLX dependency artifacts; each clean C sample recompiles all 55 translation units',limitations='Concurrent receipt qualification and repository checks; warm filesystem caches. Full-package clean/incremental before/after times and historical CLI size delta were not measured. No throughput claim.')
result['cleanRangeSeconds']=[min(clean),max(clean)]; result['incrementalRangeSeconds']=[min(incremental),max(incremental)]
out=root/'docs/qualification/unlimited-ocr-2-metal-20261030/jpeg/build-measurements.json'; out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
