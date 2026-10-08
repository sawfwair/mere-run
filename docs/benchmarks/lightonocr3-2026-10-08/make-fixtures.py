from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
import json,hashlib,sys,urllib.request
r=Path(sys.argv[1]);r.mkdir(parents=True,exist_ok=True)
public_base='https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/28fe12cdf7816b5dde94e22051b2ec8dc74267b7/'
for local,remote,expected in [('receipt.jpeg','SROIE-receipt.jpeg','d66a3287c847823c6ddab9f8b2080582b96a66b7816d178472f697ea84e227b3'),('handwriting.jpeg','iam_picture.jpeg','b1b7bd2279ed653191e5861d273cf5cbe0a8b81e34c4c92debb6581a980d8d3b')]:
 data=urllib.request.urlopen(public_base+remote).read()
 if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Public fixture hash mismatch: '+remote)
 (r/local).write_bytes(data)
font='/System/Library/Fonts/Supplemental/Arial.ttf';bold='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
def f(size,b=False):return ImageFont.truetype(bold if b else font,size)
a=Image.new('RGB',(1200,1600),'white');d=ImageDraw.Draw(a)
d.text((80,70),'MERE RUN DOCUMENT CHECK',font=f(50,True),fill='black')
d.text((80,160),'Reference: OCR-3-TEST-008',font=f(30),fill='black')
d.text((80,225),'Project summary',font=f(34,True),fill='black')
for y,line in zip(range(300,525,55),['Three sensors monitor the harbour.','The trial begins on 8 October 2026.','Station Alpha recorded 17 samples.','Station Beta recorded 23 samples.']):d.text((80,y),line,font=f(28),fill='black')
d.text((670,225),'Notes françaises',font=f(34,True),fill='black')
for y,line in zip(range(300,525,55),['Température: 18,5 degrés.','Le café est ouvert à Montréal.','Équipe: Zoé, André et Émile.','Aucun incident pendant le test.']):d.text((670,y),line,font=f(27),fill='black')
d.text((80,650),'INVOICE SUMMARY',font=f(34,True),fill='black')
xs=[80,650,820,1100];ys=[725,800,875,950,1025]
for x in xs:d.line((x,ys[0],x,ys[-1]),fill='black',width=2)
for y in ys:d.line((xs[0],y,xs[-1],y),fill='black',width=2)
rows=[['Item','Quantity','Amount'],['Harbour sensor','3','$42.00'],['Calibration kit','2','$18.50'],['Total','5','$60.50']]
for y,row in zip(ys,rows):
 for x,t in zip(xs,row):d.text((x+15,y+22),t,font=f(27,y==ys[0]),fill='black')
d.text((80,1190),'Conclusion: all five units passed inspection.',font=f(30),fill='black')
d.text((80,1470),'Page 1 / 1    Verification code: MERE-1024',font=f(24),fill='black')
a.save(r/'document.png')
a=Image.new('RGB',(1200,1200),'white');d=ImageDraw.Draw(a)
d.text((95,70),'MONTHLY SAMPLE COUNTS',font=f(46,True),fill='black')
d.text((95,160),'Samples collected in the first quarter',font=f(30),fill='black')
base=930;left=180;top=300;scale=14
for v in [0,10,20,30,40]:
 y=base-v*scale;d.line((left,y,1080,y),fill='#cccccc',width=2);d.text((100,y-18),str(v),font=f(28),fill='black')
d.line((left,top,left,base),fill='black',width=4);d.line((left,base,1080,base),fill='black',width=4)
for x,name,value in [(310,'Jan',10),(610,'Feb',25),(910,'Mar',40)]:
 d.rectangle((x-70,base-value*scale,x+70,base),fill='#246bb2');d.text((x-20,base-value*scale-55),str(value),font=f(32,True),fill='black');d.text((x-25,970),name,font=f(30),fill='black')
d.text((95,1100),'Source: deterministic qualification fixture',font=f(25),fill='black')
a.save(r/'chart.png')
Image.new('RGB',(800,800),'white').save(r/'blank.png')
manifest={'public_source':{'repo':'hf-internal-testing/fixtures_ocr','revision':'28fe12cdf7816b5dde94e22051b2ec8dc74267b7'},'cases':[
 {'id':'receipt','image':'receipt.jpeg','anchors':['CASH BILL','25/12/2018','MANIS','MODELLING CLAY','KIDDY FISH','9.00'],'source':'public receipt scan'},
 {'id':'handwriting','image':'handwriting.jpeg','anchors':['industrie'],'source':'public handwriting crop; manual transcription industrie'},
 {'id':'document','image':'document.png','anchors':['MERE RUN DOCUMENT CHECK','OCR-3-TEST-008','17 samples','23 samples','Montréal','Zoé','Harbour sensor','42.00','18.50','60.50','MERE-1024'],'source':'generated known-text multicolumn and invoice fixture'},
 {'id':'chart','image':'chart.png','anchors':['MONTHLY SAMPLE COUNTS','Jan','Feb','Mar','10','25','40'],'source':'generated known-value chart fixture'},
 {'id':'blank','image':'blank.png','anchors':[],'source':'generated blank-page termination fixture'}]}
for c in manifest['cases']:
 if c['id']=='document':
  c['table_rows']=[['Harbour sensor','3','$42.00'],['Calibration kit','2','$18.50'],['Total','5','$60.50']]
  c['grounding_regions']=[{'label':'table','box':[66.667,453.125,916.667,640.625]}]
 if c['id']=='chart':
  c['table_rows']=[['Jan','10'],['Feb','25'],['Mar','40']]
  c['grounding_regions']=[{'label':'chart','box':[83.333,250,900,837.5]}]
for c in manifest['cases']:c['sha256']=hashlib.sha256((r/c['image']).read_bytes()).hexdigest()
(r/'manifest.json').write_text(json.dumps(manifest,indent=2))
