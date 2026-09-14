"""Build labeled review aids from verified originals, without modifying them."""
import argparse
import hashlib
import json
from pathlib import Path
import textwrap
from PIL import Image, ImageDraw, ImageFont, ImageOps


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True);args=p.parse_args()
    root=args.root
    inventory=json.loads((root/'annotation-audit.json').read_text())
    if hashlib.sha256(args.annotations.read_bytes()).hexdigest()!=inventory['source_sha256'][str(args.annotations)]:
        raise ValueError('Annotation source changed')
    audit=json.loads((root/'image-audit.json').read_text())
    ids=audit['unflagged_ids_in_frozen_order']
    annotations={f"viva-{r['index']}":r for r in json.loads(args.annotations.read_text())}
    out=root/'review';out.mkdir(exist_ok=False)
    font=ImageFont.load_default(size=14);heading=ImageFont.load_default(size=18)
    pages=[]
    for start in range(0,len(ids),12):
        page=Image.new('RGB',(1200,1440),'white');draw=ImageDraw.Draw(page)
        batch=ids[start:start+12]
        for offset,id in enumerate(batch):
            index=id.split('-')[-1];x=(offset%3)*400;y=(offset//3)*360
            source=root/'images'/f'{index}.image'
            receipt=json.loads((root/'receipts'/f'{index}.json').read_text())
            if hashlib.sha256(source.read_bytes()).hexdigest()!=receipt['sha256']:raise ValueError('Image changed')
            draw.text((x+10,y+5),f'{start+offset+1}. {id}',font=heading,fill='black')
            with Image.open(source) as image:
                preview=ImageOps.contain(ImageOps.exif_transpose(image).convert('RGB'),(380,225))
                page.paste(preview,(x+10+(380-preview.width)//2,y+30))
            caption=' '.join(annotations[id]['situation_description'].split())
            lines=textwrap.wrap(caption,width=51)
            if len(lines)>6:lines=lines[:5]+[lines[5][:47]+' ...']
            draw.multiline_text((x+10,y+263),'\n'.join(lines),font=font,fill='black',spacing=2)
        name=f'page-{start//12+1:02d}.jpg';page.save(out/name,quality=90)
        pages.append(dict(file=name,ids=batch,sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()))
    (out/'manifest.json').write_text(json.dumps(dict(pages=pages,review_status='not_reviewed',
        source_image_audit_sha256=hashlib.sha256((root/'image-audit.json').read_bytes()).hexdigest(),
        limits='Contact sheets only; originals unchanged; full-size follow-up required where thumbnail is insufficient'),indent=2))
    print(json.dumps(dict(pages=len(pages),images=len(ids),output=str(out))))


if __name__=='__main__':main()
