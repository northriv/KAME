"""md -> docx on the previous edition's docx as reference, then made to look like it.
usage: build_manual.py <md> <reference.docx> <out.docx> <workdir> <title> <date> <author>"""
import sys, os, re, shutil, subprocess, zipfile, glob
md, ref, out, work, title, date, author = sys.argv[1:8]
shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
raw = os.path.join(work, 'pandoc.docx')
subprocess.run(['pandoc', md, '-f', 'gfm', '-t', 'docx', '--reference-doc=' + ref, '--toc', '--toc-depth=3',
                '--resource-path=' + os.path.dirname(os.path.abspath(md)), '-o', raw], check=True)
d = os.path.join(work, 'x'); os.makedirs(d)
with zipfile.ZipFile(raw) as z: z.extractall(d)
refd = os.path.join(work, 'ref'); os.makedirs(refd)
with zipfile.ZipFile(ref) as z: z.extractall(refd)
W = os.path.join(d, 'word'); doc = open(os.path.join(W, 'document.xml'), encoding='utf-8').read()
refdoc = open(os.path.join(refd, 'word', 'document.xml'), encoding='utf-8').read()

# --- root element: the reference's namespace declarations (a superset of pandoc's) ---
root_new = re.search(r'<w:document[^>]*>', doc).group(0); root_ref = re.search(r'<w:document[^>]*>', refdoc).group(0)
decl = dict(re.findall(r'(xmlns:\w+)="([^"]+)"', root_ref)); decl.update(dict(re.findall(r'(xmlns:\w+)="([^"]+)"', root_new)))
ign = re.search(r'mc:Ignorable="([^"]+)"', root_ref)
root = '<w:document ' + ' '.join(f'{k}="{v}"' for k, v in decl.items()) + (f' mc:Ignorable="{ign.group(1)}"' if ign and 'xmlns:mc' in decl else '') + '>'
doc = doc.replace(root_new, root, 1)

# --- title: the reference's title paragraph (logo + text runs), retexted ---
tp = re.search(r'<w:p [^>]*>(?:(?!</w:p>).)*?<w:pStyle w:val="a6"/>(?:(?!</w:p>).)*?</w:p>', refdoc, re.S).group(0)
texts = re.findall(r'<w:t(?: [^>]*)?>([^<]*)</w:t>', tp)
new_texts = [title, '  ' + date, '\u3000', author.split(' ')[0] + ' ', ' '.join(author.split(' ')[1:])]
def retext(p, olds, news):
    for o, n in zip(olds, news):
        p = p.replace(f'>{o}</w:t>', f'>{n}</w:t>', 1)
    return p
tp = retext(tp, texts, new_texts)
embed = re.search(r'r:embed="([^"]+)"', tp).group(1)
rels = open(os.path.join(W, '_rels', 'document.xml.rels'), encoding='utf-8').read()
assert f'Id="{embed}"' in rels, "logo relationship missing in pandoc output"
# drop the md's own title heading + the date/author line under it
m = re.search(r'<w:p><w:pPr><w:pStyle w:val="1" /></w:pPr><w:r><w:t xml:space="preserve">[^<]*Manual</w:t></w:r></w:p>\s*<w:p><w:pPr><w:pStyle w:val="FirstParagraph" /></w:pPr>.*?</w:p>', doc, re.S)
assert m, "md title block not found"; doc = doc[:m.start()] + doc[m.end():]
# --- TOC: move after the title, and give the field a cached result (Word rebuilds it on F9) ---
i = doc.find('<w:sdt>'); j = doc.find('</w:sdt>', i) + len('</w:sdt>'); sdt = doc[i:j]; doc = doc[:i] + doc[j:]
body_heads = re.findall(r'<w:p><w:pPr><w:pStyle w:val="([123])" /></w:pPr>(.*?)</w:p>', doc, re.S)
entries = [(lvl, ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', inner))) for lvl, inner in body_heads]
tocstyle = {'1': '10', '2': '20', '3': '30'}
ps = []
for k, (lvl, txt) in enumerate(entries):
    r = ''
    if k == 0: r += '<w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r><w:r><w:instrText xml:space="preserve">TOC \\o "1-3" \\h \\z \\u</w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r>'
    r += f'<w:r><w:t xml:space="preserve">{txt}</w:t></w:r>'
    if k == len(entries) - 1: r += '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
    ps.append(f'<w:p><w:pPr><w:pStyle w:val="{tocstyle[lvl]}"/></w:pPr>{r}</w:p>')
toc_heading = re.search(r'<w:p><w:pPr><w:pStyle w:val="TOCHeading" /></w:pPr>.*?</w:p>', sdt, re.S).group(0)
sdt = '<w:sdt><w:sdtPr><w:docPartObj><w:docPartGallery w:val="Table of Contents"/><w:docPartUnique/></w:docPartObj></w:sdtPr><w:sdtContent>' + toc_heading + ''.join(ps) + '</w:sdtContent></w:sdt>'
b = doc.find('<w:body>') + len('<w:body>')
doc = doc[:b] + tp + sdt + doc[b:]
# --- pandoc's paragraph styles the reference does not define ---
doc = doc.replace('<w:pStyle w:val="FirstParagraph" />', '<w:pStyle w:val="a0" />')
open(os.path.join(W, 'document.xml'), 'w', encoding='utf-8').write(doc)

# --- styles: define what pandoc referenced, on the reference's own styles; CJK font as the reference's runs had ---
st = open(os.path.join(W, 'styles.xml'), encoding='utf-8').read()
st = re.sub(r'(<w:style [^>]*w:styleId="a0"[^>]*>.*?<w:rFonts )', lambda m: m.group(1) + 'w:eastAsia="Hiragino Kaku Gothic Pro W3" ', st, count=1, flags=re.S)
add = ''
if 'w:styleId="Compact"' not in st:
    add += '<w:style w:type="paragraph" w:customStyle="1" w:styleId="Compact"><w:name w:val="Compact"/><w:basedOn w:val="a0"/><w:pPr><w:spacing w:before="0" w:after="0"/></w:pPr></w:style>'
if 'w:styleId="BlockText"' not in st:
    add += '<w:style w:type="paragraph" w:customStyle="1" w:styleId="BlockText"><w:name w:val="Block Text"/><w:basedOn w:val="a0"/><w:pPr><w:ind w:left="720" w:right="720"/></w:pPr><w:rPr><w:i/></w:rPr></w:style>'
if 'w:styleId="TOCHeading"' not in st:
    add += '<w:style w:type="paragraph" w:customStyle="1" w:styleId="TOCHeading"><w:name w:val="TOC Heading"/><w:basedOn w:val="1"/><w:pPr><w:outlineLvl w:val="9"/></w:pPr></w:style>'
if 'w:styleId="VerbatimChar"' not in st:
    add += '<w:style w:type="character" w:customStyle="1" w:styleId="VerbatimChar"><w:name w:val="Verbatim Char"/><w:rPr><w:rFonts w:ascii="Menlo" w:hAnsi="Menlo" w:eastAsia="Hiragino Kaku Gothic Pro W3" w:cs="Menlo"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:style>'
if 'w:styleId="Hyperlink"' not in st:
    add += '<w:style w:type="character" w:styleId="Hyperlink"><w:name w:val="Hyperlink"/><w:rPr><w:color w:val="0563C1"/><w:u w:val="single"/></w:rPr></w:style>'
st = re.sub(r'(<w:style [^>]*w:styleId="SourceCode"[^>]*>.*?<w:pPr>)(.*?)(</w:pPr>)(.*?)(</w:style>)',
            lambda m: m.group(1) + m.group(2) + '<w:spacing w:before="0" w:after="0"/>' + m.group(3) + '<w:rPr><w:rFonts w:ascii="Menlo" w:hAnsi="Menlo" w:eastAsia="Hiragino Kaku Gothic Pro W3" w:cs="Menlo"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>' + m.group(5), st, count=1, flags=re.S)
if 'w:styleId="Table"' not in st:
    # pandoc's tables name this style; the reference has none, so they came out without a rule anywhere.
    add += ('<w:style w:type="table" w:customStyle="1" w:styleId="Table"><w:name w:val="Table"/>'
            '<w:tblPr><w:tblBorders>'
            '<w:top w:val="single" w:sz="4" w:space="0" w:color="808080"/><w:left w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
            '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="808080"/><w:right w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
            '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="808080"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
            '</w:tblBorders><w:tblCellMar><w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style>')
st = st.replace('</w:styles>', add + '</w:styles>')
# tables: the whole text width, columns by content, rather than pandoc's dash-count widths
doc2 = open(os.path.join(W, 'document.xml'), encoding='utf-8').read()
doc2 = re.sub(r'<w:tblW [^/]*/>', '<w:tblW w:w="5000" w:type="pct"/>', doc2)
doc2 = doc2.replace('<w:tblLayout w:type="fixed" />', '<w:tblLayout w:type="autofit"/>')
open(os.path.join(W, 'document.xml'), 'w', encoding='utf-8').write(doc2)
open(os.path.join(W, 'styles.xml'), 'w', encoding='utf-8').write(st)

# --- media the document no longer references ---
used = set()
for rf in glob.glob(os.path.join(W, '_rels', '*.rels')):
    used |= set(re.findall(r'Target="media/([^"]+)"', open(rf, encoding='utf-8').read()))
removed = 0
for f in glob.glob(os.path.join(W, 'media', '*')):
    if os.path.basename(f) not in used: os.remove(f); removed += 1
# --- content types for the media pandoc brought in ---
ct_path = os.path.join(d, '[Content_Types].xml'); ct = open(ct_path, encoding='utf-8').read()
exts = {os.path.splitext(f)[1].lstrip('.').lower() for f in os.listdir(os.path.join(W, 'media'))}
mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'gif': 'image/gif', 'emf': 'image/x-emf', 'tiff': 'image/tiff', 'bmp': 'image/bmp'}
for e in sorted(exts):
    if f'Extension="{e}"' not in ct and e in mime:
        ct = ct.replace('<Default ', f'<Default Extension="{e}" ContentType="{mime[e]}"/><Default ', 1)
open(ct_path, 'w', encoding='utf-8').write(ct)
# --- pack ---
if os.path.exists(out): os.remove(out)
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for dp, dn, fn in os.walk(d):
        for f in fn:
            full = os.path.join(dp, f); z.write(full, os.path.relpath(full, d))
print(f"built {out}: {os.path.getsize(out)//1024} KB; toc entries {len(entries)}; media removed {removed}; numPr {doc.count('<w:numPr>')}; tables {doc.count('<w:tbl>')}; numbering.xml {'yes' if os.path.exists(os.path.join(W,'numbering.xml')) else 'NO'}")
