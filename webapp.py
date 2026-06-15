"""quorum web UI — optional FastAPI front-end over the quorum CLI core.

Launched by `quorum --web`. Reuses quorum.py for config + model calls, streams
each panelist's answer as it lands (SSE), then the judge's verdict.

Extra deps (not needed by the CLI): see requirements-web.txt.
Bound to 127.0.0.1 only — it runs with your keys, keep it local.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

import quorum as q

app = FastAPI(title="quorum")


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/models")
def api_models() -> JSONResponse:
    cfg = q.load_config()
    return JSONResponse({
        "panel": [{"name": m.name, "provider": m.provider, "model": m.model,
                   "auth": m.auth, "enabled": m.enabled} for m in cfg.panel],
        "judge": cfg.judge.name if cfg.judge else None,
        "judge_enabled": cfg.judge_enabled,
    })


@app.post("/api/ask")
async def api_ask(req: Request) -> StreamingResponse:
    body = await req.json()
    question = (body.get("question") or "").strip()
    names = set(body.get("panel") or [])
    judge_name = body.get("judge")
    use_judge = bool(body.get("use_judge", True))

    cfg = q.load_config()
    panel = [m for m in cfg.panel if m.name in names]
    judge = next((m for m in cfg.panel if m.name == judge_name), None) if judge_name else None

    async def gen():
        if not question:
            yield _sse({"type": "error", "message": "Empty question."}); return
        if not panel:
            yield _sse({"type": "error", "message": "No panel models selected."}); return
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=len(panel)) as ex:
            futs = [loop.run_in_executor(ex, q.run_model, m, question) for m in panel]
            for fut in asyncio.as_completed(futs):
                m = await fut
                yield _sse({"type": "model", "name": m.name, "provider": m.provider,
                            "model": m.model, "seconds": round(m.seconds, 1),
                            "answer": m.answer, "error": m.error})
        if use_judge and judge is not None and any(not m.error for m in panel):
            yield _sse({"type": "judge_start", "name": judge.name})
            text = await loop.run_in_executor(None, q.run_judge, judge, question, panel)
            yield _sse({"type": "judge", "name": judge.name, "text": text})
        yield _sse({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream")


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>quorum</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#262b36; --txt:#e6e9ef;
          --dim:#8b93a3; --acc:#4f8cff; --ok:#3fb950; --err:#f85149; --judge:#d9a441; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:18px 24px; border-bottom:1px solid var(--line); }
  header h1 { margin:0; font-size:20px; } header span { color:var(--dim); font-size:13px; }
  main { max-width:900px; margin:0 auto; padding:24px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
          padding:16px; margin-bottom:16px; }
  label { color:var(--dim); font-size:13px; display:block; margin-bottom:6px; }
  textarea { width:100%; min-height:80px; background:#0c0e12; color:var(--txt);
             border:1px solid var(--line); border-radius:8px; padding:10px; resize:vertical;
             font:inherit; }
  .models { display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 14px; }
  .chk { display:flex; align-items:center; gap:7px; background:#0c0e12; border:1px solid var(--line);
         padding:7px 11px; border-radius:20px; cursor:pointer; font-size:14px; }
  .chk input { accent-color:var(--acc); }
  .row { display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  select { background:#0c0e12; color:var(--txt); border:1px solid var(--line);
           border-radius:8px; padding:7px 10px; font:inherit; }
  button { background:var(--acc); color:#fff; border:0; border-radius:8px; padding:10px 20px;
           font:inherit; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  .res h3 { margin:0 0 4px; font-size:15px; }
  .res .meta { color:var(--dim); font-size:12px; margin-bottom:8px; }
  .res pre { white-space:pre-wrap; word-wrap:break-word; margin:0; font:inherit; }
  .judge { border-color:var(--judge); }
  .judge h3 { color:var(--judge); }
  .err { color:var(--err); }
  .ok { color:var(--ok); } .spin { color:var(--dim); }
  .foot { color:var(--dim); font-size:13px; margin-top:8px; }
  .md > :first-child { margin-top:0; } .md > :last-child { margin-bottom:0; }
  .md p { margin:0 0 10px; }
  .md h1,.md h2,.md h3,.md h4 { margin:14px 0 8px; line-height:1.3; }
  .md h1 { font-size:20px; } .md h2 { font-size:18px; }
  .md h3 { font-size:16px; } .md h4 { font-size:15px; }
  .md ul,.md ol { margin:0 0 10px; padding-left:22px; } .md li { margin:3px 0; }
  .md code { background:#0c0e12; border:1px solid var(--line); border-radius:4px;
             padding:1px 5px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.9em; }
  .md pre.code { background:#0c0e12; border:1px solid var(--line); border-radius:8px;
                 padding:12px; overflow:auto; margin:0 0 10px; }
  .md pre.code code { background:none; border:0; padding:0; font-size:.88em; }
  .md blockquote { margin:0 0 10px; padding:2px 12px; border-left:3px solid var(--line); color:var(--dim); }
  .md a { color:var(--acc); } .md strong { color:#fff; }
  .md hr { border:0; border-top:1px solid var(--line); margin:12px 0; }
</style></head>
<body>
<header><h1>quorum</h1> <span>ask a panel of LLMs · one judge synthesizes</span></header>
<main>
  <div class="card">
    <label>Question</label>
    <textarea id="q" placeholder="Ask anything…"></textarea>
    <label style="margin-top:14px">Panel</label>
    <div class="models" id="models"></div>
    <div class="row">
      <div><label>Judge</label><select id="judge"></select></div>
      <div style="margin-left:auto"><button id="ask">Ask</button></div>
    </div>
    <div class="foot" id="status"></div>
  </div>
  <div id="out"></div>
</main>
<script>
let MODELS = [];
async function loadModels() {
  const r = await fetch('/api/models'); const d = await r.json();
  MODELS = d.panel;
  const box = document.getElementById('models'); box.innerHTML='';
  d.panel.forEach(m => {
    const id='m_'+m.name;
    const el=document.createElement('label'); el.className='chk';
    el.innerHTML=`<input type="checkbox" id="${id}" ${m.enabled?'checked':''}> ${m.name}
                  <span style="color:var(--dim);font-size:12px">${m.provider}·${m.auth}</span>`;
    box.appendChild(el);
  });
  const js=document.getElementById('judge');
  js.innerHTML='<option value="">— no judge —</option>'+
    d.panel.map(m=>`<option value="${m.name}">${m.name}</option>`).join('');
  if (d.judge) js.value=d.judge;
}
function selectedPanel(){ return MODELS.filter(m=>document.getElementById('m_'+m.name).checked).map(m=>m.name); }

function card(html, cls){ const d=document.createElement('div'); d.className='card res '+(cls||''); d.innerHTML=html; return d; }
function esc(s){ return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

// Tiny self-contained Markdown → HTML (no external deps). Text is HTML-escaped
// first, then only our own tags are added, so model output can't inject HTML.
function mdToHtml(src){
  src=(src||'').replace(/\r\n?/g,'\n');
  const blocks=[];
  src=src.replace(/```[\w-]*\n?([\s\S]*?)```/g,(_,code)=>{
    blocks.push('<pre class="code"><code>'+esc(code.replace(/\n+$/,''))+'</code></pre>');
    return '\u0000B'+(blocks.length-1)+'\u0000';
  });
  function inline(t){
    const codes=[];
    t=esc(t).replace(/`([^`]+)`/g,(_,c)=>{codes.push(c);return '\u0000C'+(codes.length-1)+'\u0000';});
    t=t.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')
       .replace(/__([^_]+)__/g,'<strong>$1</strong>')
       .replace(/(^|[^*])\*([^*\n]+)\*/g,'$1<em>$2</em>')
       .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)"]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
    return t.replace(/\u0000C(\d+)\u0000/g,(_,n)=>'<code>'+codes[n]+'</code>');
  }
  const isBreak=l=>/^\s*$/.test(l)||/^\u0000B\d+\u0000$/.test(l)||/^(#{1,6})\s+/.test(l)
    ||/^\s*>\s?/.test(l)||/^\s*[-*+]\s+/.test(l)||/^\s*\d+[.)]\s+/.test(l)||/^\s*([-*_])\1\1+\s*$/.test(l);
  const lines=src.split('\n'), out=[]; let list=null;
  const closeList=()=>{ if(list){ out.push('</'+list+'>'); list=null; } };
  for(let n=0;n<lines.length;n++){
    const ln=lines[n]; let m;
    if(m=ln.match(/^\u0000B(\d+)\u0000$/)){ closeList(); out.push(blocks[+m[1]]); continue; }
    if(/^\s*$/.test(ln)){ closeList(); continue; }
    if(m=ln.match(/^(#{1,6})\s+(.*)$/)){ closeList(); const lv=m[1].length; out.push('<h'+lv+'>'+inline(m[2])+'</h'+lv+'>'); continue; }
    if(/^\s*([-*_])\1\1+\s*$/.test(ln)){ closeList(); out.push('<hr>'); continue; }
    if(m=ln.match(/^\s*>\s?(.*)$/)){ closeList(); out.push('<blockquote>'+inline(m[1])+'</blockquote>'); continue; }
    if(m=ln.match(/^\s*[-*+]\s+(.*)$/)){ if(list!=='ul'){ closeList(); out.push('<ul>'); list='ul'; } out.push('<li>'+inline(m[1])+'</li>'); continue; }
    if(m=ln.match(/^\s*\d+[.)]\s+(.*)$/)){ if(list!=='ol'){ closeList(); out.push('<ol>'); list='ol'; } out.push('<li>'+inline(m[1])+'</li>'); continue; }
    closeList();
    const para=[ln];
    while(n+1<lines.length && !isBreak(lines[n+1])){ para.push(lines[n+1]); n++; }
    out.push('<p>'+para.map(inline).join('<br>')+'</p>');
  }
  closeList();
  return out.join('\n');
}

async function ask(){
  const q=document.getElementById('q').value.trim();
  const panel=selectedPanel();
  const judge=document.getElementById('judge').value;
  const out=document.getElementById('out'); out.innerHTML='';
  const status=document.getElementById('status');
  if(!q){ status.textContent='Enter a question.'; return; }
  if(!panel.length){ status.textContent='Pick at least one model.'; return; }
  const btn=document.getElementById('ask'); btn.disabled=true; status.textContent='Asking '+panel.length+' model(s)…';

  const resp=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:q,panel:panel,judge:judge||null,use_judge:!!judge})});
  const reader=resp.body.getReader(); const dec=new TextDecoder(); let buf='';
  let judgeCard=null;
  while(true){
    const {value,done}=await reader.read(); if(done) break;
    buf+=dec.decode(value,{stream:true});
    let i;
    while((i=buf.indexOf('\n\n'))>=0){
      const line=buf.slice(0,i); buf=buf.slice(i+2);
      if(!line.startsWith('data: ')) continue;
      const o=JSON.parse(line.slice(6));
      if(o.type==='model'){
        const body = o.error ? `<pre class="err">ERROR: ${esc(o.error)}</pre>`
                             : `<div class="md">${mdToHtml(o.answer)}</div>`;
        out.appendChild(card(`<h3>${o.name} <span class="${o.error?'err':'ok'}">${o.error?'✗':'✓'}</span></h3>
          <div class="meta">${o.provider}/${o.model} · ${o.seconds}s</div>${body}`));
      } else if(o.type==='judge_start'){
        judgeCard=card(`<h3>Judge — ${o.name}</h3><pre class="spin">synthesizing…</pre>`,'judge');
        out.appendChild(judgeCard);
      } else if(o.type==='judge'){
        if(judgeCard) judgeCard.querySelector('pre').outerHTML=`<div class="md">${mdToHtml(o.text)}</div>`;
      } else if(o.type==='error'){ status.textContent=o.message; }
      else if(o.type==='done'){ status.textContent='Done.'; }
    }
  }
  btn.disabled=false;
}
document.getElementById('ask').addEventListener('click',ask);
loadModels();
</script>
</body></html>"""
