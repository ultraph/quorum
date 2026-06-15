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

    # attached files (read client-side as text) get appended to the question, so
    # both the panel and the judge analyze them.
    file_parts = []
    for a in (body.get("attachments") or []):
        text = a.get("text") or ""
        if not text.strip():
            continue
        fname = (a.get("name") or "file").strip() or "file"
        file_parts.append(f"=== ATTACHED FILE: {fname} ===\n{text}")
    files_block = "\n\n".join(file_parts)
    full_question = (((question or "Analyze the attached file(s).") + "\n\n" + files_block)
                     if files_block else question)

    cfg = q.load_config()
    panel = [m for m in cfg.panel if m.name in names]
    judge = next((m for m in cfg.panel if m.name == judge_name), None) if judge_name else None

    # pasted outside opinions become synthetic panelists: not called, but shown
    # as cards and handed to the judge alongside the live models.
    extras = []
    for e in (body.get("extra") or []):
        text = (e.get("text") or "").strip()
        if not text:
            continue
        label = (e.get("name") or "external").strip() or "external"
        em = q.Model(name=label, provider="external", model="pasted")
        em.answer = text
        extras.append(em)

    async def gen():
        if not full_question.strip():
            yield _sse({"type": "error", "message": "Enter a question or attach a file."}); return
        if not panel and not extras:
            yield _sse({"type": "error", "message": "Pick a model or add an external answer."}); return
        loop = asyncio.get_event_loop()
        if panel:
            with ThreadPoolExecutor(max_workers=len(panel)) as ex:
                futs = [loop.run_in_executor(ex, q.run_model, m, full_question) for m in panel]
                for fut in asyncio.as_completed(futs):
                    m = await fut
                    yield _sse({"type": "model", "name": m.name, "provider": m.provider,
                                "model": m.model, "seconds": round(m.seconds, 1),
                                "answer": m.answer, "error": m.error,
                                "tokens_in": m.tokens_in, "tokens_out": m.tokens_out,
                                "tokens_est": m.tokens_est})
        for m in extras:
            yield _sse({"type": "model", "name": m.name, "provider": "external",
                        "model": "pasted", "seconds": 0, "answer": m.answer,
                        "error": "", "external": True})
        judge_panel = panel + extras
        if use_judge and judge is not None and any(not m.error for m in judge_panel):
            yield _sse({"type": "judge_start", "name": judge.name})
            text = await loop.run_in_executor(None, q.run_judge, judge, full_question, judge_panel)
            yield _sse({"type": "judge", "name": judge.name, "text": text,
                        "tokens_in": judge.tokens_in, "tokens_out": judge.tokens_out,
                        "tokens_est": judge.tokens_est})
        yield _sse({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream")


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>quorum</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#262b36; --txt:#e6e9ef;
          --dim:#8b93a3; --acc:#4f8cff; --ok:#3fb950; --err:#f85149; --judge:#d9a441;
          --inbg:#0c0e12; --hover:#1c2029; --btntxt:#fff; --strong:#fff; }
  :root[data-theme="light"] {
          --bg:#f4f6f9; --panel:#ffffff; --line:#d3dae3; --txt:#1b2027;
          --dim:#586173; --acc:#2563eb; --ok:#197f34; --err:#c81e26; --judge:#9a6a00;
          --inbg:#eef1f6; --hover:#e7ecf3; --btntxt:#ffffff; --strong:#0a0d12; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:18px 24px; border-bottom:1px solid var(--line);
           display:flex; align-items:center; gap:16px; }
  header h1 { margin:0; font-size:20px; font-weight:700;
    background:linear-gradient(90deg,var(--acc),var(--judge));
    -webkit-background-clip:text; background-clip:text;
    -webkit-text-fill-color:transparent; color:transparent; }
  header span { color:var(--dim); font-size:13px; }
  .theme { background:var(--inbg); color:var(--txt); border:1px solid var(--line);
           border-radius:8px; padding:6px 10px; font-size:16px; font-weight:400;
           line-height:1; cursor:pointer; }
  .theme:hover { background:var(--hover); }
  main { max-width:900px; margin:0 auto; padding:24px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
          padding:16px; margin-bottom:16px; }
  label { color:var(--dim); font-size:13px; display:block; margin-bottom:6px; }
  textarea { width:100%; min-height:80px; background:var(--inbg); color:var(--txt);
             border:1px solid var(--line); border-radius:8px; padding:10px; resize:vertical;
             font:inherit; }
  .models { display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 14px; }
  .chk { display:flex; align-items:center; gap:7px; background:var(--inbg); border:1px solid var(--line);
         padding:7px 11px; border-radius:20px; cursor:pointer; font-size:14px;
         transition:border-color .15s ease, background .15s ease; }
  .chk:hover { border-color:var(--acc); }
  .chk input { accent-color:var(--acc); }
  .dot { width:9px; height:9px; border-radius:50%; flex:none; display:inline-block; }
  .row { display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  select { background:var(--inbg); color:var(--txt); border:1px solid var(--line);
           border-radius:8px; padding:7px 10px; font:inherit; }
  button { background:var(--acc); color:var(--btntxt); border:0; border-radius:8px; padding:10px 20px;
           font:inherit; font-weight:600; cursor:pointer;
           transition:transform .1s ease, filter .15s ease, opacity .15s ease; }
  button:hover:not(:disabled) { transform:translateY(-1px); filter:brightness(1.08); }
  button:active:not(:disabled) { transform:translateY(0); }
  button:disabled { opacity:.5; cursor:default; }
  .res h3 { margin:0 0 4px; font-size:15px; }
  .res .meta { color:var(--dim); font-size:12px; margin-bottom:8px; }
  .res pre { white-space:pre-wrap; word-wrap:break-word; margin:0; font:inherit; }
  .judge { border-color:var(--judge); }
  .judge h3 { color:var(--judge); }
  details.card { padding:0; }
  details.card > summary { padding:14px 16px; cursor:pointer; list-style:none;
    display:flex; align-items:center; gap:10px; }
  details.card > summary::-webkit-details-marker { display:none; }
  details.card > summary::before { content:'▶'; color:var(--dim); font-size:10px;
    transition:transform .2s ease; }
  details.card[open] > summary::before { transform:rotate(90deg); }
  details.card > summary { transition:background .15s ease; }
  details.card > summary:hover { background:var(--hover); }
  details.card > summary .name { font-weight:600; font-size:15px; }
  details.card > summary .meta { margin-left:auto; margin-bottom:0; }
  details.card > .md, details.card > pre { padding:0 16px 16px; margin:0; }
  .err { color:var(--err); }
  .ok { color:var(--ok); } .spin { color:var(--dim); }
  .foot { color:var(--dim); font-size:13px; margin-top:8px; }
  .md > :first-child { margin-top:0; } .md > :last-child { margin-bottom:0; }
  .md p { margin:0 0 10px; }
  .md h1,.md h2,.md h3,.md h4 { margin:14px 0 8px; line-height:1.3; }
  .md h1 { font-size:20px; } .md h2 { font-size:18px; }
  .md h3 { font-size:16px; } .md h4 { font-size:15px; }
  .md ul,.md ol { margin:0 0 10px; padding-left:22px; } .md li { margin:3px 0; }
  .md code { background:var(--inbg); border:1px solid var(--line); border-radius:4px;
             padding:1px 5px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.9em; }
  .md pre.code { background:var(--inbg); border:1px solid var(--line); border-radius:8px;
                 padding:12px; overflow:auto; margin:0 0 10px; }
  .md pre.code code { background:none; border:0; padding:0; font-size:.88em; }
  .md blockquote { margin:0 0 10px; padding:2px 12px; border-left:3px solid var(--line); color:var(--dim); }
  .md a { color:var(--acc); } .md strong { color:var(--strong); }
  .md hr { border:0; border-top:1px solid var(--line); margin:12px 0; }

  /* --- liveliness: motion, skeletons, status --- */
  @keyframes cardIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:none; } }
  @keyframes fadeIn { from { opacity:0; } to { opacity:1; } }
  @keyframes shimmer { from { background-position:-450px 0; } to { background-position:450px 0; } }
  @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:.3; } }
  @keyframes ellipsis { 0% { content:''; } 25% { content:'·'; } 50% { content:'··'; } 75%,100% { content:'···'; } }
  @media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }

  .res { animation:cardIn .35s ease both; transition:box-shadow .2s ease, border-color .2s ease; }
  .res:hover { box-shadow:0 4px 18px rgba(0,0,0,.16); }
  details.card[open] > .md, details.card[open] > pre { animation:fadeIn .25s ease both; }
  summary .dot { margin-right:2px; }

  .pending .pendrow { display:flex; align-items:center; gap:10px; padding:14px 16px; }
  .pending .name { font-weight:600; font-size:15px; }
  .pending .timer { margin-left:auto; font-variant-numeric:tabular-nums; }
  .pulse { animation:blink 1.2s ease-in-out infinite; }
  .dots::after { content:''; animation:ellipsis 1.3s steps(1,end) infinite; }
  .shimmer { height:9px; border-radius:5px; margin:7px 16px;
    background:linear-gradient(90deg, var(--inbg) 25%, var(--hover) 37%, var(--inbg) 63%);
    background-size:900px 100%; animation:shimmer 1.4s linear infinite; }
  .shimmer.s2 { width:82%; } .shimmer.s3 { width:55%; margin-bottom:16px; }

  .banner { display:inline-flex; align-items:center; gap:7px; padding:4px 12px; border-radius:20px;
    font-size:13px; font-weight:600; border:1px solid var(--line); }
  .banner.ok { color:var(--ok); border-color:var(--ok); }
  .banner.warn { color:var(--judge); border-color:var(--judge); }
  .banner.err { color:var(--err); border-color:var(--err); }

  /* external (pasted) answers input */
  .ghost { background:var(--inbg); color:var(--txt); border:1px solid var(--line);
           font-weight:500; padding:7px 12px; font-size:13px; }
  .ghost:hover:not(:disabled) { background:var(--hover); filter:none; }
  .extra { display:flex; flex-direction:column; gap:6px; border:1px solid var(--line);
           border-radius:8px; padding:10px; margin-bottom:8px; background:var(--inbg); }
  .extra .exrow { display:flex; gap:8px; align-items:center; }
  .extra input { flex:1; background:var(--panel); color:var(--txt); border:1px solid var(--line);
                 border-radius:6px; padding:6px 9px; font:inherit; font-size:13px; }
  .extra textarea { min-height:70px; background:var(--panel); color:var(--txt);
                    border:1px solid var(--line); border-radius:6px; padding:8px; font:inherit; resize:vertical; }
  .exdel { background:none; border:0; color:var(--dim); cursor:pointer; font-size:15px; padding:2px 7px; }
  .exdel:hover:not(:disabled) { background:none; filter:none; transform:none; color:var(--err); }
  .extra.accepted { flex-direction:row; align-items:center; padding:8px 10px; border-color:var(--ok); }
  .extra.accepted .exrow, .extra.accepted .extext { display:none; }
  .extra:not(.accepted) .exsum { display:none; }
  .exsum { display:flex; align-items:center; gap:8px; width:100%; }
  .exsum-label { font-size:14px; }
  .exedit { margin-left:auto; background:none; border:0; color:var(--acc); cursor:pointer;
            font-size:13px; padding:2px 6px; }
  .exedit:hover:not(:disabled) { background:none; filter:none; transform:none; text-decoration:underline; }
  .attachbar { display:flex; align-items:center; flex-wrap:wrap; gap:8px; margin-top:8px; }
  .files { display:flex; flex-wrap:wrap; gap:8px; }
  .fchip { display:inline-flex; align-items:center; gap:6px; background:var(--inbg);
           border:1px solid var(--line); border-radius:16px; padding:4px 6px 4px 10px; font-size:13px; }
  .fchip .fdel { background:none; border:0; color:var(--dim); cursor:pointer; font-size:13px;
                 padding:0 4px; line-height:1; }
  .fchip .fdel:hover:not(:disabled) { background:none; filter:none; transform:none; color:var(--err); }
</style>
<script>
  // set theme before first paint to avoid a flash of the wrong colors
  (function(){
    var t = localStorage.getItem('theme')
            || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
    if (t === 'light') document.documentElement.setAttribute('data-theme','light');
  })();
</script></head>
<body>
<header>
  <div style="flex:1"><h1>quorum</h1> <span>ask a panel of LLMs · one judge synthesizes</span></div>
  <button id="theme" class="theme" title="Toggle light / dark theme">🌙</button>
</header>
<main>
  <div class="card">
    <label>Question</label>
    <textarea id="q" placeholder="Ask anything…"></textarea>
    <div class="attachbar">
      <input type="file" id="file" multiple style="display:none"
             accept=".md,.markdown,.txt,.text,.csv,.tsv,.json,.log,.yaml,.yml,.xml,.html,.htm,.rst,.ini,.toml,.py,.js,.ts,.sh,.c,.cpp,.h,.java,.go,.rs">
      <button type="button" id="attachBtn" class="ghost">📎 Attach file</button>
      <div id="files" class="files"></div>
    </div>
    <label style="margin-top:14px">Panel</label>
    <div class="models" id="models"></div>
    <label style="margin-top:14px">External answers <span style="font-weight:400">— paste opinions from other chats (optional)</span></label>
    <div id="extras"></div>
    <button type="button" id="addExtra" class="ghost">+ Add a pasted answer</button>
    <div class="row" style="margin-top:14px">
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
    el.innerHTML=`<input type="checkbox" id="${id}" ${m.enabled?'checked':''}>
                  <span class="dot" style="background:${providerColor(m.provider)}"></span> ${esc(m.name)}
                  <span style="color:var(--dim);font-size:12px">${esc(m.provider)}·${esc(m.auth)}</span>`;
    box.appendChild(el);
  });
  const js=document.getElementById('judge');
  js.innerHTML='<option value="">— no judge —</option>'+
    d.panel.map(m=>`<option value="${m.name}">${m.name}</option>`).join('');
  if (d.judge) js.value=d.judge;
}
function selectedPanel(){ return MODELS.filter(m=>document.getElementById('m_'+m.name).checked).map(m=>m.name); }

function addExtraRow(){
  const wrap=document.createElement('div'); wrap.className='extra';
  wrap.innerHTML=`<div class="exrow">
      <input class="exname" placeholder="Source (e.g. Gemini Pro 3)">
      <button type="button" class="exok ghost" title="Accept and collapse">✓ Accept</button>
      <button type="button" class="exdel" title="Remove">✕</button>
    </div>
    <textarea class="extext" placeholder="Paste an answer from another chat to add it to the discussion…"></textarea>
    <div class="exsum"></div>`;
  const textEl=()=>wrap.querySelector('.extext');
  const accept=()=>{
    const t=textEl().value.trim();
    if(!t){ textEl().focus(); return; }            // nothing pasted yet
    const name=(wrap.querySelector('.exname').value||'').trim()||'External';
    wrap.querySelector('.exsum').innerHTML=
      `<span class="exsum-label">📋 ${esc(name)} <span class="meta">· ${t.length} chars</span></span>`+
      `<button type="button" class="exedit">edit</button>`+
      `<button type="button" class="exdel" title="Remove">✕</button>`;
    wrap.classList.add('accepted');
  };
  wrap.addEventListener('click',ev=>{                // one delegated handler for both states
    const c=ev.target.classList;
    if(c.contains('exdel')) wrap.remove();
    else if(c.contains('exok')) accept();
    else if(c.contains('exedit')){ wrap.classList.remove('accepted'); textEl().focus(); }
  });
  document.getElementById('extras').appendChild(wrap);
  textEl().focus();
}
function collectExtras(){
  return [...document.querySelectorAll('#extras .extra')].map(w=>({
    name:(w.querySelector('.exname').value||'').trim(),
    text:(w.querySelector('.extext').value||'').trim(),
  })).filter(e=>e.text);
}

let ATTACH=[];   // {name, size, text} — files read client-side as text
function fmtSize(n){ return n<1024 ? n+' B' : n<1048576 ? (n/1024).toFixed(1)+' KB' : (n/1048576).toFixed(1)+' MB'; }
function renderFiles(){
  const box=document.getElementById('files'); box.innerHTML='';
  ATTACH.forEach((a,i)=>{
    const chip=document.createElement('span'); chip.className='fchip';
    chip.innerHTML=`📄 ${esc(a.name)} <span class="meta">${fmtSize(a.size)}</span>`+
      `<button type="button" class="fdel" data-i="${i}" title="Remove">✕</button>`;
    box.appendChild(chip);
  });
}
async function onFiles(ev){
  for(const f of ev.target.files){
    try { ATTACH.push({name:f.name, size:f.size, text:await f.text()}); }
    catch(e){ document.getElementById('status').textContent='Could not read '+f.name; }
  }
  ev.target.value='';   // let the same file be re-picked later
  renderFiles();
}

function card(html, cls){ const d=document.createElement('div'); d.className='card res '+(cls||''); d.innerHTML=html; return d; }
function esc(s){ return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

// per-provider accent colors (fall back to the theme accent)
const PCOLOR={anthropic:'#d97757',openai:'#10a37f',gemini:'#4285f4',deepseek:'#4d6bfe',
              mistral:'#ff7000',groq:'#f55036',xai:'#8b5cf6'};
function providerColor(p){ return PCOLOR[(p||'').toLowerCase()] || 'var(--acc)'; }
function fmtTok(o){
  const ti=o.tokens_in||0, to=o.tokens_out||0;
  if(!ti && !to) return '';
  return ` · ${o.tokens_est?'~':''}${ti.toLocaleString()}→${to.toLocaleString()} tok`;
}

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
  const extra=collectExtras();
  const attachments=ATTACH.map(a=>({name:a.name, text:a.text}));
  const judge=document.getElementById('judge').value;
  const out=document.getElementById('out'); out.innerHTML='';
  const status=document.getElementById('status');
  if(!q && !attachments.length){ status.textContent='Enter a question or attach a file.'; return; }
  if(!panel.length && !extra.length){ status.textContent='Pick a model or add an external answer.'; return; }
  const btn=document.getElementById('ask'); btn.disabled=true;
  status.className='foot';
  status.textContent='Asking '+panel.length+' model(s)'+(extra.length?' · '+extra.length+' pasted':'')
    +(attachments.length?' · '+attachments.length+' file(s)':'')+'…';

  let judgeCard=null;
  if(judge){   // reserve the verdict slot at the top up front so nothing jumps later
    judgeCard=card(`<h3>Judge — ${esc(judge)}</h3><pre class="spin">waiting for panel<span class="dots"></span></pre>`,'judge');
    out.appendChild(judgeCard);
  }

  // skeleton placeholder + live timer for each model while it thinks
  const pending={};
  panel.forEach(name=>{
    const m=MODELS.find(x=>x.name===name); const col=providerColor(m&&m.provider);
    const el=document.createElement('div'); el.className='card res pending';
    el.style.borderLeft='3px solid '+col;
    el.innerHTML=`<div class="pendrow"><span class="dot pulse" style="background:${col}"></span>`+
      `<span class="name">${esc(name)}</span>`+
      `<span class="dots" style="color:var(--dim);font-size:13px">thinking</span>`+
      `<span class="timer meta">0.0s</span></div>`+
      `<div class="shimmer"></div><div class="shimmer s2"></div><div class="shimmer s3"></div>`;
    out.appendChild(el); pending[name]={el, t0:performance.now()};
  });
  const tick=setInterval(()=>{ const now=performance.now();
    for(const n in pending) pending[n].el.querySelector('.timer').textContent=((now-pending[n].t0)/1000).toFixed(1)+'s'; }, 100);
  let okN=0, errN=0, tokIn=0, tokOut=0;

  const resp=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:q,panel:panel,extra:extra,attachments:attachments,judge:judge||null,use_judge:!!judge})});
  const reader=resp.body.getReader(); const dec=new TextDecoder(); let buf='';
  while(true){
    const {value,done}=await reader.read(); if(done) break;
    buf+=dec.decode(value,{stream:true});
    let i;
    while((i=buf.indexOf('\n\n'))>=0){
      const line=buf.slice(0,i); buf=buf.slice(i+2);
      if(!line.startsWith('data: ')) continue;
      const o=JSON.parse(line.slice(6));
      if(o.type==='model'){
        o.error ? errN++ : okN++;
        tokIn+=o.tokens_in||0; tokOut+=o.tokens_out||0;
        const ext=!!o.external;
        const col=ext ? '#8b8f9a' : providerColor(o.provider);
        const body = o.error ? `<pre class="err">ERROR: ${esc(o.error)}</pre>`
                             : `<div class="md">${mdToHtml(o.answer)}</div>`;
        const mark = ext ? '📋' : `<span class="${o.error?'err':'ok'}">${o.error?'✗':'✓'}</span>`;
        const meta = ext ? 'pasted answer' : `${esc(o.provider)}/${esc(o.model)} · ${o.seconds}s${fmtTok(o)}`;
        const d=document.createElement('details'); d.className='card res';
        d.style.borderLeft='3px solid '+col;
        if(o.error) d.open=true;                        // keep errors visible
        d.innerHTML=`<summary><span class="dot" style="background:${col}"></span>`+
          `<span class="name">${esc(o.name)} ${mark}</span>`+
          `<span class="meta">${meta}</span></summary>${body}`;
        const ph=pending[o.name];                       // swap the skeleton in place
        if(ph){ ph.el.replaceWith(d); delete pending[o.name]; } else out.appendChild(d);
      } else if(o.type==='judge_start'){
        if(!judgeCard){   // fallback: judge wasn't pre-reserved
          judgeCard=card(`<h3>Judge — ${esc(o.name)}</h3><pre class="spin"></pre>`,'judge');
          out.insertBefore(judgeCard, out.firstChild);
        }
        const p=judgeCard.querySelector('pre'); if(p) p.innerHTML='synthesizing<span class="dots"></span>';
      } else if(o.type==='judge'){
        if(judgeCard){
          judgeCard.querySelector('pre').outerHTML=`<div class="md">${mdToHtml(o.text)}</div>`;
          const b=fmtTok(o); const h=judgeCard.querySelector('h3');
          if(b && h) h.innerHTML+=`<span class="meta">${b}</span>`;
        }
        tokIn+=o.tokens_in||0; tokOut+=o.tokens_out||0;
      } else if(o.type==='error'){ status.className='foot'; status.innerHTML=`<span class="banner err">✗ ${esc(o.message)}</span>`; }
      else if(o.type==='done'){
        clearInterval(tick);
        const total=okN+errN;
        let cls='ok', icon='✓', txt=`all ${total} answered`;
        if(okN===0){ cls='err'; icon='✗'; txt=`all ${total} failed`; }
        else if(errN){ cls='warn'; icon='⚠'; txt=`${okN}/${total} answered · ${errN} failed`; }
        const tk=(tokIn||tokOut) ? ` <span class="meta">· ${(tokIn+tokOut).toLocaleString()} tok `
          +`(${tokIn.toLocaleString()} in · ${tokOut.toLocaleString()} out)</span>` : '';
        status.innerHTML=`<span class="banner ${cls}">${icon} ${txt}</span>`+tk;
        const p=judgeCard && judgeCard.querySelector('pre.spin');   // judge never ran (panel empty/all failed)
        if(p) p.outerHTML=`<pre class="meta">No verdict — the panel produced no answers to judge.</pre>`;
      }
    }
  }
  clearInterval(tick);   // safety: stop timers if the stream ends without a 'done'
  btn.disabled=false;
}
document.getElementById('ask').addEventListener('click',ask);
document.getElementById('addExtra').addEventListener('click',addExtraRow);
document.getElementById('attachBtn').addEventListener('click',()=>document.getElementById('file').click());
document.getElementById('file').addEventListener('change',onFiles);
document.getElementById('files').addEventListener('click',ev=>{
  if(ev.target.classList.contains('fdel')){ ATTACH.splice(+ev.target.dataset.i,1); renderFiles(); }
});

const themeBtn=document.getElementById('theme');
function curTheme(){ return document.documentElement.getAttribute('data-theme')==='light' ? 'light' : 'dark'; }
function applyTheme(t){
  if(t==='light') document.documentElement.setAttribute('data-theme','light');
  else document.documentElement.removeAttribute('data-theme');
  localStorage.setItem('theme',t);
  themeBtn.textContent = t==='light' ? '☀️' : '🌙';
}
applyTheme(curTheme());   // sync the icon with the theme set before paint
themeBtn.addEventListener('click',()=>applyTheme(curTheme()==='light'?'dark':'light'));

loadModels();
</script>
</body></html>"""
