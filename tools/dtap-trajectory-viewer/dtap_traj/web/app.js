const state = {
  facets: null, episodes: [], total: 0, selected: null, tab: 'policy',
  page: 0, limit: 100, filters: {run_name:'', domain:'', threat_model:'', status:'', attack_success:'', q:''},
  cache: new Map(), loading: false,
};
const $ = (s, root=document) => root.querySelector(s);
const esc = (v='') => String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}
function qsFilters() {
  const p = new URLSearchParams({limit: state.limit, offset: state.page * state.limit});
  Object.entries(state.filters).forEach(([k,v]) => { if (v !== '' && v !== null) p.set(k,v); });
  return p.toString();
}
async function loadFacets(){ state.facets = await api('/api/facets'); }
async function loadEpisodes(){
  state.loading = true; render();
  const data = await api(`/api/episodes?${qsFilters()}`);
  state.episodes = data.items; state.total = data.total; state.loading = false;
  if (!state.selected || !state.episodes.some(x => x.episode_id === state.selected.episode_id)) state.selected = state.episodes[0] || null;
  render(); if (state.selected) loadDetail();
}
async function loadDetail(){
  const ep = state.selected; if (!ep) return;
  const key = ep.episode_id;
  if (state.cache.has(key)) { renderDetail(); return; }
  $('#viewer').innerHTML = '<div class="loading">Loading trajectory…</div>';
  const [traj, config, judges] = await Promise.all([
    api(`/api/episodes/${encodeURIComponent(key)}/trajectory?view=combined`),
    api(`/api/episodes/${encodeURIComponent(key)}/config`),
    api(`/api/episodes/${encodeURIComponent(key)}/judges`),
  ]);
  state.cache.set(key,{...traj, config: config.comparison, judges: judges.judges}); renderDetail();
}
function options(items, selected, all='All') {
  return `<option value="">${all}</option>${(items||[]).map(x=>`<option value="${esc(x.value)}" ${String(x.value)===String(selected)?'selected':''}>${esc(x.value)} · ${x.count}</option>`).join('')}`;
}
function sidebar(){
  const f=state.facets||{}; const s=state.filters;
  return `<aside class="sidebar">
    <div class="eyebrow">Run</div><div class="filter"><select data-filter="run_name">${options(f.runs,s.run_name)}</select></div>
    <div class="eyebrow">Trajectory</div>
    <div class="filter"><label>Domain</label><select data-filter="domain">${options(f.domains,s.domain)}</select></div>
    <div class="filter"><label>Threat model</label><select data-filter="threat_model">${options(f.threat_models,s.threat_model)}</select></div>
    <div class="filter"><label>Status</label><select data-filter="status">${options(f.statuses,s.status)}</select></div>
    <div class="filter"><label>Attack success</label><div class="toggle">
      <button data-attack="" class="${s.attack_success===''?'active':''}">All</button>
      <button data-attack="true" class="${s.attack_success==='true'?'active':''}">Yes</button>
      <button data-attack="false" class="${s.attack_success==='false'?'active':''}">No</button>
    </div></div>
    <button class="reset" id="resetFilters">Reset filters</button>
  </aside>`;
}
function episodeRow(ep){
  const active=state.selected?.episode_id===ep.episode_id;
  return `<div class="episode ${active?'active':''}" data-episode="${esc(ep.episode_id)}">
    <div class="ep-top"><span class="domain">${esc(ep.domain||'unknown')}</span><span class="threat">${esc(ep.threat_model||'—')}</span></div>
    <div class="ep-id">${esc(ep.episode_id)}</div>
    <div class="ep-meta"><span class="${ep.attack_success?'bad':'ok'}"><i class="dot"></i>${ep.attack_success?'attack success':'contained'}</span><span>${ep.policy_events??'—'} policy</span><span>${ep.victim_events??'—'} victim</span></div>
  </div>`;
}
function listPane(){
  const start=state.total? state.page*state.limit+1:0, end=Math.min((state.page+1)*state.limit,state.total);
  return `<section class="listpane"><div class="listhead"><div class="listhead-row"><h2>Episodes</h2><span>${start}–${end} / ${state.total}</span></div></div>
    <div class="episodes">${state.loading?'<div class="loading">Index query…</div>':state.episodes.map(episodeRow).join('')}</div>
    <div class="pager"><button id="prev" ${state.page===0?'disabled':''}>← Previous</button><button id="next" ${end>=state.total?'disabled':''}>Next →</button></div></section>`;
}
function detailShell(){
  const ep=state.selected;
  if(!ep) return `<main class="detail"><div class="empty"><div><strong>No trajectory selected</strong>Adjust filters or index a run.</div></div></main>`;
  return `<main class="detail"><div class="detailhead">
      <div class="detail-title"><h1>${esc(ep.domain||'Episode')}</h1><span class="badge">${esc(ep.threat_model||'unknown')}</span><span class="badge">${esc(ep.status||ep.episode_status||'unknown')}</span></div>
      <div class="metrics"><span class="metric">episode <b>${esc(ep.episode_id)}</b></span><span class="metric">policy <b>${ep.policy_events??'—'}</b></span><span class="metric">victim <b>${ep.victim_events??'—'}</b></span><span class="metric">placements <b>${ep.placements_verified??0}</b></span><span class="metric">attack <b class="${ep.attack_success?'bad':'ok'}">${ep.attack_success?'success':'contained'}</b></span></div>
    </div><div class="tabs">${['policy','victim','combined','judges','config'].map(t=>`<button data-tab="${t}" class="${state.tab===t?'active':''}">${t==='config'?'Config Diff':t==='judges'?'DTAP Judges':t[0].toUpperCase()+t.slice(1)}</button>`).join('')}</div><div class="viewer" id="viewer"></div></main>`;
}
function eventCard(e,i){
  const kind=e.kind||'event'; const tool=[e.server,e.tool].filter(Boolean).join(':');
  let body=e.text;
  if(kind==='tool_call') body=JSON.stringify(e.args||{},null,2);
  if(!body) body='';
  return `<article class="event"><div class="event-head"><span class="kind ${esc(kind)}">${esc(kind.replace('_',' '))}</span>${tool?`<span class="toolname">${esc(tool)}</span>`:''}<span style="margin-left:auto;color:#657184;font-size:9px">#${i+1}</span></div><div class="event-body">${esc(body)}</div></article>`;
}
function timeline(events){ return `<div class="timeline">${(events||[]).map(eventCard).join('')||'<div class="empty">No events</div>'}</div>`; }
function diffHtml(diff){
  if(!diff) return '<div class="empty">No original/submitted config pair found.</div>';
  return `<pre class="diff">${esc(diff).split('\n').map(line=>{
    const c=line.startsWith('+++')||line.startsWith('---')||line.startsWith('@@')?'hdr':line.startsWith('+')?'add':line.startsWith('-')?'del':'';
    return `<span class="${c}">${line}</span>`;
  }).join('\n')}</pre>`;
}
function judgeOutcome(component){
  if(component.success === null || component.success === undefined) return ['unknown','neutral'];
  if(component.name === 'attack') return [component.success?'attack succeeded':'attack contained',component.success?'bad':'ok'];
  return [component.success?'task succeeded':'task failed',component.success?'ok':'bad'];
}
function judgeCard(component){
  const [outcome,outcomeClass]=judgeOutcome(component);
  const source=component.source==='llm_as_judge'?'LLM-as-judge':'Deterministic';
  const metadata=component.metadata&&Object.keys(component.metadata).length
    ? `<details><summary>Judge metadata</summary><pre>${esc(JSON.stringify(component.metadata,null,2))}</pre></details>`:'';
  return `<article class="judge-card"><div class="judge-head"><h3>${esc(component.name)} judge</h3><span class="judge-source ${esc(component.source)}">${source}</span><span class="judge-outcome ${outcomeClass}">${esc(outcome)}</span></div>${component.message?`<div class="judge-message">${esc(component.message)}</div>`:''}${metadata}</article>`;
}
function judgesHtml(judges){
  if(!judges||!judges.available) return '<div class="empty">No DTAP judge artifacts found.</div>';
  const cards=(judges.components||[]).map(judgeCard).join('');
  const firewall=judges.reward_firewall&&Object.keys(judges.reward_firewall).length
    ? `<article class="judge-card firewall"><div class="judge-head"><h3>Reward firewall verdict</h3><span class="judge-source trusted">Trusted projection</span></div><pre>${esc(JSON.stringify(judges.reward_firewall,null,2))}</pre></article>`:'';
  const error=judges.error?`<article class="judge-card error"><div class="judge-head"><h3>Judge error</h3></div><div class="judge-message">${esc(judges.error)}</div></article>`:'';
  return `<div class="judges">${cards}${firewall}${error}</div>`;
}
function renderDetail(){
  const viewer=$('#viewer'); if(!viewer||!state.selected) return;
  const data=state.cache.get(state.selected.episode_id);
  if(!data){ viewer.innerHTML='<div class="loading">Loading trajectory…</div>'; return; }
  if(state.tab==='policy') viewer.innerHTML=timeline(data.policy);
  else if(state.tab==='victim') viewer.innerHTML=timeline(data.victim);
  else if(state.tab==='combined') viewer.innerHTML=`<div class="lanes"><section><div class="lane-title">Policy trajectory · ${(data.policy||[]).length}</div>${timeline(data.policy)}</section><section><div class="lane-title">Victim trajectory · ${(data.victim||[]).length}</div>${timeline(data.victim)}</section></div>`;
  else if(state.tab==='judges') viewer.innerHTML=judgesHtml(data.judges);
  else viewer.innerHTML=diffHtml(data.config?.diff);
}
function bind(){
  $('[data-filter="run_name"]')?.addEventListener('change',filterChange);
  $('[data-filter="domain"]')?.addEventListener('change',filterChange);
  $('[data-filter="threat_model"]')?.addEventListener('change',filterChange);
  $('[data-filter="status"]')?.addEventListener('change',filterChange);
  document.querySelectorAll('[data-attack]').forEach(b=>b.addEventListener('click',()=>{state.filters.attack_success=b.dataset.attack; state.page=0; loadEpisodes();}));
  document.querySelectorAll('[data-episode]').forEach(el=>el.addEventListener('click',()=>{state.selected=state.episodes.find(x=>x.episode_id===el.dataset.episode); render(); loadDetail();}));
  document.querySelectorAll('[data-tab]').forEach(b=>b.addEventListener('click',()=>{state.tab=b.dataset.tab; document.querySelectorAll('[data-tab]').forEach(x=>x.classList.toggle('active',x===b)); renderDetail();}));
  $('#prev')?.addEventListener('click',()=>{if(state.page>0){state.page--;loadEpisodes();}});
  $('#next')?.addEventListener('click',()=>{if((state.page+1)*state.limit<state.total){state.page++;loadEpisodes();}});
  $('#resetFilters')?.addEventListener('click',()=>{state.filters={run_name:'',domain:'',threat_model:'',status:'',attack_success:'',q:''};state.page=0;$('#globalSearch').value='';loadEpisodes();});
  $('#globalSearch')?.addEventListener('input',debounce(e=>{state.filters.q=e.target.value.trim();state.page=0;loadEpisodes();},250));
}
function filterChange(e){state.filters[e.target.dataset.filter]=e.target.value;state.page=0;loadEpisodes();}
function debounce(fn,ms){let t;return(...a)=>{clearTimeout(t);t=setTimeout(()=>fn(...a),ms)}}
function render(){
  const f=state.facets||{};
  $('#app').innerHTML=`<div class="shell"><header class="topbar"><div class="brand"><div class="logo"></div><div><strong>DTAP Explorer</strong><small>trajectory observability</small></div></div><div class="search"><input id="globalSearch" placeholder="Search episode, risk category, artifact path…" value="${esc(state.filters.q)}"></div><div class="statline"><span><b>${f.total??0}</b> episodes</span><span><b>${f.domains?.length??0}</b> domains</span><span><b>${f.attack_successes??0}</b> attacks</span></div></header><div class="workspace">${sidebar()}${listPane()}${detailShell()}</div></div>`;
  bind(); renderDetail();
}
(async()=>{ await loadFacets(); await loadEpisodes(); })().catch(err=>{ console.error(err); $('#app').innerHTML=`<div class="empty"><div><strong>Explorer failed to load</strong>${esc(err.message)}</div></div>`; });
