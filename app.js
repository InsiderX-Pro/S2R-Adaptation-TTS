'use strict';
const data = window.DEMO_DATA;
const $ = (id) => document.getElementById(id);
const languageNames = {my:'Burmese', lo:'Lao', en:'English', zh:'Chinese'};
const labels = {S:'S-stage', R:'R-stage', SR:'S → R · cubic', RS:'R → S', OMNI_BASE:'Base model'};
let currentGroup = 'my';
const escapeHTML = (value) => String(value).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const groups = {
  my: {title:'Burmese · zero-shot voice cloning', description:'One selected text, one reference voice. Compare OmniVoice Base and FireRedTTS3 at each adaptation stage, including our cubic S→R method.'},
  lo: {title:'Lao · adaptation-stage comparison', description:'One selected FLEURS text with a shared Lao reference. Compare S-stage, R-stage, reverse-order R→S, and our cubic S→R method.'}
};
function selectedModels(c) {
  return [...c.models].sort((a,b) => ['OMNI_BASE','S','SR','R','RS'].indexOf(a.system) - ['OMNI_BASE','S','SR','R','RS'].indexOf(b.system));
}
function condition(c, m) {
  if (m.system === 'OMNI_BASE') return 'Pretrained model';
  return {S:'Synthetic speech adaptation',R:'Real speech adaptation',SR:'Synthetic → reliability-weighted real speech',RS:'Real → synthetic speech adaptation'}[m.system];
}
function audioElement(c, m, name) {
  return `<audio controls preload="metadata" src="${escapeHTML(m.src)}" aria-label="${escapeHTML(languageNames[c.language]+' '+c.case+', '+name)}"></audio><span class="audio-error" role="status" hidden>Audio could not load. <a href="${escapeHTML(m.src)}" download>Download the WAV file</a>.</span>`;
}
function renderAudio() {
  document.querySelectorAll('audio').forEach(a => a.pause());
  const c = data.cases.find(c => c.language === currentGroup);
  const models = selectedModels(c);
  $('group-title').textContent = groups[currentGroup].title;
  $('group-description').textContent = groups[currentGroup].description;
  $('audio-count').innerHTML = `<strong>${models.length + 1} <small>audio clips</small></strong><span>${models.length} models · 1 reference</span>`;
  const cards = models.map(m => {
    const name = (m.system==='OMNI_BASE'?'OmniVoice':'FireRedTTS3')+' '+labels[m.system];
    return `<div class="model-audio ${m.system==='SR'?'progressive':''}"><div class="model-heading"><h4>${labels[m.system]}</h4>${m.system==='SR'?'<span class="model-tag">OURS</span>':''}</div><span class="backbone">${m.system==='OMNI_BASE'?'OmniVoice':'FireRedTTS3'}</span><p class="condition">${condition(c,m)}</p>${audioElement(c,m,name)}</div>`;
  }).join('');
  $('sample-card').innerHTML = `<div class="sample-top"><span class="language-badge">${languageNames[c.language]}</span><span class="small-label">SELECTED EXAMPLE · ${c.language==='my'?'COMMON400 / CLONE300 SUBSET':'FLEURS404'}</span></div><p class="sample-text" lang="${c.language}">${escapeHTML(c.text)}</p><div class="reference-row"><div><h4>Reference voice <span>${languageNames[c.referenceLanguage]||c.referenceLanguage}</span></h4><p>A separate prompt utterance; its text can differ from the target.</p><details><summary>Reference transcript</summary><p lang="${c.referenceLanguage}">${escapeHTML(c.referenceText)}</p></details></div><div>${audioElement(c,c.reference,'Reference voice')}</div></div><div class="audio-grid">${cards}</div><details class="source-detail"><summary>Sample details</summary><p>Source ID: <code>${escapeHTML(c.sourceId)}</code><br>One example per language, selected by the highest cubic overall listening score in the available collection. <a href="assets/demo-data.json" download>Download metadata</a>.</p></details>`;
  $('audio-panel').setAttribute('aria-labelledby','tab-'+currentGroup);
}
function changeGroup(group) {
  currentGroup=group;
  document.querySelectorAll('[data-group]').forEach(b=>{const active=b.dataset.group===group; b.setAttribute('aria-selected',active); b.tabIndex=active?0:-1;});
  renderAudio();
}
document.addEventListener('click', e=>{
  const tab=e.target.closest('[data-group]'); if(tab) changeGroup(tab.dataset.group);
});
document.addEventListener('play', e=>{if(e.target.tagName==='AUDIO'){document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause();});}},true);
document.addEventListener('error', e=>{if(e.target.tagName==='AUDIO')e.target.nextElementSibling.hidden=false;},true);
document.querySelectorAll('[role=tablist]').forEach(list=>list.addEventListener('keydown', e=>{
  if(!['ArrowRight','ArrowLeft','Home','End'].includes(e.key))return;
  const tabs=[...list.querySelectorAll('[role=tab]')];const index=tabs.indexOf(document.activeElement);if(index<0)return;e.preventDefault();
  const next=e.key==='Home'?0:e.key==='End'?tabs.length-1:(index+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;
  tabs[next].focus();tabs[next].click();
}));
const resultData = [
  [['17.427','0.45126','58.36','3.15'],['54.093','0.73139','56.41','1.86'],['23.274','0.69424','72.89','—'],['20.914','0.69962','74.24','—'],['14.714','0.36357','50.98','2.72'],['16.902','0.69973','75.97','4.12']],
  [['13.488','0.61081','71.61','3.38'],['37.062','0.75372','68.60','2.12'],['14.987','0.69931','76.74','—'],['14.598','0.69677','76.74','—'],['10.643','0.59592','71.50','3.27'],['13.965','0.69684','77.00','3.97']],
  [['8.21','0.6864','78.54','4.23'],['10.74','0.7328','80.48','2.35'],['10.21','0.7142','79.56','—'],['8.03','0.7094','80.10','—'],['7.36','0.6948','79.41','4.32'],['6.85','0.7098','80.57','4.50']]
];
const strategies=['S · synthetic only','R · real only','S → R · uniform 1.0','S → R · uniform 0.5','R → S · reverse order','S → R · cubic'];
function renderResults(index) {
  const rows=resultData[index];
  const best=rows[0].map((_,col)=>{const values=rows.map(r=>Number(r[col])).filter(Number.isFinite);return col===0?Math.min(...values):Math.max(...values);});
  $('results-body').innerHTML=rows.map((row,i)=>`<tr class="${i===5?'ours':''}"><th scope="row">${strategies[i]}${i===5?'<span class="table-tag">PROPOSED</span>':''}</th>${row.map((value,col)=>`<td>${Number(value)===best[col]?'<strong>'+value+'</strong>':value}</td>`).join('')}</tr>`).join('');
  $('table-caption').textContent='Table 1 · '+$('result-tab-'+index).textContent;
  $('result-protocol').textContent=index===1?'Lao: FLEURS404 · 260 inputs · one reference · Gemini CER scoring. CER seed SD: 0.031–0.119 points. Base MOS: 2.54.':`Burmese: Common400 CER · Clone300 SIM-O · Omnilingual ASR Unlimited 7B v2 scoring. Base MOS: ${index===0?'2.17':'4.40'}.${index===2?' The H margin of cubic over R is 0.08/100.':''}`;
  $('results-panel').setAttribute('aria-labelledby','result-tab-'+index);
  document.querySelectorAll('[data-result]').forEach(b=>{const active=Number(b.dataset.result)===index;b.setAttribute('aria-selected',active);b.tabIndex=active?0:-1;});
}
document.querySelectorAll('[data-result]').forEach(b=>b.addEventListener('click',()=>renderResults(Number(b.dataset.result))));
$('copy-citation').addEventListener('click',async()=>{
  const text=$('bibtex').textContent;
  try {if(!navigator.clipboard)throw new Error('Clipboard unavailable');await navigator.clipboard.writeText(text);$('copy-citation').textContent='Copied ✓';$('copy-status').textContent='Citation copied to clipboard.';}
  catch {const range=document.createRange();range.selectNodeContents($('bibtex'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);$('copy-citation').textContent='Text selected';$('copy-status').textContent='Citation selected. Press Ctrl+C or Command+C to copy.';}
});
renderAudio();renderResults(0);
