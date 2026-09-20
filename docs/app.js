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
// Paper Table 1: mean ± sample SD; MOS mean [approximate 95% CI].
const resultData = [
  [
    ['142.52','0.7244','0.00','2.17 [1.97, 2.38]','103.56'],
    ['17.43 ± 0.45','0.4513 ± 0.0041','58.36','3.15 [2.97, 3.34]','20.45'],
    ['54.09 ± 0.68','0.7314 ± 0.0005','56.41','1.87 [1.69, 2.06]','68.64'],
    ['14.71 ± 0.62','0.3636 ± 0.0038','50.98','2.72 [2.52, 2.92]','17.93'],
    ['23.27 ± 0.15','0.6942 ± 0.0014','72.89','3.78 [3.55, 4.00]','26.82'],
    ['20.91 ± 0.37','0.6996 ± 0.0005','74.24','3.91 [3.71, 4.11]','25.06'],
    ['16.90 ± 0.26','0.6997 ± 0.0016','75.97','4.12 [3.93, 4.30]','19.34']
  ],
  [
    ['100.45','0.7343','0.00','2.55 [2.34, 2.76]','100.45'],
    ['13.49 ± 0.26','0.6108 ± 0.0006','71.61','3.39 [3.22, 3.56]','18.03'],
    ['37.06 ± 0.36','0.7537 ± 0.0019','68.60','2.12 [1.93, 2.32]','44.94'],
    ['10.64 ± 0.09','0.5959 ± 0.0014','71.50','3.27 [3.07, 3.47]','17.13'],
    ['14.99 ± 0.25','0.6993 ± 0.0010','76.74','3.78 [3.61, 3.94]','23.85'],
    ['14.60 ± 0.14','0.6968 ± 0.0026','76.74','3.84 [3.65, 4.02]','23.33'],
    ['13.97 ± 0.36','0.6968 ± 0.0045','77.00','3.97 [3.77, 4.17]','18.53']
  ],
  [
    ['10.35','0.7278','80.34','4.40 [4.23, 4.56]','12.62'],
    ['8.21 ± 0.20','0.6864 ± 0.0010','78.54','4.23 [4.05, 4.40]','11.63'],
    ['10.74 ± 0.41','0.7328 ± 0.0013','80.48','2.35 [2.15, 2.56]','12.81'],
    ['7.36 ± 0.12','0.6948 ± 0.0035','79.41','4.32 [4.12, 4.50]','11.65'],
    ['10.21 ± 0.22','0.7142 ± 0.0017','79.56','4.35 [4.17, 4.51]','11.69'],
    ['8.03 ± 0.33','0.7094 ± 0.0009','80.10','4.42 [4.24, 4.58]','11.41'],
    ['6.85 ± 0.08','0.7098 ± 0.0026','80.57','4.51 [4.35, 4.66]','9.83']
  ]
];
const strategies=['Base · pretrained','S · synthetic only','R · real only','R → S · reverse order','S → R · uniform 1.0','S → R · uniform 0.5','S → R · cubic'];
function renderResults(index) {
  const rows=resultData[index];
  const best=rows[0].map((_,col)=>{const values=rows.map(r=>parseFloat(r[col]));return col===0||col===4?Math.min(...values):Math.max(...values);});
  $('results-body').innerHTML=rows.map((row,i)=>`<tr class="${i===6?'ours':''}"><th scope="row">${strategies[i]}${i===6?'<span class="table-tag">PROPOSED</span>':''}</th>${row.map((value,col)=>`<td>${parseFloat(value)===best[col]?'<strong>'+value+'</strong>':value}</td>`).join('')}</tr>`).join('');
  $('table-caption').textContent='Table 1 · '+$('result-tab-'+index).textContent;
  $('result-protocol').textContent=index===1?'Lao: FLEURS404 · 260 inputs · one reference · primary evaluation with Gemini; independent evaluation with XLS-R Lao.':'Burmese: Common400 CER · Clone300 SIM-O · primary evaluation with Omnilingual ASR Unlimited 7B v2; independent evaluation with Dolphin-small.'+(index===2?' Cubic exceeds R in H by only 0.08/100.':'');
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
