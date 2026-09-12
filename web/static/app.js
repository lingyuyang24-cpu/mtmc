'use strict';
const $ = id => document.getElementById(id);
const terminal = new Set(['completed', 'failed', 'cancelled']);
const statusText = {starting:'加载模型', running:'追踪中', stopping:'停止中', completed:'已完成', failed:'失败', cancelled:'已停止'};
const cameraText = {waiting:'等待处理', loading:'加载独立模型', running:'追踪中', reconnecting:'无新帧 · 重连中', fusing:'等待融合', rendering:'输出全局 ID', completed:'已完成'};
const state = {mode:'online', streams:[''], files:[], selected:null, jobs:[], busy:false, online:false, refreshing:false, uploads:new Map(), cards:new Map(), limit:0};
function node(tag, className, text) {const n=document.createElement(tag); if(className)n.className=className; if(text!==undefined)n.textContent=text; return n;}
function errorText(data) {return Array.isArray(data.detail) ? data.detail.map(e=>`${e.loc?.slice(1).join('.')}：${e.msg}`).join('；') : data.detail || '请求失败，请稍后重试。';}
function showError(id, message='') {$(id).textContent=message; $(id).hidden=!message;}
async function api(path, options={}) {
  const response=await fetch(path, {...options, headers:{'X-MTMC-Client':'web', ...(options.body ? {'Content-Type':'application/json'} : {}), ...options.headers}, signal:AbortSignal.timeout(15000)});
  let data; try {data=await response.json();} catch {throw new Error('后端响应异常，请确认服务已启动。');}
  if(!response.ok)throw new Error(errorText(data));
  return data;
}
function formatSize(bytes) {return bytes>=1024**3 ? `${(bytes/1024**3).toFixed(2)} GB` : `${(bytes/1024**2).toFixed(1)} MB`;}
function activeJob() {return state.jobs.find(job=>!terminal.has(job.status));}
function setBusy(value) {state.busy=value; updateControls();}
function updateControls() {
  document.querySelectorAll('#job-form button, #job-form input, #job-form select, [role=tab]').forEach(el=>el.disabled=state.busy);
  $('start').disabled=state.busy || !state.online || !!activeJob();
  $('min-frames').disabled=state.busy || state.mode==='online';
  $('offline-workers').disabled=state.busy || state.mode==='online';
  $('offline-concurrency').hidden=state.mode==='online';
  $('start').textContent=state.busy ? '正在提交…' : activeJob() ? '请等待或停止当前任务' : state.mode==='online' ? '开始在线追踪 →' : '开始离线追踪 →';
  $('stop').disabled=state.busy || state.selected?.status==='stopping';
}
function setMode(mode) {
  if(state.busy)return;
  state.mode=mode;
  for(const value of ['online','offline']) {
    $(value+'-tab').setAttribute('aria-selected',String(value===mode));
    $(value+'-tab').tabIndex=value===mode ? 0 : -1;
    $(value+'-panel').hidden=value!==mode;
  }
  $('source-title').textContent=mode==='online'?'接入视频流':'导入视频文件';
  showError('form-error'); updateControls();
}
function renderStreams() {
  $('streams').replaceChildren();
  state.streams.forEach((value,index)=>{
    const row=node('div','stream-row'), title=node('div','stream-title'), label=node('label','',`摄像头 ${String(index+1).padStart(2,'0')}`);
    const id=`stream-${index}`; label.htmlFor=id; title.append(label);
    if(state.streams.length>1){const remove=node('button','remove','移除');remove.type='button';remove.setAttribute('aria-label',`移除摄像头 ${index+1}`);remove.onclick=()=>{state.streams.splice(index,1);renderStreams();updateControls();};title.append(remove);}
    const input=node('input');Object.assign(input,{id,type:'text',value,placeholder:'rtsp://192.168.1.10/live',autocomplete:'off',spellcheck:false});
    input.oninput=()=>state.streams[index]=input.value;row.append(title,input);$('streams').append(row);
  });
}
function addFiles(files) {
  if(state.busy)return;
  const errors=[];
  for(const file of files){
    if(!/\.(mp4|avi|mov|mkv|webm|m4v)$/i.test(file.name)){errors.push(`${file.name}：不支持的格式`);continue;}
    if(file.size===0){errors.push(`${file.name}：文件为空`);continue;}
    if(state.limit&&file.size>state.limit){errors.push(`${file.name}：超过单文件 ${formatSize(state.limit)} 限制`);continue;}
    if(!state.files.some(f=>f.name===file.name&&f.size===file.size&&f.lastModified===file.lastModified))state.files.push(file);
  }
  showError('form-error',errors.join('；'));renderFiles();
}
function renderFiles() {
  $('file-list').replaceChildren();
  state.files.forEach((file,index)=>{
    const row=node('div','file-item'), label=node('span','',file.name);label.append(node('small','',`摄像头 ${index+1} · ${formatSize(file.size)}${state.uploads.has(file)?' · 已上传':''}`));
    const remove=node('button','remove','移除');remove.type='button';remove.setAttribute('aria-label',`移除 ${file.name}`);remove.onclick=()=>{state.files.splice(index,1);renderFiles();};row.append(label,remove);$('file-list').append(row);
  });
}
function upload(file, index, total) {
  return new Promise((resolve,reject)=>{
    const xhr=new XMLHttpRequest();xhr.open('POST',`/api/uploads?filename=${encodeURIComponent(file.name)}`);xhr.setRequestHeader('X-MTMC-Client','web');xhr.setRequestHeader('Content-Type','application/octet-stream');
    xhr.upload.onprogress=e=>{const percent=e.lengthComputable?Math.round(e.loaded/e.total*100):0;$('start').textContent=`上传 ${index}/${total} · ${percent}%`;};
    xhr.onload=()=>{let data;try{data=JSON.parse(xhr.responseText);}catch{reject(new Error('上传失败：后端响应异常。'));return;}if(xhr.status>=200&&xhr.status<300)resolve(data);else reject(new Error(errorText(data)));};
    xhr.onerror=()=>reject(new Error('上传连接中断，请检查后端服务后重试。'));
    xhr.onabort=()=>reject(new Error('上传已取消。'));
    xhr.send(file);
  });
}
function selectedOptions() {return {confidence:Number($('confidence').value),reid_threshold:Number($('threshold').value),batch_size:Number($('batch-size').value),min_reid_frames:Number($('min-frames').value),offline_workers:Number($('offline-workers').value)};}
function validateStreams(values) {
  if(!values.length)throw new Error('请至少添加一路流地址。');
  return values.map((raw,i)=>{
    const url=raw.trim();let parsed;try{parsed=new URL(url);}catch{throw new Error(`摄像头 ${i+1} 的流地址无效。`);}
    if(!['rtsp:','http:','https:'].includes(parsed.protocol)||!parsed.hostname||/\s/.test(url))throw new Error(`摄像头 ${i+1} 需要有效的 RTSP / HTTP / HTTPS 地址。`);
    return {name:`摄像头 ${String(i+1).padStart(2,'0')}`,url};
  });
}
async function submit(event) {
  event.preventDefault();showError('form-error');
  if(state.busy||activeJob())return;
  try{
    const mode=state.mode, options=selectedOptions();
    let body;
    if(mode==='online')body={streams:validateStreams(state.streams),options};
    else if(!state.files.length)throw new Error('请至少选择一个视频文件。');
    setBusy(true);
    if(mode==='offline'){
      const ids=[];
      for(let i=0;i<state.files.length;i++){
        const file=state.files[i];let saved=state.uploads.get(file);
        if(!saved){saved=await upload(file,i+1,state.files.length);state.uploads.set(file,saved);}
        ids.push(saved.id);
      }
      body={upload_ids:ids,options};renderFiles();
    }
    const job=await api(`/api/jobs/${mode}`,{method:'POST',body:JSON.stringify(body)});
    state.jobs=[job,...state.jobs];selectJob(job);
  }catch(error){showError('form-error',error.message);}finally{setBusy(false);}
}
function clearCards() {
  for(const card of state.cards.values()) {
    card.livePlayer?.stop();
    card.exactPlayer?.stop();
    card.video.pause();card.video.removeAttribute('src');card.video.load();
    card.image.removeAttribute('src');
  }
  state.cards.clear();$('camera-grid').replaceChildren();
}
function selectJob(job) {if(state.selected?.id!==job.id)clearCards();state.selected=job;renderJob(job);renderHistory();}
function renderJob(job) {
  state.selected=job;
  $('task-label').textContent=`${job.mode==='online'?'实时在线':'离线视频'} · ${new Date(job.created_at*1000).toLocaleString('zh-CN',{hour12:false})}`;
  $('task-status').textContent=statusText[job.status]||job.status;$('task-status').className=`badge ${job.status}`;
  $('stop').hidden=terminal.has(job.status);
  $('metric-cameras').textContent=job.cameras.length;$('metric-frames').textContent=job.processed_frames.toLocaleString();
  $('metric-identities').textContent=job.identity_count;$('metric-fps').textContent=`${job.fps.toFixed(1)}`;
  $('metric-fps').title='各摄像头合计处理帧数 / 处理耗时（帧/秒）';
  $('progress-area').hidden=false;$('progress-message').textContent=job.message;
  $('progress-percent').textContent=job.progress===null?'':`${Math.floor(job.progress)}%`;
  if(job.progress===null)$('progress').removeAttribute('value');else $('progress').value=job.progress;
  $('progress').hidden=terminal.has(job.status)&&job.progress===null;
  showError('task-error',job.error||'');$('empty-view').hidden=true;
  for(const camera of job.cameras){
    let card=state.cards.get(camera.id);
    if(!card){
      const root=node('article','camera-card'),heading=node('div','camera-head'),label=node('span','',camera.name),status=node('small');
      heading.append(label,status);const waiting=node('div','camera-wait','等待模型与视频源…'),image=node('img');image.hidden=true;image.alt=`${camera.name} 追踪结果`;
      const replay=node('button','replay-button','播放器回放');replay.hidden=true;
      const exact=node('button','replay-button','逐帧回放（不主动跳帧）');exact.hidden=true;
      exact.title='逐帧确认绘制，负载高时放慢速度；从头回放，暂不支持拖动进度';
      const canvas=node('canvas','live-canvas');canvas.hidden=true;
      const video=node('video','result-video');video.controls=true;video.playsInline=true;video.preload='metadata';video.hidden=true;
      const liveLabel=node('p','media-caption','连续原画正在连接…');liveLabel.hidden=true;
      const annotationLabel=node('p','media-caption');annotationLabel.hidden=true;
      const playbackLabel=node('p','media-caption');playbackLabel.hidden=true;
      card={root,status,waiting,image,replay,exact,canvas,video,liveLabel,annotationLabel,playbackLabel,version:-1,playing:false,exactPlaying:false};
      const reference=card;
      video.onerror=()=>{playbackLabel.hidden=false;playbackLabel.textContent='视频解码失败，请下载结果或检查浏览器 H.264 支持。';};
      video.onwaiting=()=>{if(reference.exactPlaying)return;playbackLabel.hidden=false;playbackLabel.textContent='正在缓冲视频…';};
      video.ontimeupdate=()=>{if(reference.exactPlaying)return;const quality=video.getVideoPlaybackQuality?.();if(quality){playbackLabel.hidden=false;playbackLabel.textContent=`已解码 ${quality.totalVideoFrames} 帧 · 浏览器丢弃 ${quality.droppedVideoFrames} 帧`;}};
      exact.onclick=()=>{
        const start=!reference.exactPlaying||reference.exactPlayer?.ended;
        reference.exactPlayer?.stop();reference.exactPlayer=null;
        reference.exactPlaying=start;reference.playing=start;
        video.pause();video.hidden=true;canvas.hidden=!start;image.hidden=start;waiting.hidden=true;
        exact.textContent=start?'停止逐帧回放':'逐帧回放（不主动跳帧）';replay.textContent='播放器回放';
        playbackLabel.hidden=!start;
        if(start){
          playbackLabel.textContent='逐帧回放准备中；负载高时放慢速度，不主动跳帧…';
          reference.exactPlayer=new LiveCanvasPlayer(canvas,playbackLabel,`/api/jobs/${job.id}/cameras/${camera.id}/frames`,true,()=>{exact.textContent='重新逐帧回放';});
        }
      };
      replay.onclick=()=>{
        if(reference.exactPlaying){reference.exactPlayer?.stop();reference.exactPlayer=null;reference.exactPlaying=false;reference.playing=false;canvas.hidden=true;exact.textContent='逐帧回放（不主动跳帧）';}
        reference.playing=!reference.playing;replay.textContent=reference.playing?'停止回放':'播放器回放';
        if(reference.mp4Available) {
          image.hidden=reference.playing;video.hidden=!reference.playing;playbackLabel.hidden=!reference.playing;
          if(reference.playing){
            if(!video.getAttribute('src')) video.src=`/api/jobs/${job.id}/cameras/${camera.id}/video.mp4`;
            playbackLabel.textContent='正在准备播放缓冲…';
            playBufferedVideo(video,()=>reference.playing&&state.cards.get(camera.id)===reference).catch(()=>{playbackLabel.textContent='播放未开始，请点击播放器播放按钮，或检查视频连接。';});
          } else video.pause();
        } else {
          image.src=reference.playing?`/api/jobs/${job.id}/cameras/${camera.id}/replay.mjpeg?t=${Date.now()}`:`/api/jobs/${job.id}/cameras/${camera.id}/frame.jpg?v=${reference.version}`;
        }
      };
      image.onerror=()=>{if(!reference.playing){image.hidden=true;waiting.hidden=false;waiting.textContent='画面暂时不可用，等待更新…';reference.version=-1;}};
      image.onload=()=>{image.hidden=reference.playing&&reference.mp4Available;waiting.hidden=true;};
      root.append(heading,liveLabel,canvas,waiting,annotationLabel,image,video,playbackLabel,exact,replay);$('camera-grid').append(root);state.cards.set(camera.id,card);
    }
    const cameraStatus=job.status==='cancelled'?'已停止':job.status==='failed'?'已中断':cameraText[camera.status]||camera.status;
    card.status.textContent=`${cameraStatus} · ${camera.frames} 帧`;
    if(job.mode==='online') {
      card.liveLabel.hidden=false;card.annotationLabel.hidden=false;
      card.annotationLabel.textContent=`模型标注 · 已收 ${camera.decoded_frames??0} 帧 / 已推理 ${camera.frames} 帧 / 缓冲 ${camera.inference_queue??0} 帧 / 推理溢出 ${camera.inference_skipped??0} 帧 / 播放发布丢帧 ${camera.preview_dropped??0} 帧 · 标注耗时 ${camera.inference_age_ms??0} ms`;
      if(!terminal.has(job.status)&&!card.livePlayer)card.livePlayer=new LiveCanvasPlayer(card.canvas,card.liveLabel,`/api/jobs/${job.id}/cameras/${camera.id}/live`);
      if(terminal.has(job.status)){card.livePlayer?.stop();card.liveLabel.textContent='连续原画已停止；下方为最后一次模型标注';}
    }
    card.mp4Available=job.artifacts.some(a=>a.name===`camera-${camera.id+1}.mp4`);
    if(terminal.has(job.status)&&!camera.preview_version)card.waiting.textContent='此任务未收到可显示的画面';
    if(camera.preview_version>0&&card.version!==camera.preview_version&&!card.playing){card.version=camera.preview_version;card.image.src=`/api/jobs/${job.id}/cameras/${camera.id}/frame.jpg?v=${camera.preview_version}`;}
    card.replay.hidden=!terminal.has(job.status)||(!card.mp4Available&&!job.artifacts.some(a=>a.name===`camera-${camera.id+1}.avi`));
    card.exact.hidden=job.mode!=='offline'||!terminal.has(job.status)||!card.mp4Available;
    card.replay.title=card.mp4Available?'MP4 原生缓冲播放，可拖动进度':'旧任务仅有 AVI，使用兼容 MJPEG 回放；新建任务将生成 MP4';
  }
  $('downloads').hidden=!job.artifacts.length||!terminal.has(job.status);
  $('artifact-list').replaceChildren();
  for(const artifact of job.artifacts){const row=node('div','artifact'),link=node('a','',artifact.name);link.href=`/api/jobs/${job.id}/artifacts/${encodeURIComponent(artifact.name)}`;link.download=artifact.name;row.append(link,node('span','muted',formatSize(artifact.size)));$('artifact-list').append(row);}
  updateControls();
}
function renderHistory() {
  $('job-list').replaceChildren();
  if(!state.jobs.length){$('job-list').append(node('p','muted','还没有追踪任务。'));return;}
  for(const job of state.jobs){const row=node('button',`history-item${state.selected?.id===job.id?' selected':''}`),label=node('span','',`${job.mode==='online'?'实时在线':'离线视频'} / ${job.cameras.length} 路摄像头`);label.append(node('small','',new Date(job.created_at*1000).toLocaleString('zh-CN',{hour12:false})));row.append(label,node('span','',statusText[job.status]||job.status));row.onclick=()=>selectJob(job);$('job-list').append(row);}
}
async function refresh() {
  if(state.refreshing)return;state.refreshing=true;
  try{
    const [health,result]=await Promise.all([api('/api/health'),api('/api/jobs')]);
    state.online=true;state.limit=health.max_upload_bytes;state.jobs=result.jobs;
    $('health').textContent=health.model_files_ready?'服务已连接 · 权重就绪':'服务已连接 · 请配置本地模型';$('health').className='connection connected';
    let selected=state.jobs.find(j=>j.id===state.selected?.id);
    if(!selected&&state.selected)selected=await api(`/api/jobs/${state.selected.id}`);
    if(selected)renderJob(selected);else if(!state.selected&&state.jobs.length)selectJob(activeJob()||state.jobs[0]);
    renderHistory();
  }catch(error){state.online=false;$('health').textContent='后端连接中断，正在重试…';$('health').className='connection';}
  finally{state.refreshing=false;updateControls();}
}
async function poll(){await refresh();setTimeout(poll,1200);}
$('online-tab').onclick=()=>setMode('online');$('offline-tab').onclick=()=>setMode('offline');
document.querySelector('[role=tablist]').onkeydown=e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();setMode(e.key==='Home'?'online':e.key==='End'?'offline':state.mode==='online'?'offline':'online');$(state.mode+'-tab').focus();}};
$('add-stream').onclick=()=>{state.streams.push('');renderStreams();$(`stream-${state.streams.length-1}`).focus();};
$('files').onchange=e=>{addFiles(e.target.files);e.target.value='';};
$('dropzone').ondragover=e=>{e.preventDefault();$('dropzone').classList.add('drag');};
$('dropzone').ondragleave=()=>$('dropzone').classList.remove('drag');
$('dropzone').ondrop=e=>{e.preventDefault();$('dropzone').classList.remove('drag');addFiles(e.dataTransfer.files);};
$('job-form').onsubmit=submit;$('refresh').onclick=refresh;
$('stop').onclick=async()=>{if(!state.selected)return;try{setBusy(true);const job=await api(`/api/jobs/${state.selected.id}/stop`,{method:'POST'});renderJob(job);}catch(e){showError('task-error',e.message);}finally{setBusy(false);}};
renderStreams();setMode('online');poll();
window.addEventListener('pagehide',clearCards);
// 可用时暴露与页面一致的“配置”动作；不暗中启动推理，也不返回流密码。
if(document.modelContext?.registerTool){
  const lifecycle=new AbortController();
  Promise.resolve(document.modelContext.registerTool({name:'configure_online_streams',title:'配置在线流',description:'切换到实时在线模式并填写流地址，不启动追踪任务。',inputSchema:{type:'object',properties:{urls:{type:'array',items:{type:'string'},minItems:1}},required:['urls'],additionalProperties:false},annotations:{readOnlyHint:false},execute(input){if(state.busy)throw new Error('任务提交中，暂不可修改。');if(!input||!Array.isArray(input.urls)||!input.urls.every(v=>typeof v==='string'))throw new Error('urls 必须是字符串数组。');validateStreams(input.urls);state.streams=input.urls.map(v=>v.trim());setMode('online');renderStreams();return{mode:'online',camera_count:state.streams.length,started:false};}},{signal:lifecycle.signal})).catch(()=>{});
  window.addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
}
