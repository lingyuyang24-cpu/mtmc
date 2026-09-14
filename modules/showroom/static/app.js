'use strict';
const $ = id => document.getElementById(id);
const base = '/api/showroom/v1';
const state = {
  sid: '',
  stores: [],
  jobs: [],
  layouts: [],
  floorplan: null,
  mapImage: null,
  cameraImage: null,
  vehicles: [],
  cameras: [],
  mode: null,
  polygon: [],
  points: [],
  pending: null,
  live: [],
  dirty: false,
  busy: false
};
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;'
} [c]));
const statuses = {
  draft: '当日草稿',
  complete: '统计已完成',
  partial: '数据不完整',
  no_data: '无有效数据'
};

function notify(message, error = false) {
  $('notice').textContent = message;
  $('notice').className = error ? 'error' : '';
  $('notice').hidden = false;
}

function ask(message) {
  return new Promise(resolve => {
    const dialog = $('confirm-dialog');
    $('confirm-message').textContent = message;
    const done = value => {
      dialog.close();
      resolve(value);
    };
    $('confirm-yes').onclick = () => done(true);
    $('confirm-no').onclick = () => done(false);
    dialog.oncancel = e => {
      e.preventDefault();
      done(false);
    };
    dialog.showModal();
  });
}
async function api(path, body, raw = false) {
  const response = await fetch(base + path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: body === undefined ? {} : {
      'X-MTMC-Client': 'web',
      'Content-Type': raw ? 'application/octet-stream' : 'application/json'
    },
    body: body === undefined ? undefined : raw ? body : JSON.stringify(body)
  });
  const content = await response.json();
  if (!response.ok) throw new Error(Array.isArray(content.detail) ? content.detail.map(e => e.msg)
    .join('；') : content.detail || '请求失败');
  return content;
}

function route(suffix = '') {
  if (!state.sid) throw new Error('请先创建或选择门店。');
  return '/stores/' + state.sid + suffix;
}

function action(id, fn, event = 'click') {
  $(id).addEventListener(event, async e => {
    e.preventDefault();
    if (state.busy) return notify('请等待当前操作完成。');
    state.busy = true;
    $('stores').disabled = true;
    const button = e.currentTarget;
    button.disabled = true;
    try {
      await fn(e);
    } catch (error) {
      notify(error.message, true);
    } finally {
      button.disabled = false;
      $('stores').disabled = false;
      state.busy = false;
    }
  });
}

function tab(name) {
  document.querySelectorAll('[data-tab]').forEach(b => b.classList.toggle('selected', b.dataset
    .tab === name));
  document.querySelectorAll('.tab-panel').forEach(p => p.hidden = p.id !== name);
}
document.querySelectorAll('[data-tab]').forEach(b => b.addEventListener('click', () => tab(b.dataset
  .tab)));

function localTime(date) {
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 19);
}

function selectedStore() {
  return state.stores.find(s => s.id === state.sid);
}

function storeDay() {
  return new Intl.DateTimeFormat('sv-SE', {
    timeZone: selectedStore()?.timezone || 'Asia/Shanghai',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit'
  }).format(new Date());
}

function effectiveDefault(first = false) {
  const d = new Date();
  if (first) d.setHours(0, 0, 0, 0);
  else d.setSeconds(d.getSeconds() + 30);
  $('effective').value = localTime(d);
}

function imageFrom(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('图片无法加载，请检查上传文件或在线任务。'));
    img.src = url;
  });
}

function mapSize() {
  const w = Number($('map-width').value),
    h = Number($('map-height').value);
  if (!(w > 0 && h > 0)) throw new Error('请输入有效平面图尺寸。');
  return [w, h];
}

function resetDraft() {
  Object.assign(state, {
    floorplan: null,
    mapImage: null,
    cameraImage: null,
    vehicles: [],
    cameras: [],
    mode: null,
    polygon: [],
    points: [],
    pending: null,
    dirty: false
  });
  effectiveDefault(true);
  renderEditors();
}
async function refreshStores(preferred) {
  state.stores = (await api('/stores')).stores;
  $('stores').replaceChildren(...state.stores.map(s => new Option(s.name, s.id)));
  if (!state.stores.length) $('stores').add(new Option('请先创建门店', ''));
  state.sid = preferred || state.stores[0]?.id || '';
  $('stores').value = state.sid;
  await changeStore();
}
async function changeStore() {
  resetDraft();
  state.layouts = [];
  state.live = [];
  $('report-summary').replaceChildren();
  $('report-day').value = storeDay();
  if (state.sid) {
    state.layouts = (await api(route('/layouts'))).layouts;
    if (state.layouts.length) await loadLayout();
    await Promise.all([refreshLive(), refreshReports()]);
  } else {
    drawLive();
    $('store-form').hidden = false;
  }
}
async function loadLayout() {
  if (!state.layouts.length) throw new Error('尚无已发布布局。');
  const layout = state.layouts.at(-1);
  state.floorplan = layout.floorplan_id;
  state.mapImage = await imageFrom(base + route('/floorplans/' + state.floorplan + '.png'));
  $('map-width').value = layout.width_m;
  $('map-height').value = layout.height_m;
  state.vehicles = structuredClone(layout.vehicles);
  state.cameras = layout.cameras.map(({
    camera_id,
    width,
    height,
    points
  }) => ({
    camera_id,
    width,
    height,
    points: structuredClone(points)
  }));
  state.mode = null;
  state.polygon = [];
  state.points = [];
  state.pending = null;
  state.dirty = false;
  effectiveDefault();
  $('published').textContent = `已载入布局 v${layout.version}。修改后需重新发布；旧版本保留。`;
  renderEditors();
}
async function refreshJobs() {
  state.jobs = (await api('/tracking/jobs')).jobs;
  const selected = $('jobs').value;
  $('jobs').replaceChildren(new Option('请选择在线任务', ''), ...state.jobs.map(j => new Option(
    `${j.id.slice(0,8)} · ${j.status} · ${j.cameras.length} 路`, j.id)));
  if (state.jobs.some(j => j.id === selected)) $('jobs').value = selected;
}

function canvasBase(id, img, ratio) {
  const canvas = $(id);
  canvas.width = 1000;
  canvas.height = Math.round(1000 / Math.max(.15, Math.min(6, ratio || 1.54)));
  const c = canvas.getContext('2d');
  c.fillStyle = '#edf3f3';
  c.fillRect(0, 0, canvas.width, canvas.height);
  if (img) c.drawImage(img, 0, 0, canvas.width, canvas.height);
  else {
    c.fillStyle = '#779096';
    c.textAlign = 'center';
    c.font = '20px sans-serif';
    c.fillText('尚未配置画面', canvas.width / 2, canvas.height / 2);
  }
  return c;
}

function drawPolygon(c, points, w, h, label, pending = false) {
  if (!points.length) return;
  const canvas = c.canvas;
  c.beginPath();
  points.forEach(([x, y], i) => c[i ? 'lineTo' : 'moveTo'](x / w * canvas.width, y / h * canvas
    .height));
  if (!pending) c.closePath();
  c.strokeStyle = pending ? '#e89732' : '#078579';
  c.lineWidth = 3;
  c.fillStyle = pending ? '#e8973233' : '#13a48c33';
  c.fill();
  c.stroke();
  points.forEach(([x, y], i) => drawPoint(c, x / w * canvas.width, y / h * canvas.height, pending ?
    String(i + 1) : '', pending ? '#bd7717' : '#087d73'));
  if (label) {
    const x = points.reduce((s, p) => s + p[0], 0) / points.length / w * canvas.width,
      y = points.reduce((s, p) => s + p[1], 0) / points.length / h * canvas.height;
    c.font = 'bold 18px sans-serif';
    c.textAlign = 'center';
    c.fillStyle = '#075e56';
    c.fillText(label, x, y);
  }
}

function drawPoint(c, x, y, label, color = '#087d73') {
  c.beginPath();
  c.arc(x, y, 7, 0, Math.PI * 2);
  c.fillStyle = color;
  c.fill();
  c.strokeStyle = 'white';
  c.lineWidth = 2;
  c.stroke();
  if (label) {
    c.font = 'bold 17px sans-serif';
    c.textAlign = 'left';
    c.strokeStyle = 'white';
    c.lineWidth = 4;
    c.strokeText(label, x + 11, y - 8);
    c.fillStyle = color;
    c.fillText(label, x + 11, y - 8);
  }
}

function drawEditors() {
  let w = Number($('map-width').value) || 20,
    h = Number($('map-height').value) || 13;
  const c = canvasBase('edit-map', state.mapImage, w / h);
  state.vehicles.forEach(v => drawPolygon(c, v.polygon, w, h, v.name));
  drawPolygon(c, state.polygon, w, h, '', true);
  state.points.forEach((p, i) => drawPoint(c, p[2] / w * c.canvas.width, p[3] / h * c.canvas.height,
    String(i + 1), '#325fd0'));
  const camera = canvasBase('camera-map', state.cameraImage, state.cameraImage ? state.cameraImage
    .width / state.cameraImage.height : 16 / 9);
  const cw = Number($('camera-width').value),
    ch = Number($('camera-height').value);
  state.points.forEach((p, i) => drawPoint(camera, p[0] / cw * camera.canvas.width, p[1] / ch *
    camera.canvas.height, String(i + 1), '#325fd0'));
  if (state.pending) drawPoint(camera, state.pending[0] / cw * camera.canvas.width, state.pending[
    1] / ch * camera.canvas.height, '待配对', '#bd7717');
}

function payload() {
  const [width_m, height_m] = mapSize();
  return {
    floorplan_id: state.floorplan,
    width_m,
    height_m,
    effective_at: new Date($('effective').value).toISOString(),
    vehicles: state.vehicles,
    cameras: state.cameras
  };
}

function renderEditors() {
  drawEditors();
  $('edit-mode').textContent = state.mode === 'vehicle' ? '绘制车辆区域' : state.mode === 'camera' ?
    '地面对应点' : '选择绘制模式';
  $('edit-hint').textContent = state.mode === 'camera' ? (state.pending ? '请在此图点击对应地面位置。' :
      '先点击下方摄像头画面中的地面点。') : state.mode === 'vehicle' ? `已选 ${state.polygon.length} 点；至少 3 点后保存区域。` :
    '从右侧开始绘制车辆区域，或在下方开始标定。';
  $('point-hint').textContent =
    `已配对 ${state.points.length} 组 / 至少 4 组。${state.pending?'等待平面图配对。':''}`;
  $('points').innerHTML = state.points.map(p =>
    `<li>画面 (${p[0].toFixed(1)}, ${p[1].toFixed(1)}) → 平面图 (${p[2].toFixed(2)}, ${p[3].toFixed(2)}) 米</li>`
    ).join('');
  list('vehicle-list', state.vehicles, v => `${v.name} · ${v.id}`, i => {
    state.vehicles.splice(i, 1);
    state.dirty = true;
    renderEditors();
  });
  list('camera-list', state.cameras, c =>
    `摄像头 ${c.camera_id} · ${c.width} × ${c.height} · ${c.points.length} 点`, i => {
      state.cameras.splice(i, 1);
      state.dirty = true;
      renderEditors();
    });
  try {
    $('config-preview').textContent = JSON.stringify(payload(), null, 2);
  } catch {
    $('config-preview').textContent = '请完善尺寸与生效时间。';
  }
}

function list(id, items, title, remove) {
  $(id).replaceChildren(...items.map((v, i) => {
    const row = document.createElement('div');
    row.className = 'item';
    const text = document.createElement('span');
    text.textContent = title(v);
    const button = document.createElement('button');
    button.textContent = '移除';
    button.className = 'secondary';
    button.onclick = () => remove(i);
    row.append(text, button);
    return row;
  }));
}

function clickPosition(e, width, height) {
  const rect = e.currentTarget.getBoundingClientRect();
  return [(e.clientX - rect.left) / rect.width * width, (e.clientY - rect.top) / rect.height *
    height
  ];
}
$('edit-map').addEventListener('click', e => {
  try {
    if (!state.mapImage) throw new Error('请先上传平面图。');
    const p = clickPosition(e, ...mapSize());
    if (state.mode === 'vehicle') {
      if (state.polygon.length >= 32) throw new Error('每个区域最多 32 点。');
      state.polygon.push(p);
    } else if (state.mode === 'camera' && state.pending) {
      state.points.push([...state.pending, ...p]);
      state.pending = null;
    } else return;
    renderEditors();
  } catch (error) {
    notify(error.message, true);
  }
});
$('camera-map').addEventListener('click', e => {
  if (state.mode !== 'camera' || !state.cameraImage) return;
  if (state.points.length >= 32) return notify('每个摄像头最多 32 组标定点。', true);
  state.pending = clickPosition(e, Number($('camera-width').value), Number($('camera-height')
    .value));
  renderEditors();
});
action('new-store', () => {
  $('store-form').hidden = !$('store-form').hidden;
});
action('store-form', async () => {
  const store = await api('/stores', {
    name: $('store-name').value.trim(),
    timezone: $('timezone').value,
    opens: $('opens').value,
    closes: $('closes').value,
    min_dwell: Number($('min-dwell').value)
  });
  await refreshStores(store.id);
  $('store-form').hidden = true;
  tab('configure');
  notify('门店已创建，请上传平面图并配置摄像头。');
}, 'submit');
$('stores').addEventListener('change', async () => {
  try {
    if (state.dirty && !await ask('切换门店会放弃未发布的布局修改，是否继续？')) {
      $('stores').value = state.sid;
      return;
    }
    state.sid = $('stores').value;
    await changeStore();
  } catch (error) {
    notify(error.message, true);
  }
});
action('refresh-jobs', refreshJobs);
action('bind', async () => {
  const job = state.jobs.find(j => j.id === $('jobs').value);
  if (!job) throw new Error('请选择在线任务。');
  if (!await ask(
      `请确认：此任务的摄像头顺序与标定一致。\n${job.cameras.map(c=>`${c.id}：${c.name}`).join('\n')}\n绑定后会从该任务开头补读可用日志。`
      )) return;
  await api(route('/bindings'), {
    job_id: job.id
  });
  await refreshLive();
  notify('已绑定，分析服务将持续读取事件。');
});
action('floorplan-file', async () => {
  route();
  const file = $('floorplan-file').files[0];
  if (!file) return;
  if (file.size > 8 * 1024 * 1024) throw new Error('图片不能超过 8 MiB。');
  if ((state.vehicles.length || state.cameras.length) && !await ask(
      '更换平面图会清空待发布的车辆区域和标定，是否继续？')) return;
  const asset = await api(route('/floorplans'), file, true);
  state.floorplan = asset.id;
  state.mapImage = await imageFrom(base + route('/floorplans/' + asset.id + '.png'));
  state.vehicles = [];
  state.cameras = [];
  state.points = [];
  state.polygon = [];
  state.pending = null;
  state.mode = null;
  state.dirty = true;
  renderEditors();
  notify('平面图已上传。请核对实际尺寸。');
}, 'change');
for (const id of ['map-width', 'map-height']) $(id).addEventListener('change', () => {
  state.dirty = true;
  renderEditors();
});
for (const id of ['camera-width', 'camera-height']) $(id).addEventListener('change', () => {
  state.points = [];
  state.pending = null;
  renderEditors();
});
action('draw-vehicle', () => {
  if (!state.mapImage) throw new Error('请先上传平面图。');
  state.mode = 'vehicle';
  state.polygon = [];
  state.pending = null;
  renderEditors();
});
action('finish-vehicle', () => {
  if (state.mode !== 'vehicle' || state.polygon.length < 3) throw new Error('请先绘制至少 3 个点。');
  const id = $('vehicle-id').value.trim(),
    name = $('vehicle-name').value.trim();
  if (!/^[A-Za-z0-9_-]{1,48}$/.test(id) || !name) throw new Error('请填写编号（字母、数字、下划线、短横线）和展车名称。');
  if (state.vehicles.some(v => v.id === id)) throw new Error('车辆编号已存在，请先移除旧区域再绘制。');
  state.vehicles.push({
    id,
    name,
    model: $('vehicle-model').value.trim(),
    polygon: state.polygon
  });
  state.polygon = [];
  state.mode = null;
  state.dirty = true;
  renderEditors();
});
action('undo', () => {
  if (state.mode === 'vehicle') state.polygon.pop();
  else if (state.pending) state.pending = null;
  else state.points.pop();
  renderEditors();
});
action('cancel-drawing', () => {
  state.mode = null;
  state.polygon = [];
  state.pending = null;
  state.points = [];
  renderEditors();
});
action('camera-file', async () => {
  const file = $('camera-file').files[0];
  if (!file) return;
  if (file.size > 8 * 1024 * 1024) throw new Error('截图不能超过 8 MiB。');
  const url = URL.createObjectURL(file);
  try {
    state.cameraImage = await imageFrom(url);
    $('camera-width').value = state.cameraImage.width;
    $('camera-height').value = state.cameraImage.height;
  } finally {
    URL.revokeObjectURL(url);
  }
  state.points = [];
  state.pending = null;
  renderEditors();
  notify('截图已载入，若截图被缩小，请填写原视频尺寸后再开始标定。');
}, 'change');
action('fetch-frame', async () => {
  const id = $('jobs').value;
  if (!id) throw new Error('请先在实时概览选择在线任务。');
  await refreshJobs();
  const job = state.jobs.find(j => j.id === id),
    camera = job?.cameras.find(c => c.id === Number($('camera-id').value));
  if (!camera?.source_width || !camera?.source_height) throw new Error(
    '任务尚未提供原视频尺寸，请升级后新建在线任务并等待首帧，或上传截图手工填写尺寸。');
  state.cameraImage = await imageFrom(
    `${base}/tracking/jobs/${id}/cameras/${camera.id}/frame.jpg?t=${Date.now()}`);
  $('camera-width').value = camera.source_width;
  $('camera-height').value = camera.source_height;
  state.points = [];
  state.pending = null;
  renderEditors();
});
action('start-calibration', () => {
  if (!state.mapImage || !state.cameraImage) throw new Error('请先载入平面图和摄像头画面。');
  const ratio = Number($('camera-width').value) / Number($('camera-height').value),
    actual = state.cameraImage.width / state.cameraImage.height;
  if (Math.abs(ratio / actual - 1) > .01) throw new Error('原视频尺寸与截图比例不一致，请核对，禁止使用裁剪截图。');
  state.mode = 'camera';
  state.points = [];
  state.pending = null;
  state.polygon = [];
  renderEditors();
});
action('save-camera', () => {
  if (state.mode !== 'camera' || state.points.length < 4 || state.pending) throw new Error(
    '请完成至少 4 组地面对应点。');
  const camera_id = Number($('camera-id').value);
  const camera = {
    camera_id,
    width: Number($('camera-width').value),
    height: Number($('camera-height').value),
    points: state.points
  };
  state.cameras = state.cameras.filter(c => c.camera_id !== camera_id);
  state.cameras.push(camera);
  state.points = [];
  state.pending = null;
  state.mode = null;
  state.dirty = true;
  renderEditors();
  notify('摄像头标定已加入草稿；发布时校验几何有效性。');
});
action('load-layout', async () => {
  if (state.dirty && !await ask('载入会放弃未发布修改，是否继续？')) return;
  state.layouts = (await api(route('/layouts'))).layouts;
  await loadLayout();
});
action('publish', async () => {
  if (state.mode) throw new Error('请先保存或取消正在绘制的区域 / 标定。');
  const layout = await api(route('/layouts'), payload());
  state.dirty = false;
  state.layouts = (await api(route('/layouts'))).layouts;
  $('published').textContent =
    `布局 v${layout.version} 已发布 · 控制点拟合误差：${layout.cameras.map(c=>`摄像头 ${c.camera_id} ${c.fit_error_m.toFixed(3)} 米`).join('；')}。这不是独立测点的定位精度。`;
  notify('布局已发布。请到实时概览确认并绑定在线任务。');
  await refreshLive();
});

function saveBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
action('export-config', () => saveBlob(new Blob([JSON.stringify(payload(), null, 2)], {
  type: 'application/json'
}), 'showroom-layout.json'));
action('import-config', async () => {
  route();
  const file = $('import-config').files[0];
  if (!file) return;
  if (file.size > 1024 * 1024) throw new Error('配置文件过大。');
  const value = JSON.parse(await file.text());
  if (!Array.isArray(value.vehicles) || !Array.isArray(value.cameras) || !Number.isFinite(
      value.width_m) || !Number.isFinite(value.height_m) || !/^[a-f0-9]{32}$/.test(value
      .floorplan_id)) throw new Error('配置结构不完整。');
  const img = await imageFrom(base + route('/floorplans/' + value.floorplan_id + '.png'));
  state.floorplan = value.floorplan_id;
  state.mapImage = img;
  state.vehicles = value.vehicles;
  state.cameras = value.cameras;
  $('map-width').value = value.width_m;
  $('map-height').value = value.height_m;
  $('effective').value = localTime(new Date(value.effective_at));
  state.mode = null;
  state.points = [];
  state.polygon = [];
  state.pending = null;
  state.dirty = true;
  renderEditors();
  notify('配置已载入草稿，发布时进行完整校验。图片 ID 必须属于当前门店；跨服务请先重新上传图片。');
}, 'change');
let liveImageKey = '',
  liveImage = null;
async function drawLive() {
  const layout = state.layouts.filter(l => l.effective_at <= Date.now() / 1000).at(-1);
  if (!layout) {
    canvasBase('live-map', null);
    $('live-hint').textContent = '请先发布已生效的布局。';
    return;
  }
  const key = state.sid + layout.floorplan_id;
  if (liveImageKey !== key) {
    liveImage = await imageFrom(base + route('/floorplans/' + layout.floorplan_id + '.png'));
    liveImageKey = key;
  }
  const c = canvasBase('live-map', liveImage, layout.width_m / layout.height_m);
  layout.vehicles.forEach(v => drawPolygon(c, v.polygon, layout.width_m, layout.height_m, v
  .name));
  state.live.filter(p => p.layout === layout.version && p.x !== null && p.y !== null).forEach(p =>
    drawPoint(c, p.x / layout.width_m * c.canvas.width, p.y / layout.height_m * c.canvas.height,
      `G${p.gid} / C${p.camera}`, '#3659b9'));
  $('live-hint').textContent = `当前布局 v${layout.version} · 显示模型位置近似。跨运行身份不自动去重；区域冲突会在日报中剔除。`;
}
async function refreshLive() {
  if (!state.sid) return;
  const sid = state.sid;
  const data = await api(route('/live'));
  if (sid !== state.sid) return;
  state.layouts = data.layouts;
  state.live = data.positions;
  $('live-count').textContent = `${data.positions.length} 个观察`;
  $('bindings').replaceChildren(...data.bindings.map(b => {
    const row = document.createElement('div');
    row.className = 'item';
    const text = document.createElement('div');
    text.innerHTML =
      `<strong>${esc(b.run.slice(0,8))}</strong><small>${b.paused?'分析已暂停':b.error?esc(b.error):b.caught_up?'已读至当前日志末尾':'等待 / 补读中'} · 上游 ${esc(b.status)}</small><small>事件丢弃 ${Number(b.dropped)||0}</small>`;
    const button = document.createElement('button');
    button.className = 'secondary';
    button.textContent = b.paused ? '恢复' : '暂停';
    button.onclick = async () => {
      button.disabled = true;
      try {
        await api(route('/bindings/' + b.run + '/pause'), {
          paused: !b.paused
        });
        await refreshLive();
      } catch (error) {
        notify(error.message, true);
      } finally {
        button.disabled = false;
      }
    };
    row.append(text, button);
    return row;
  }));
  if (!data.bindings.length) $('bindings').textContent = '尚未绑定追踪任务。';
  if (!document.activeElement?.closest('#positions')) {
    $('positions').innerHTML = data.positions.length ? data.positions.map(p =>
      `<tr><td>${esc(p.run.slice(0,8))} / ${p.gid}</td><td>${p.camera}</td><td>${p.x===null?'未知':p.x.toFixed(2)+', '+p.y.toFixed(2)}</td><td>${esc(p.vehicle||'区域外 / 未知')}</td><td><select aria-label="标记 GID ${p.gid} 的角色" data-run="${esc(p.run)}" data-gid="${p.gid}"><option value="staff" ${p.role==='staff'?'selected':''}>员工（日报排除）</option><option value="visitor" ${p.role==='visitor'?'selected':''}>访客</option><option value="unknown" ${!p.role||p.role==='unknown'?'selected':''}>未知</option></select></td></tr>`
      ).join('') : '<tr><td colspan="5">暂无最近 10 秒的观察；请检查任务、绑定与消费状态。</td></tr>';
  }
  await drawLive();
}
$('positions').addEventListener('change', async e => {
  const select = e.target;
  if (!select.dataset.run || !select.value) return;
  try {
    await api(route('/roles'), {
      run_id: select.dataset.run,
      global_id: Number(select.dataset.gid),
      role: select.value
    });
    notify('角色已保存，请重新生成受影响日期的报告。');
  } catch (error) {
    notify(error.message, true);
  }
});
async function refreshReports() {
  if (!state.sid) return;
  const data = await api(route('/reports'));
  $('report-history').innerHTML = data.reports.length ? data.reports.map(r =>
    `<div class="item"><span><strong>${esc(r.day)}</strong><small>修订 ${r.revision} · ${esc(new Date(r.generated*1000).toLocaleString())}</small></span><span><a target="_blank" rel="noopener" href="${base+route('/reports/'+r.day+'.html')}">查看 HTML ↗</a> · <a href="${base+route('/reports/'+r.day+'.json?download=true')}">JSON 下载</a> · <a href="${base+route('/reports/'+r.day+'.html?download=true')}">HTML 下载</a></span></div>`
    ).join('') : '<p class="muted">还没有日报。可手动生成当日草稿。</p>';
}
action('refresh-reports', refreshReports);
action('generate', async () => {
  const day = $('report-day').value;
  if (!day) throw new Error('请选择日期。');
  const p = await api(route('/reports/' + day), {});
  $('report-summary').innerHTML =
    `<p><span class="pill">${esc(statuses[p.status])}</span> ${esc(day)} · ${esc(p.timezone)} · 修订 ${p.revision}</p><div class="cards">${[['人员会话',p.summary.people],['全部展车',p.summary.vehicles],['区域停留 / 分钟',(p.summary.valid_seconds/60).toFixed(1)],['未映射观察',p.summary.unmapped_observations]].map(([label,value])=>`<div class="card"><small>${esc(label)}</small><strong>${esc(value)}</strong></div>`).join('')}</div><p class="note">${p.warnings.map(esc).join('<br>')}</p>`;
  await refreshReports();
  notify('日报已生成；无数据或覆盖不足会明确标记，不作为零客流结论。');
});
$('effective').addEventListener('change', () => {
  state.dirty = true;
  renderEditors();
});
window.addEventListener('beforeunload', e => {
  if (state.dirty) {
    e.preventDefault();
    e.returnValue = '';
  }
});
async function heartbeat() {
    try {
      const health = await api('/health');
      $('connection').textContent = health.worker_error ? '分析后台异常：' + health.worker_error :
        '分析服务已连接';
      const url = new URL(health.tracking_url);
      if (['http:', 'https:'].includes(url.protocol)) {
        for (const mode of ['online', 'offline']) {
          const target = new URL(url.href);
          target.searchParams.set('mode', mode);
          $('nav-' + mode).href = target.href;
          $('nav-' + mode).setAttribute('aria-disabled', 'false');
        }
      }
      if (state.sid) await refreshLive();
    } catch (error) {
      $('connection').textContent = '分析服务连接中断';
    } finally {
      setTimeout(heartbeat, 3000);
    }
  }
  (async () => {
    try {
      effectiveDefault(true);
      await refreshStores();
      try {
        await refreshJobs();
      } catch (error) {
        notify(error.message, true);
      }
      await heartbeat();
    } catch (error) {
      notify(error.message, true);
      $('connection').textContent = '连接失败，请刷新页面重试';
    }
  })();
