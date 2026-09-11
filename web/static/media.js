'use strict';
// 首播先积累一秒可播数据，避免拿到首帧便抢跑。短片等待整段就绪。
async function playBufferedVideo(video, stillPlaying) {
  video.preload = 'auto';
  if (video.readyState < HTMLMediaElement.HAVE_ENOUGH_DATA) {
    await new Promise((resolve, reject) => {
      const events = ['progress', 'canplaythrough', 'loadeddata'];
      let timer;
      const cleanup = () => {events.forEach(e=>video.removeEventListener(e, check));video.removeEventListener('error', fail);clearTimeout(timer);};
      const fail = () => {cleanup();reject(new Error('视频缓冲失败'));};
      const check = () => {
        const end = video.buffered.length ? video.buffered.end(video.buffered.length-1) : 0;
        const needed = Number.isFinite(video.duration) ? Math.min(1, video.duration-video.currentTime) : 1;
        if (video.readyState >= HTMLMediaElement.HAVE_ENOUGH_DATA || end-video.currentTime >= needed) {cleanup();resolve();}
      };
      events.forEach(e=>video.addEventListener(e, check));video.addEventListener('error', fail);
      timer=setTimeout(fail,15000);check();
    });
  }
  if (stillPlaying()) await video.play();
}

// 原画播放器与模型标注分离；逐帧解码、绘制、ACK，不靠定时刷新 img。
class LiveCanvasPlayer {
  constructor(canvas, label, path, exact = false, onEnded = () => {}) {
    this.canvas = canvas;
    this.label = label;
    this.path = path;
    this.exact = exact;
    this.onEnded = onEnded;
    this.ended = false;
    this.closed = false;
    this.frames = 0;
    this.skipped = 0;
    this.started = performance.now();
    this.onVisibility = () => {
      // 离线保留当前帧与连接，恢复可见后继续 ACK，不从头播放或跳过积压。
      if (this.exact) return;
      clearTimeout(this.timer);
      if (document.hidden) this.socket?.close();
      else if (!this.closed && (!this.socket || this.socket.readyState > WebSocket.OPEN)) this.connect();
    };
    document.addEventListener('visibilitychange', this.onVisibility);
    this.connect();
  }

  connect() {
    if (this.closed || (!this.exact && document.hidden)) return;
    this.started = performance.now();
    this.frames = 0;
    this.skipped = 0;
    const url = new URL(this.path, location.href);
    url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(url);
    this.socket = socket;
    let metadata = null;
    socket.onmessage = async event => {
      if (typeof event.data === 'string') {
        const message = JSON.parse(event.data);
        if (message.type === 'waiting') {
          if (socket.readyState === WebSocket.OPEN) socket.send('waiting');
        } else if (message.type === 'ended') {
          this.ended = true;
          const complete = this.frames === message.frames && this.skipped === 0;
          this.label.textContent = `${complete ? '逐帧回放完成' : '回放帧数核对失败'} · 已绘制 ${this.frames} / ${message.frames} 帧 · 应用跳帧 ${this.skipped}`;
          this.onEnded();
        } else metadata = message;
        return;
      }
      const frameInfo = metadata;
      if (!frameInfo || this.closed) return;
      let image, objectUrl;
      try {
        if (this.exact && frameInfo.seq !== this.frames+1) throw new Error('帧序号不连续');
        if ('createImageBitmap' in window) image = await createImageBitmap(event.data);
        else {
          objectUrl = URL.createObjectURL(event.data);
          image = new Image(); image.src = objectUrl; await image.decode();
        }
        await new Promise(resolve => requestAnimationFrame(resolve));
        if (this.closed || this.socket !== socket) return;
        if (this.canvas.width !== image.width) this.canvas.width = image.width;
        if (this.canvas.height !== image.height) this.canvas.height = image.height;
        this.canvas.getContext('2d', {alpha:false}).drawImage(image, 0, 0);
        this.canvas.hidden = false;
        this.frames++;
        this.skipped += frameInfo.skipped;
        const fps = this.frames * 1000 / Math.max(1, performance.now()-this.started);
        this.label.textContent = this.exact
          ? `逐帧回放 · 已绘制 ${this.frames} 帧 · ${fps.toFixed(1)} 帧/秒 · 应用跳帧 ${this.skipped}`
          : `连续原画 · ${fps.toFixed(1)} 帧/秒 · 会话跳帧 ${this.skipped}`;
        if (socket.readyState === WebSocket.OPEN) socket.send(String(frameInfo.seq));
      } catch {
        this.label.textContent = this.exact ? '回放解码失败，请重新开始回放。' : '画面解码失败，正在重新连接…';
        socket.close();
      } finally {
        image?.close?.();
        if (objectUrl) URL.revokeObjectURL(objectUrl);
      }
    };
    socket.onclose = event => {
      if (this.closed) return;
      if (this.exact) {
        if (!this.ended) this.label.textContent = `逐帧回放中断 · 已绘制 ${this.frames} 帧；请重新开始回放。`;
        return;
      }
      this.label.textContent = event.code === 1000 ? '连续原画已结束' : '播放连接中断，正在重连…';
      if (event.code !== 1000 && !document.hidden) this.timer = setTimeout(()=>this.connect(), 1500);
    };
    socket.onerror = () => socket.close();
  }

  stop() {
    this.closed = true;
    clearTimeout(this.timer);
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.socket?.close();
  }
}
