# MTMC 多摄像头行人追踪与重识别

使用 **YOLO11 + Deep SORT + TransReID** 检测和追踪行人，并在不同摄像头之间关联同一个人的全局身份（GID）。支持嵌入式 Web 工作台、实时视频流和离线视频文件。

每个摄像头都有独立的本地跟踪器；跨摄像头关联由 ReID 外观特征完成。GID 是模型推断的轨迹身份，不代表经过认证的真实人员身份。

## 1. 快速启动网页

先获取项目：

```bash
git clone https://github.com/lingyuyang24-cpu/mtmc.git
cd mtmc
```

仓库包含源码、网页、测试与示例展示素材，不包含模型权重、输入视频、运行结果或虚拟环境。首次运行前请按下方“模型文件”准备权重。

以下命令在项目的 `mtmc` 目录执行，建议使用 Python 3.12（已验证），推理依赖要求 Python 3.10 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-web.txt
python serve.py
```

Windows 激活虚拟环境时使用 `.venv\Scripts\activate`。如果项目已有可用的 `.venv`，直接激活、安装 Web 依赖并启动即可，无需重新创建。

打开 [本地工作台](http://127.0.0.1:8765/)。前端 HTML、CSS、JavaScript 由 FastAPI 直接提供，**无需 Node.js、npm 或单独启动前端服务**。交互式接口文档位于 [接口文档](http://127.0.0.1:8765/docs)，OpenAPI 描述位于 `/openapi.json`；接口文档页面的第三方静态资源可能需要联网，工作台自身不依赖 CDN。

默认仅监听本机。改变端口：

```bash
python serve.py --port 9000
```

模型在点击“开始追踪”后由独立子进程加载，网页和接口不会等待模型加载才启动。页面显示“权重就绪”仅表示所需文件存在，不等于模型已经加载成功。

### 模型文件

Web 工作台使用本地 YOLO11 检测权重和 TransReID MSMT17 权重，**不自动下载模型，也不接受浏览器指定的模型路径或 Python 适配器**。

默认位置：

```text
mtmc/
├── yolo11l.pt
└── external/transreid/
    ├── repo/
    └── weights/vit_transreid_msmt.pth
```

缺少 TransReID 时，先在联网环境中准备：

```bash
python transreid_adapter.py --download --variant msmt17
```

缺少 YOLO 权重时，可在联网环境执行以下命令下载，随后将文件保存在项目根目录：

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"
```

权重和 TransReID 源码准备完成后，离线视频追踪无需联网；实时模式仅需能够访问对应的视频流。默认自动选择 CUDA，否则使用 CPU；有 NVIDIA GPU 时请安装与驱动兼容的 PyTorch。当前默认推理路径不会自动选择 Apple MPS。

## 2. 页面使用

### 实时在线追踪

1. 选择“实时在线”。
2. 填写一个或多个 RTSP、HTTP 或 HTTPS 视频流地址，点击“添加视频流”继续增加，也可逐路移除。
3. 可展开“追踪参数”调整检测置信度、跨镜匹配阈值和特征提取批大小。
4. 点击“开始在线追踪”，等待模型加载。每路摄像头独立显示追踪画面、帧数和连接状态。
5. 点击“停止任务”释放视频流，并下载已经写出的 `tracks.jsonl` 轨迹日志。

输入数量不写死，输入顺序决定从 0 开始的 `camera_id`。GID 仅在同一任务内有效，不跨任务继承。实际可承载路数取决于内存、显存、分辨率、网络与推理速度。HTTP/HTTPS 必须指向 OpenCV 可以解码的视频流，不是摄像头管理网页。

Web 在线采用独立拉流、连续原画发布、模型推理三条路径。拉流线程不等待模型，每路推理队列最多保留 30 帧且不超过 64 MiB；短时积压按顺序处理，超出上限才丢弃最旧的待推理帧并累计计数，不会无限增加内存或延迟。RTSP 使用 TCP，Web 不启用 `nobuffer`。断流自动尝试重连；某一路异常不立刻终止其他摄像头，所有流长时间没有新画面则报告失败。

每路画面上方为 WebSocket 连续原画，下方为对应推理帧的模型标注，标注约每秒刷新一次。两者刻意分开：慢推理不会降低原画帧率，也不会把旧框叠到不对应的新帧上。灰色 `GID:?` 表示身份尚未确认。页面显示已收帧数、已推理帧数、排队帧数、推理溢出帧数、播放发布丢帧和标注耗时。在线默认只保存轨迹日志与最新标注图；需要录像可使用命令行 `--output`。

连续播放按收到的帧时间调度，浏览器逐帧解码、绘制并确认；单个浏览器最多一帧待确认。短缓冲吸收抖动，慢观众不会阻塞其他观众或拉流。每路编码输入缓存上限 60 帧/32 MiB，进程间传输上限 64 张压缩图，服务端共享缓存上限 512 张/32 MiB。只有发生积压溢出才跳帧；新连接从接近实时的位置加入，不补放历史。页面隐藏时暂停原画连接以释放资源，恢复可见后重连。

### 离线视频追踪

1. 选择“离线视频”。
2. 一次选择或拖入多个视频；也可以继续添加、移除文件。
3. 点击“开始离线追踪”。页面先显示逐文件上传进度，然后显示模型加载、逐帧追踪、身份融合和视频输出阶段。
4. 处理中显示的是本地轨迹 ID（LID）。跨摄像头融合完成后，最终画面和输出视频显示全局 ID（GID）。
5. 完成后可选择“逐帧回放（不主动跳帧）”或“播放器回放”，也可下载视频与身份映射。

支持 MP4、AVI、MOV、MKV、WEBM、M4V 扩展名，实际是否可读取决于文件编码与本机 PyAV/FFmpeg。文件不能解码时会报告任务失败。单文件默认上限 2 GiB，可由环境变量修改；文件数量不写死。重复选择同名、同大小、同修改时间的文件会在页面去重。

上传文件使用服务器生成的唯一 ID 保存，不会因同名文件相互覆盖。每个文件视为一个摄像头，按所选顺序编号；不同分辨率、长度和帧率的视频分别处理。MP4 保留输入帧的相对时间戳，支持可变帧率；为兼容 H.264 的 YUV420 格式，奇数宽高会补一列/行黑边，不裁剪原画。不同摄像头不假定做过硬件同步。

离线 Web 流程分两遍顺序读取文件，不缓存整段视频或全部人体裁剪：第一遍检测、跟踪、抽样保存外观特征并写入逐帧记录；融合后第二遍重新读取原视频，生成标注结果。每条轨迹最多保留 64 个均匀蓄水池抽样特征，融合代表库最多 32 个；内存仍会随轨迹数量增加，但不随全部视频像素量线性累积。

结果文件：

| 文件 | 内容 |
| --- | --- |
| `camera-1.mp4`、`camera-2.mp4` 等 | 原生浏览器播放的 H.264 全局 ID 标注视频，保留逐帧时间戳，支持缓冲和进度拖动 |
| `camera-1.avi`、`camera-2.avi` 等 | 各摄像头的全局 ID 标注视频，MJPEG 编码，不保留音轨 |
| `id_mapping.json` | 摄像头、本地 ID、内部数字轨迹 ID、全局 ID、起止帧和样本数 |
| `tracks.jsonl` | 每行一条目标观察：摄像头、帧号、时间、本地/全局 ID 与边框 |

新任务提供两种回放方式：

- **逐帧回放：**按 MP4 时间戳逐帧读取，通过 WebSocket 发送 JPEG，浏览器绘制后确认，才读取下一帧。负载较高时放慢速度，不主动跳帧追赶；隐藏页面时暂停绘制和取帧，重新可见后继续。页面显示实际绘制帧数，结束时核对总帧数。当前从头回放，不支持拖动；展示缩放至最大 960×720，完整分辨率保存在下载文件中。
- **播放器回放：**使用原生 MP4 播放器，支持 Range 分段读取、播放缓冲、暂停、拖动进度和倍速（取决于浏览器控件）。首播先等一秒可播缓冲或浏览器确认数据充足。画面下方显示浏览器报告的解码帧数和丢弃帧数；浏览器可能为保持实时速度主动丢弃显示帧，逐帧检查请用上一种模式。

MP4 使用 `faststart`，不必下载完整文件才开始播放；编码不使用抽帧/fps 滤镜，写完后核对编码帧数与处理帧数。容器声明了帧数时，会与实际解码帧数核对；两遍解码帧数不一致或解码出错也会报告失败，不把已发现的截断结果作为完成。AVI 为固定平均帧率兼容导出；可变帧率素材请使用 MP4。两种输出均不保留音轨。

已有历史任务仅有 AVI 时仍提供原 MJPEG 兼容回放，不自动修改已有结果；重新运行该离线任务即可生成新的 MP4。

**保帧边界：**离线处理不主动抽帧，播放帧率与模型推理耗时解耦。在线“原画播放完整”和“模型逐帧处理”是不同指标：模型速度长期低于输入速度时，有限内存下无法同时做到逐帧推理、低延迟和无限连续运行。网络/摄像头在解码前丢失的帧无法从 OpenCV 的本地序号统计得知；浏览器或硬件过载也可能丢弃显示帧，因此不能承诺任意网络、硬件和路数下绝对零丢帧。

### 任务管理与停止

- 一个服务同时运行一个追踪任务，在线与离线共用该限制。再次提交会返回 HTTP 409，防止多个模型实例同时抢占资源。
- 页面关闭或刷新不会停止后台任务，重新打开后可从“最近任务”查看进度和结果。
- 点击停止先请求正常结束；若解码器或模型调用没有响应，约 5 秒后强制终止子进程。只有已经完整生成并登记的结果文件可下载，停止时的部分结果不等于完整追踪结果。
- 任务信息持久化在磁盘上，列表显示最近 50 个任务。服务重启后保留历史结果，将未结束任务标记为中断，不自动恢复推理。

## 3. 后端与接口

```text
浏览器页面（同源 HTTP API）
    ↓ 上传 / 启动 / 查询 / 停止 / 预览 / 下载
FastAPI + 单任务管理器（持久化任务 JSON）
    ↓ 独立推理子进程
YOLO11 → 每摄像头 Deep SORT → TransReID 特征
    ├── 在线：多帧确认、身份复核、流重连、最新画面
    └── 离线：全轨迹融合、二次视频渲染、结果文件
```

任务状态为 `starting → running → completed / failed`，主动停止时为 `stopping → cancelled`。

所有写入接口必须带上 `X-MTMC-Client: web` 请求头；浏览器另外必须同源。该请求头是跨站请求防护的一部分，**不是用户认证或访问令牌**。

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/health` | 服务、权重文件状态、上传限制、当前任务 |
| POST | `/api/uploads?filename=xxx.mp4` | 上传一个原始二进制视频，返回上传 ID |
| POST | `/api/jobs/online` | 用自定义数量的流地址启动在线任务 |
| POST | `/api/jobs/offline` | 用自定义数量的上传 ID 启动离线任务 |
| GET | `/api/jobs` | 最近任务 |
| GET | `/api/jobs/{id}` | 状态、进度、摄像头统计和结果列表 |
| POST | `/api/jobs/{id}/stop` | 停止任务，重复请求安全 |
| GET | `/api/jobs/{id}/cameras/{camera_id}/frame.jpg` | 最新追踪画面 |
| GET | `/api/jobs/{id}/cameras/{camera_id}/video.mp4` | 原生 MP4 播放，支持 HTTP Range / 206 |
| WebSocket | `/api/jobs/{id}/cameras/{camera_id}/live` | 连续原画；同源握手，每帧 JSON 元数据 + JPEG，客户端绘制后回传序号字符串 |
| WebSocket | `/api/jobs/{id}/cameras/{camera_id}/frames` | 离线逐帧回放；同源握手、逐帧确认，不跳帧；EOF 返回 ended 与总帧数 |
| GET | `/api/jobs/{id}/cameras/{camera_id}/replay.mjpeg` | 已完成的离线视频回放 |
| GET | `/api/jobs/{id}/artifacts/{name}` | 下载该任务已登记的结果 |

上传使用原始二进制流，不是 multipart 表单；浏览器逐文件发送，后端分块写磁盘，不把整个文件读入内存。接口示例：

```bash
curl -X POST "http://127.0.0.1:8765/api/uploads?filename=camera1.mp4" \
  -H "X-MTMC-Client: web" -H "Content-Type: application/octet-stream" \
  --data-binary @videos/init/camera1.mp4
```

启动离线任务，替换下面的上传 ID：

```bash
curl -X POST http://127.0.0.1:8765/api/jobs/offline \
  -H "X-MTMC-Client: web" -H "Content-Type: application/json" \
  -d '{"upload_ids":["第一个上传ID","第二个上传ID"],"options":{"confidence":0.35,"reid_threshold":0.30,"batch_size":4,"min_reid_frames":10}}'
```

启动在线任务：

```bash
curl -X POST http://127.0.0.1:8765/api/jobs/online \
  -H "X-MTMC-Client: web" -H "Content-Type: application/json" \
  -d '{"streams":[{"name":"入口","url":"rtsp://camera-a/live"},{"name":"走廊","url":"rtsp://camera-b/live"}],"options":{"confidence":0.35,"reid_threshold":0.30,"batch_size":4}}'
```

错误状态：400 参数/空文件错误，403 跨站或缺少请求头，404 资源不存在，409 任务冲突，413 上传或参数过大，415 文件扩展名不支持，422 参数验证失败，507 磁盘空间不足。模型或视频解码错误会写入任务的 `failed` 状态，需查询任务结果，而不是只看启动接口的 HTTP 202。

### 参数和部署配置

网页参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `confidence` | 0.35 | 检测置信度，范围 0.05–1 |
| `reid_threshold` | 0.30 | 归一化特征余弦距离阈值，越小越保守 |
| `batch_size` | 4 | ReID 批大小，API 支持 1–32，低内存时调小 |
| `min_reid_frames` | 10 | 离线轨迹融合最少有效特征帧，范围 1–64；不影响在线模式 |

其他在线多帧确认与质量控制沿用 `GlobalIDManager` 和 `CameraTracker` 默认设置。Web 离线跟踪器最大失联帧数为 100，初始化确认帧数为 3。

服务器环境变量：

| 变量 | 默认值 / 用途 |
| --- | --- |
| `MTMC_DATA_DIR` | 项目下 `web_data`，存放上传、任务、中间记录与结果 |
| `MTMC_MAX_UPLOAD_MB` | 2048，单文件大小上限，按 MiB 计算 |
| `MTMC_CPU_THREADS` | 2，PyTorch CPU 推理线程数 |
| `MTMC_DETECTOR` | 项目下 `yolo11l.pt`，可信的本地 YOLO 权重 |
| `MTMC_TRANSREID_REPO` | 项目下 `external/transreid/repo` |
| `MTMC_TRANSREID_WEIGHTS` | 项目下 `external/transreid/weights/vit_transreid_msmt.pth` |
| `MTMC_ALLOWED_HOSTS` | 默认允许本机主机名；内网访问时填写允许的 IP/域名，逗号分隔，不含端口 |

需要内网访问时，管理员可配置允许的主机名并显式监听网络接口：

```bash
MTMC_ALLOWED_HOSTS=127.0.0.1,localhost,192.168.1.20 python serve.py --host 0.0.0.0
```

将示例 IP 换成服务器实际 IP。此服务没有用户登录和权限隔离；流地址可访问服务器能够到达的网络，因此**只应向受信任用户开放**。不要直接暴露到公网。多用户或公网部署需另加认证网关、TLS、上传配额和网络访问限制。

服务默认单 worker，并对数据目录加进程锁；不要使用多 worker 启动共享同一个目录。流地址只在任务内存中使用，不写入任务 JSON，也不会回显到参数验证错误；第三方解码器的原始输出在推理子进程中被屏蔽，以避免泄露流密码。对外错误只返回经过处理的信息。

数据默认保存在本机，没有自动清理或磁盘总配额。上传和 AVI 结果可能占用较多磁盘空间，长期使用应定期备份并由管理员清理已经不需要的任务；先停止相关任务，避免删除正在使用的文件。输出目录已加入 `.gitignore`。

## 4. 原有命令行入口

Web 功能不替代原有脚本。只使用命令行时可安装较少的依赖：

```bash
python -m pip install -r requirements-runtime.txt
```

历史 `requirements.txt` 保留了旧训练库和 TensorFlow 时代的版本约束，不建议用于当前推理环境。

### 离线视频

```bash
python demo.py --videos videos/init/camera1.mp4 videos/init/camera2.mp4 \
  --detector yolo11l.pt --reid-backend transreid \
  --encoder-batch-size 4 --reid-batch-size 4 \
  --reid-threshold 0.30 --reid-margin 0.05 --reid-gallery-size 32
```

默认输出到 `videos/output/`：

- `tracking.avi`：融合前的数字轨迹 ID。
- `Complete.avi`：融合后的全局 ID。
- `tracking_side_by_side.avi`、`Complete_side_by_side.avi`：多路输入的并排展示。
- `tracking.txt`：旧版八列记录。
- `id_mapping.json`：摄像头、本地轨迹、数字轨迹与全局 ID 映射。

旧版 `tracking.txt` 字段为 `拼接后的帧号(从1开始), tracking_id, x1, y1, x2, y2, 原宽度, 原高度`，**不是 MOTChallenge 标准十列格式**。评测前需要按摄像头拆分并转换格式。映射文件中的每路起止帧从 0 开始。并排视频按第一路帧率展示，不构成严格时间同步。

命令行离线脚本仍保留原有整段视频内存处理方式；长视频、多文件优先使用 Web 流式离线流程。Web 和命令行复用检测、跟踪与融合模块，但采样和输出组织不同，不能假定结果逐位一致。

### 在线视频流

```bash
python demo_stream.py --streams rtsp://camera-a/live rtsp://camera-b/live \
  --display false --output videos/output/live.avi \
  --track-log videos/output/live.jsonl
```

也可用本地文件测试在线算法。需要完整逐帧覆盖时使用有界队列：

```bash
python demo_stream.py --streams videos/init/camera1.mp4 videos/init/camera2.mp4 \
  --stream-mode queue --stream-queue-size 30 --display false \
  --output videos/output/test.avi --track-log videos/output/test.jsonl
```

`latest` 为最新帧策略，`queue` 按解码顺序处理。低延迟 FFmpeg 的 `nobuffer` 选项只对全网络流输入生效，避免本地文件丢帧。命令行合成录像固定为 25 FPS，不等于实际推理速度。

在线日志包含 `camera_id, frame, timestamp, local_id, global_id, bbox, feature_quality`。帧号是拉流线程的序号；时间戳为本机读取时间，不是同步摄像头硬件时间。未确认的全局 ID 为 `null`。Web 离线日志帧号从 0 开始，时间戳取该路解码帧的 PTS 并相对首帧归零，缺失 PTS 时才用帧率推算；在线/离线时间语义不同。

### 可选模型后端

命令行还支持 Torchreid、自定义 ReID 和 torchvision 检测器；Web 页面目前固定使用 YOLO + TransReID。

```bash
python demo.py --videos videos/init/camera1.mp4 \
  --reid-backend torchreid --reid-weights model_data/models/model.pth

python demo.py --videos videos/init/camera1.mp4 \
  --detector-backend torchvision --detector fasterrcnn_resnet50_fpn

python demo.py --videos videos/init/camera1.mp4 \
  --reid-backend custom --custom-reid-name my_reid_model
```

Torchreid 训练库可能额外需要 TensorBoard、h5py、imageio 等依赖。自定义模型目录格式见 [自定义 ReID 说明](custom_reid_models/README.md)。只加载来自可信来源的权重和 Python 适配器。

## 5. 身份关联与准确率控制

### 共同逻辑

- 修正 RGB/BGR 通道约定，保留检测置信度。
- 默认沿用检测器内部 NMS，不重复删除重叠框。
- 可用 `--post-nms nms --nms-max-overlap 0.4` 做额外的按分数排序 IoU NMS 对照。
- 零向量、NaN、Inf 等无效外观特征不进入特征库，由运动/IoU 匹配兜底。
- 不同输入摄像头采用独立本地跟踪器；内部轨迹键为 `(camera_id, local_id)`。

### 离线融合

融合前检查整个同摄像头轨迹时间区间，与已有分组中所有成员都不能发生时间冲突；要求最佳与次佳匹配距离具有足够差距，合并后更新有界、质量加权的代表特征库。短轨迹和无有效特征的轨迹保留独立身份，不会被静默丢弃。不同摄像头可同时看到同一个人。

### 在线确认与复核

| 命令行参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--global-confirm-windows` | 3 | 复用现有 GID 前，需要连续多个不重叠特征窗口确认 |
| `--global-feature-aggregate-frames` | 5 | 每个质量加权特征窗口的有效观察数 |
| `--global-revalidate-threshold` | 0.4 | 外观不一致时暂停特征库更新并隐藏 GID |
| `--global-revalidate-windows` | 3 | 连续不一致后重新分配身份 |
| `--global-pending-timeout` | 10 秒 | 释放没有新增有效证据的停滞候选 |
| `--same-camera-reconnect-timeout` | 5 秒 | 位置重连候选的最大时间间隔 |
| `--global-feature-max-occlusion` | 0.6 | 遮挡过高时跳过 ReID 入库，但继续保留跟踪框 |

新身份仍需满足 `--global-delay` 和 `--global-min-features`。位置重连也必须接受外观确认，不能直接依据位置复用身份。同一摄像头同一帧内对 GID 做排他占用，避免两个人使用同一全局 ID；不同摄像头可共享身份。

TransReID 推理固定使用 `cam_label=0`，不会把新视频的摄像头编号误当成 MSMT17 训练摄像头编号。未假定摄像头拓扑或跨镜行走时间限制。

余弦阈值变小通常会减少错误合并，但也可能造成身份碎片化。离线 `--reid-distance euclidean` 与在线 `--global-distance euclidean` 都使用归一化向量的普通欧氏距离，不是旧版离线的平方欧氏距离；切换距离度量时必须重新标定全部相关阈值。

## 6. 测试与评估

```bash
python -m unittest discover -s tests -v
python -m compileall -q web serve.py demo.py demo_stream.py deep_sort
```

测试覆盖重叠检测、无效特征、独立摄像头跟踪、多帧确认、错误重连、身份漂移、低帧率候选、离线多视频融合、实际文件解码，以及 Web 上传、状态、取消、并发限制、持久化、预览与下载。媒体回归还覆盖 90 帧/30 FPS 完整编码、可变时间戳、奇数分辨率、截断帧数检查、有界队列溢出统计、慢推理不阻塞原画发布、HTTP Range，以及慢客户端的 WebSocket 顺序回放、断开和同源校验。自动测试使用确定性检测/特征输出，不需要下载权重。

真实模型的流程验证与场景准确率评测是两件事。回归测试通过不代表在现场视频上的准确率已经提高某个百分比。建议建立包含遮挡、相似衣着、远距离行人和跨摄像头移动的标注集，固定输入并逐项比较漏检、ID 切换、错误合并和漏关联。

## 7. 主要目录

```text
serve.py                 # Web 启动入口
web/
  app.py                 # HTTP API、同源防护、上传、静态页面、回放
  schemas.py             # 参数校验
  jobs.py                # 独立进程任务调度与磁盘持久化
  worker.py              # 模型加载、状态报告、预览编码
  pipeline.py            # 在线与流式离线推理流程
  media.py               # 保留时间戳的 MP4、独立原画编码与有界播放缓存
  static/                # 嵌入式中文页面
demo.py                  # 原有离线命令行入口
demo_stream.py           # 在线命令行入口、流读取、单摄像头跟踪
torch_detector.py        # YOLO / torchvision 检测适配
global_identity.py       # 在线全局身份管理
offline_association.py   # 离线轨迹融合
reid_backends/           # ReID 后端注册与适配
transreid_adapter.py     # TransReID 源码、权重与推理适配
deep_sort/               # 单摄像头多目标跟踪
tests/                   # 自动回归测试
web_data/                # 运行时数据，不提交版本控制
```

## 致谢与参考

项目基于多摄像头行人追踪与 ReID 相关开源工作演进，感谢以下项目：

- [原始多摄像头追踪项目](https://github.com/samihormi/Multi-Camera-Person-Tracking-and-Re-Identification)
- [Deep SORT](https://github.com/nwojke/deep_sort) 与 [余弦度量学习](https://github.com/nwojke/cosine_metric_learning)
- [Deep Person ReID](https://github.com/KaiyangZhou/deep-person-reid)
- [TransReID](https://github.com/damo-cv/TransReID)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [Deep SORT YOLOv3](https://github.com/Qidian213/deep_sort_yolov3)
- [Keras YOLO4](https://github.com/Ma-Dan/keras-yolo4)
- [Human Tracking Multicam](https://github.com/lyrgwlr/Human-tracking-multicam)
- [FastAPI 静态文件文档](https://fastapi.tiangolo.com/tutorial/static-files/) 与 [服务生命周期文档](https://fastapi.tiangolo.com/advanced/events/)

部署或商用前，请分别核对各模型、权重、代码和训练数据的许可证与使用条件。
