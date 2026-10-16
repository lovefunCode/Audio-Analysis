# 音频分析 - 演讲陪练助手（SUT）

输入演讲音频（mp3/m4a）和用户指令，按指定时间段做四维度演讲诊断，以 SSE 流式返回改进建议。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/ready` | 探活，模型加载完成后返回 200（加载中返回 503） |
| POST | `/generate_advice` | multipart：`item_id`、`query`、`audio`，返回 `text/event-stream` |

SSE 事件顺序：`advice_start` → 多个 `advice`（`append` 依次拼接即完整文本）→ `done`；出现异常时发送 `error`。
超时长、无有效语音、指令违规等情况也走 `advice` 输出纯文本提示，再发 `done`，不会返回空结果。

## 处理流程

```
query ─► 安全规则 ─► 意图解析（时间段 / 关注点）
audio ─► 落盘 ─► ffprobe 时长 ─► 超限/越界提示
       ─► ffmpeg 截取片段 16k ─► Paraformer + VAD + 标点（句级时间戳）
       ─► 转写安全复核（规则 + LLM）
       ─► 声学统计（语速/分段语速/停顿/音量） + 文本统计（口头禅/句长/结构词/听众词）
       ─► Qwen2.5-14B（vLLM AsyncEngine，prefix cache）逐字段续写，逐 token 流式
```

格式由服务端保证：总述、四个章节标题、`• 优点：`、`• 待优化点：`、`N• ` 都是程序写入的固定文本，模型只续写每行内容（遇到换行即停），因此结构不会缺项、乱序。

## 目录

```
app/
  config.py               环境变量配置
  api/server.py           FastAPI：/ready、/generate_advice（SSE）
  api/pipeline.py         流程编排与各类提示文案
  analysis/intent.py      时间段与关注点解析
  analysis/safety.py      安全规则与 LLM 审核 prompt
  analysis/report.py      分析 prompt 与结构受控的流式生成
  analysis/formatter.py   格式校验（测试/监控用）
  analysis/text_stats.py  文本统计
  audio/loader.py         音频落盘、时长探测、片段解码
  audio/features.py       声学特征
  asr/engine.py           FunASR 封装
  llm/engine.py           vLLM 异步引擎封装
tests/test_units.py       无 GPU 单测（格式、意图、安全、SSE 协议）
smoke_test.py             端到端冒烟（打印首字时延/总耗时/格式校验）
```

## 部署

```bash
bash download_models.sh                       # 模型下载到 ./models
docker build -t speech-coach .
docker run --gpus all --restart=always -p 80:80 -v $(pwd)/models:/workspace/models speech-coach
curl localhost/ready
python smoke_test.py case.mp3 "帮我听5-10分钟，指出表达节奏问题"
python tests/test_units.py
```

## 关键配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_MODEL_PATH` | `/workspace/models/qwen2.5-14b-instruct` | LLM 目录 |
| `ASR_MODEL_PATH` | `/workspace/models/paraformer-zh` | ASR 目录（含 `vad/`、`punc/`） |
| `LLM_GPU_MEMORY_UTILIZATION` | 0.60 | vLLM 显存占比，其余留给 ASR |
| `MAX_AUDIO_DURATION_S` | 3600 | 单次分析上限，另有 30 秒容差 |
| `SAFETY_LLM_CHECK` | true | 是否对转写文本做 LLM 安全复核 |
| `EARLY_FIRST_TOKEN` | false | true 时在 ASR 前先输出开头总述，降低首字时延，但与"转写完成后发 advice_start"的约定不一致 |
| `READY_BEFORE_WARMUP` | false | true 时模型加载期间 `/ready` 也返回 200，请求会排队等加载完成；评测探活超时很短时可开启 |
| `ASR_TIMEOUT_S` | 900 | 单次 ASR 上限，超时输出提示文本 |
| `LLM_FIRST_TOKEN_TIMEOUT_S` / `LLM_TOKEN_TIMEOUT_S` | 120 / 30 | 单字段首 token / 相邻 token 超时，超时该字段走兜底文本 |
| `REQUEST_SOFT_DEADLINE_S` | 1500 | 请求总耗时软上限，超过后剩余字段直接用兜底文本 |
| `EXIT_ON_ENGINE_DEAD` | true | vLLM 引擎死亡（CUDA 错误）时退出进程，交给容器重启 |

## 稳定性设计（流程任一环节断掉时的行为）

| 环节 | 失败方式 | 处理 |
|---|---|---|
| 安全 / 意图 | 纯正则，不会失败 | 拦截→固定提示文本；解析不到时间段→按整段分析 |
| 时长校验 | ffprobe 失败返回 0 | 先解码，再按实际解码时长补查上限 |
| 解码 | ffmpeg 缺失 / 损坏 / 超时 | 回退 librosa；仍失败→提示文本 |
| ASR | 超时 / 异常 / 无语音 | `ASR_TIMEOUT_S` 超时提示；异常提示；字数不足提示 |
| 统计 | 任意异常 | 该项标记"不可用"，报告照常生成 |
| 安全复核 | LLM 超时 / 异常 | 跳过 LLM 复核（规则层已过），继续 |
| 报告 | 某字段 LLM 失败 / 超时 / 超软截止 | 该字段兜底文本，结构保持完整；连续 3 次失败后余下字段全部兜底 |
| 服务层 | 未预期异常 | 以 `advice` 输出提示文本并 `done`；引擎死亡→退出进程由容器重启 |

所有用户可见失败都走 `advice` 文本 + `done`，只有"一条文本都没产出"才发 `error`。

## 启动行为

模型在后台线程加载，HTTP 端口立即监听：加载中 `/ready` 返回 503，完成后 200；加载失败返回 503 并带 `message`。
加载期间到达的 `/generate_advice` 请求会等待加载完成后再处理，不会直接报错。
