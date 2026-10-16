# speech_assistant

音频分析-演讲伴练助手榜

## 项目结构

```
speech_assistant/
├── starting-kit/
│   ├── main.py             # 模拟推理服务：POST /generate_advice 流式接口（multipart 接收音频+query，SSE 返回改进建议）
│   ├── API.md              # starting-kit 接口文档
│   ├── requirements.txt    # starting-kit 依赖（Flask）
│   ├── Dockerfile          # starting-kit 镜像构建
│   ├── start.sh            # starting-kit 容器启动脚本
│   └── build.sh            # starting-kit 镜像构建推送脚本
├── run_for_darvin.py       # Judge Flow：读数据集 → 循环调用流式接口 → 组装 JSON → 记录指标 → 推送标注平台
├── run_for_score.py        # 评分脚本：拉取人工标注结果 → 最终分数计算
├── utils/                  # logger / file / labelx / leaderboard / request 工具
├── dataset/dataset.zip     # 数据集（data.json + audio/*.mp3）
├── Dockerfile / start.sh / build.sh   # Judge Flow 镜像
└── requirements.txt
```

## 数据集格式

`data.json`：
```json
[
  {"id": 1, "query": "帮我听完整音频，给我演讲整体改进建议，提升演讲吸引力", "file_path": "audio/case1-01-12.mp3"}
]
```

## 流式接口协议（POST /generate_advice）

请求：`multipart/form-data`，字段 `item_id`、`query`、`audio`（音频文件二进制）。

响应：SSE（`text/event-stream`）：
- `event: advice_start` / `data: {"item_id": ..., "query": ...}`
- `event: advice` / `data: {"append": "<增量建议文本>"}`  ← 逐块累积
- `event: done` / `data: {"item_id": ...}`
- `event: error` / `data: {"message": ...}`

流式数据最终组装为一个 JSON：
```json
{"item_id": "1", "query": "...", "advice": "完整改进建议文本"}
```

## 评测指标（run_for_darvin 输出 metrics.json）

- `avg_first_token_latency`：每个 case 的首字延迟（请求发出 → 第一块建议文本）平均（秒）
- `avg_generation_time`：每个 case 的生成时间平均
- `failure_rate`：失败率，超过阈值（默认 10%）终止程序，不推送标注平台

## 评分逻辑（run_for_score，最终指标 auto_scoring_metrics.json）

- case 有效性：`overall_evaluation` 有且仅有 `["以上问题均未出现"]`，否则该 case 直接 0 分
- 有效 case 分数 = `model_understand + model_execute_query + out_four_segment + four_sagement_contains_good + end_suggest + content_score + language_expression + improve_direction + audio_content + audio_segment + content_compliance`
- `case平均分` = 所有 case 总分 / 数据集中的数据量（metrics.json 的 `total_item_count`）
- `失败率` = （数据集中 case 数 - 人工评分大于 0 的 case 数）/ 数据集中 case 数
- `平均首字延迟(ms)`：取 metrics.json 的 `avg_first_token_latency`（秒）换算为毫秒

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `DARVIN_INFERENCE_ADDRESS` | `http://localhost:8089` | starting-kit 服务地址 |
| `DARVIN_TESTING_DATASET_ZIP_FILE` | `./dataset/dataset.zip` | 数据集 ZIP |
| `DARVIN_EVALUATION_OUTPUT_DIR` | `./evaluation_results` | 输出目录 |
| `FAILURE_RATE_THRESHOLD` | `0.10` | 失败率阈值 |
| `FIRST_TOKEN_TIMEOUT` | `60` | 单 case 首字延迟上限（秒） |
| `REQUEST_TIMEOUT` | `1200` | 流式请求超时（秒） |
| `DATASET_LIMIT` | `0`（不限） | 截取前 N 条调试 |

## 运行

```bash
# 1. 启动模拟服务（真实场景由选手提供 starting-kit）
python starting-kit/main.py   # 默认端口 80（MOCK_SERVER_PORT 可覆盖）

# 2. Judge Flow：跑数据集并推送标注
python run_for_darvin.py

# 3. 人工标注完成后拉取结果算分
python run_for_score.py
```
