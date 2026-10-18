# compshare-agent

优云算力共享 AI 助手（CompShare GPU 平台）。Go 1.25 单二进制，`server` 子命令提供 HTTP/WS 服务。

## 配置

配置分为 `deploy/conf/config.local.yaml` 和 `deploy/conf/config.prod.yaml`。前者保存共享运行基线（授权、模型、数据库、STS 与可观测性配置）；后者通过 `extends` 继承基线，只保留生产网络覆盖项，不再使用 `.env` 或 `*.example` 模板。

本地默认启动路径是 `deploy/conf/config.local.yaml`：

```bash
go build -o compshare-agent ./cmd
./compshare-agent server
```

需要覆盖配置路径时：

```bash
./compshare-agent server -c /path/to/config.yaml
```

## 公开技术资料检索

控制台助手可通过 [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
搜索当前的 GPU、AI 部署和 Linux 技术资料，并获取来源链接与节选。公开网页检索默认可用，
使用匿名 Fast 模式，不需要 Parallel 账号或 API key；免费服务有使用额度。仅发送公开技术问题，
不要将账号信息、实例 ID、密码或私有日志放入检索内容。

部署可在 YAML 中关闭公开网页检索或调整单次请求超时（默认 20000 毫秒）：

```yaml
agent:
  web_search:
    disabled: true
    timeout_ms: 20000
```

`SearchKnowledge` 和 `ReadChunk` 仍只访问配置的 CompShare 知识库。
网页结果是外部参考，不证明平台规则、账号或实例状态；知识问答专用模式和外部群的工具权限保持各自的限制。

## 部署

生产交付物是一个自包含 Docker 镜像，内含 Go 服务、`config.local.yaml` 与 `config.prod.yaml`、Python 3.13、Claude Agent SDK、
Claude CLI 和 SSH-ops harness；生产知识检索通过 `config.prod.yaml` 中的
`agent.retrieval.mcp_url` 访问独立的 `compshare-kb` 服务。生产配置会进入镜像层，私有仓库的 pull 权限
按生产凭据权限管理。

GitLab 持续交付遵循 `compshare-kb` 的三阶段约定：`main` 分支先执行测试和静态检查，再由
Kaniko 推送提交短 SHA 与 `latest` 两个镜像标签；生产部署必须在 GitLab 中手动确认。Kubernetes
清单、部署前置条件和所需 CI 变量见 [`deploy/k8s/README.md`](deploy/k8s/README.md)。

## 本地数据库

后端存储使用 PostgreSQL。新库和升级都执行同一条命令，迁移可重复执行，已应用过的会跳过：

```bash
for f in deploy/migrations/*.sql; do
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 < "$f"
done
```

`deploy/conf/config.local.yaml` 中的 `agent.mysql.dsn` 需要填 PostgreSQL libpq URL，例如：

```text
postgresql://user:pass@host:5432/compshare_agent?sslmode=disable
```

## 实例内排查（SSH-ops）

生产开启写能力后，由模型根据完整对话选择实例 ID；服务器用当前租户 STS 点查该 ID，并将凭据、审计与执行范围固定到同一实例，不再二次解析自然语言来判定目标。助手随后直接 SSH 进入实例定位问题，并自主完成范围内的
可恢复修复与验证，不再弹入口卡或逐命令确认；删除数据、格式化、重启关机、改账号密码、关 SSH/网络，以及跨租户或控制面操作仍一律拒绝。

**比其他功能多一套部署工作**：诊断循环跑在 Claude Agent SDK 的独立子进程里，不在本二进制内，
所以机器上还要有一个装了 `deploy/ssh_ops_harness/requirements.txt` 的 Python 环境、`claude` CLI，
以及一条数据库迁移。Claude CLI 通过 Anthropic 协议直连 ModelVerse，不需要额外路由器。

部署步骤与排错见 [`deploy/ssh_ops_harness/README.md`](deploy/ssh_ops_harness/README.md)。

## 测试

```bash
go test ./... -count=1
```
