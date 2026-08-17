# TG 防撞客机器人 V1.9 SSCD-AI COPY-GUARD

GitHub + Railway 群聊专用版本。私聊关闭。

## 识别目标

**一张照片 = 一个客户。** 系统判断的是“是否为同一张照片的编辑/截图/裁剪/压缩版本”，不是判断两张不同照片里的人是否为同一个人。

支持的主要变化：原图重存、Telegram 压缩、缩放、裁剪、放大、黑白边、圆形头像、轻微旋转、左右镜像、亮度/滤镜变化、头像出现在资料页截图中、部分模糊等。

## V1.9 AI

V1.9 新增 SSCD Copy Detection AI 作为主同图 AI，原来的 MobileNet 只在 SSCD 不可用时作为备用。SSCD 候选与 pHash/局部特征候选取并集，再用 SIFT/AKAZE/RANSAC 等做配对验证。

首次部署会下载约 99MB 的 SSCD TorchScript 模型到 `/data/models`；Railway Volume 挂载 `/data` 后只需下载一次。

## 判定规则

- 完全同文件：100%
- >=90% 且有强同图证据：自动撞客
- 84%~89.99%：疑似撞客，人工确认/误判
- 明显不同：新客户

低信息图片会采用更保守的自动判定，避免不同纯色/简单默认头像因 pHash 相同而误撞。

## Railway 必须设置

Variables：

```text
BOT_TOKEN=你的机器人Token
DATA_DIR=/data
```

给 worker 挂载 Railway Volume：

```text
/data
```

V1.9 默认在 Railway 上强制检查 Volume。没有 Volume 时会拒绝启动，不会静默使用临时空数据库。

## AI 索引

升级后旧客户图片会在后台逐批生成 SSCD 描述符。群里发送：

```text
/aistatus
```

可以查看 AI 模型和历史图库索引进度。

## 隐私模式

BotFather -> `/setprivacy` -> 选择机器人 -> `Disable`，并建议把机器人设为正式群管理员。
