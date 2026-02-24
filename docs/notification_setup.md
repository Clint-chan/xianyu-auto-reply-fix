# 通知渠道配置指南

## 问题说明

如果你看到日志中出现以下警告：
```
📱 账号 xxx 未配置消息通知，跳过通知发送
```

说明你的账号还没有配置通知渠道。

## 配置方法

### 方法1：通过Web界面配置（推荐）

1. **登录系统**
   - 访问 `http://localhost:8090`（或你的服务器地址）
   - 使用你的账号登录

2. **进入账号管理**
   - 点击左侧菜单的"账号管理"
   - 找到你要配置通知的账号

3. **配置通知渠道**
   - 点击账号右侧的"通知设置"按钮
   - 选择通知类型（QQ、钉钉、飞书、Bark、邮件等）
   - 填写对应的配置参数
   - 点击"保存"

### 方法2：直接操作数据库

如果Web界面无法访问，可以直接操作数据库：

```sql
-- 查看当前账号的通知配置
SELECT * FROM account_notifications WHERE cookie_id = '你的账号ID';

-- 添加QQ通知（示例）
INSERT INTO account_notifications (cookie_id, notification_type, config, enabled, created_at)
VALUES (
    '你的账号ID',
    'qq',
    '{"qq_number": "你的QQ号", "api_url": "http://your-cqhttp-server:5700"}',
    1,
    datetime('now')
);

-- 添加钉钉通知（示例）
INSERT INTO account_notifications (cookie_id, notification_type, config, enabled, created_at)
VALUES (
    '你的账号ID',
    'dingtalk',
    '{"webhook": "https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN"}',
    1,
    datetime('now')
);
```

## 支持的通知类型

### 1. QQ通知
需要配置：
- `qq_number`: 接收通知的QQ号
- `api_url`: go-cqhttp服务器地址

### 2. 钉钉通知
需要配置：
- `webhook`: 钉钉机器人Webhook地址

### 3. 飞书通知
需要配置：
- `webhook`: 飞书机器人Webhook地址

### 4. Bark通知（iOS）
需要配置：
- `device_key`: Bark设备密钥
- `server_url`: Bark服务器地址（可选，默认官方服务器）

### 5. 邮件通知
需要配置：
- `smtp_server`: SMTP服务器地址
- `smtp_port`: SMTP端口
- `sender_email`: 发件人邮箱
- `sender_password`: 发件人密码
- `receiver_email`: 收件人邮箱

### 6. 企业微信通知
需要配置：
- `webhook`: 企业微信机器人Webhook地址

### 7. Telegram通知
需要配置：
- `bot_token`: Telegram Bot Token
- `chat_id`: 接收消息的Chat ID

### 8. Server酱通知
需要配置：
- `send_key`: Server酱SendKey

## 通知冷却时间说明

为了避免频繁通知，系统设置了通知冷却时间：

- **普通消息通知**：5分钟内相同消息不重复发送
- **Token刷新通知**：3小时内不重复发送

这些参数在代码中定义：
```python
self.notification_cooldown = 300  # 5分钟
self.token_refresh_notification_cooldown = 18000  # 3小时
```

## 测试通知

配置完成后，可以通过以下方式测试：

1. **发送测试消息**
   - 在闲鱼上给你的账号发送一条消息
   - 检查是否收到通知

2. **查看日志**
   ```bash
   # 查看通知发送日志
   tail -f logs/xianyu_*.log | grep "通知"
   ```

3. **检查配置**
   ```bash
   # 查看数据库中的通知配置
   sqlite3 data/xianyu_data.db "SELECT * FROM account_notifications WHERE cookie_id = '你的账号ID';"
   ```

## 常见问题

### Q1: 配置了通知但没有收到
**检查项**：
1. 通知配置是否启用（`enabled = 1`）
2. 通知参数是否正确（Webhook地址、Token等）
3. 是否在冷却期内（查看日志）
4. 网络是否可达（防火墙、代理等）

### Q2: 如何修改冷却时间
修改 `XianyuAutoAsync.py` 中的参数：
```python
# 第755行
self.notification_cooldown = 300  # 改为你想要的秒数

# 第756行
self.token_refresh_notification_cooldown = 18000  # 改为你想要的秒数
```

### Q3: 如何禁用某个账号的通知
```sql
UPDATE account_notifications 
SET enabled = 0 
WHERE cookie_id = '你的账号ID';
```

### Q4: 如何删除通知配置
```sql
DELETE FROM account_notifications 
WHERE cookie_id = '你的账号ID' 
AND notification_type = '通知类型';
```

## 重启程序

配置完成后，**需要重启程序**才能生效：

```bash
# Docker部署
docker restart xianyu-auto-reply

# 本地部署
# 停止程序（Ctrl+C）
# 重新启动
python Start.py
```

## 相关代码位置

- 通知发送逻辑：`XianyuAutoAsync.py` 第4310-4360行
- 通知配置获取：`db_manager.py` 中的 `get_account_notifications` 方法
- 通知冷却时间：`XianyuAutoAsync.py` 第755-756行
