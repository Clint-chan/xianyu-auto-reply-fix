"""
AI回复引擎模块 - 统一意图识别与回复生成

【重构版本】
- 将意图判断和回复生成合并为一次AI调用
- AI根据完整上下文自行判断意图并生成回复
- 避免关键词误判导致的不当回复
"""

import json
import time
import requests
import threading
from typing import List, Dict, Optional
from loguru import logger
from openai import OpenAI
from db_manager import db_manager


class AIReplyEngine:
    """AI回复引擎 - 统一意图识别与回复生成"""
    
    def __init__(self):
        self._init_default_prompts()
        # 用于控制同一chat_id消息的串行处理
        self._chat_locks = {}
        self._chat_locks_lock = threading.Lock()
        # 修复旧格式迁移预设（补充完整角色模板）
        self._fix_migrated_presets()
    
    def _init_default_prompts(self):
        """初始化默认提示词（用于构建统一提示词）"""
        self.default_prompts = {
            'price': '''【议价场景】
策略：不接受议价，但用销冠思维制造紧迫感促成下单。
- 友好但坚定：已经是官方1/10的价格，全网最低档位了
- 制造稀缺："这个价格最近咨询的人特别多，这批额度快分完了，建议趁现在入手"
- 如果犹豫：推荐10元=128刀尝鲜体验，"先拍个小的体验一下，好用再囤"
- 如果还在砍价："真没利润空间了，我们走量的，您看中的话直接拍就行，自动发货秒到"''',

            'tech': '''【技术/产品问题】
基于知识库回答，遇到不确定的问题不要编造。
回答技术问题时顺带强调优势（稳定性、速度、性价比）。
如果问题超出知识库范围：引导加V深度交流，"这个加V：greatluck2023聊，我给您详细演示"''',

            'default': '''【一般咨询】
基于知识库回答发货、配置等问题，同时寻找机会推动下单。
如果客户询问退款：先挽留（"额度还没到期呢，趁有效期内用完更划算哦，我们性价比真的超高"），再说"实在需要退款的话，请您先提交退货申请，稍后帮您办理"
如果遇到复杂售后问题（报错、异常等），引导加V：greatluck2023'''
        }
    
    def _create_openai_client(self, cookie_id: str) -> Optional[OpenAI]:
        """创建指定账号的OpenAI客户端（无状态）"""
        settings = db_manager.get_ai_reply_settings(cookie_id)
        if not settings['ai_enabled'] or not settings['api_key']:
            return None

        try:
            base_url = settings['base_url'].rstrip('/')
            # 确保 base_url 以 /v1 结尾（OpenAI SDK 要求）
            if not base_url.endswith('/v1'):
                base_url = base_url + '/v1'
            logger.info(f"创建OpenAI客户端: base_url={base_url}")
            client = OpenAI(
                api_key=settings['api_key'],
                base_url=base_url
            )
            return client
        except Exception as e:
            logger.error(f"创建OpenAI客户端失败 {cookie_id}: {e}")
            return None

    def _is_dashscope_api(self, settings: dict) -> bool:
        """判断是否为DashScope API"""
        model_name = settings.get('model_name', '')
        base_url = settings.get('base_url', '')
        is_custom_model = model_name.lower() in ['custom', '自定义', 'dashscope', 'qwen-custom']
        is_dashscope_url = 'dashscope.aliyuncs.com' in base_url
        return is_custom_model and is_dashscope_url

    def _is_gemini_api(self, settings: dict) -> bool:
        """判断是否为Gemini API"""
        model_name = settings.get('model_name', '').lower()
        return 'gemini' in model_name
    
    def _is_anthropic_api(self, settings: dict) -> bool:
        """判断是否为Anthropic API（模型名含claude或配置了anthropic_api_key）"""
        model_name = settings.get('model_name', '').lower()
        anthropic_key = settings.get('anthropic_api_key', '')
        return 'claude' in model_name or bool(anthropic_key and anthropic_key.strip())
    
    def _call_anthropic_api(self, settings: dict, messages: list, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """
        调用Anthropic Claude API
        """
        api_key = settings.get('anthropic_api_key') or settings.get('api_key')
        model_name = settings.get('model_name', 'claude-3-5-sonnet-20241022')

        # 确定base_url：优先使用anthropic_base_url，否则回退到通用base_url
        anthropic_base = settings.get('anthropic_base_url', '')
        if anthropic_base and anthropic_base.strip() and anthropic_base != 'https://api.anthropic.com':
            base_url = anthropic_base.rstrip('/')
        elif settings.get('base_url', ''):
            # 使用通用base_url，去掉/v1后缀（Anthropic API自己拼接/v1/messages）
            base_url = settings['base_url'].rstrip('/')
            if base_url.endswith('/v1'):
                base_url = base_url[:-3].rstrip('/')
        else:
            base_url = 'https://api.anthropic.com'

        url = f"{base_url}/v1/messages"
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # 转换消息格式
        system_content = ""
        anthropic_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_content = msg['content']
            else:
                anthropic_messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        
        payload = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages
        }
        
        if system_content:
            payload["system"] = system_content
        
        logger.info(f"Calling Anthropic API: {url}")
        logger.debug(f"Anthropic Payload: {json.dumps(payload, ensure_ascii=False)}")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            logger.error(f"Anthropic API 请求失败: {response.status_code} - {response.text}")
            raise Exception(f"Anthropic API 请求失败: {response.status_code} - {response.text}")
        
        result = response.json()
        logger.debug(f"Anthropic API 响应: {json.dumps(result, ensure_ascii=False)}")
        
        try:
            reply_text = result['content'][0]['text']
            return reply_text.strip()
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Anthropic API 响应格式错误: {result} - {e}")
            raise Exception(f"Anthropic API 响应格式错误: {result}")
    def get_default_system_prompt(self) -> str:
        """返回内置的默认系统提示词，供前端新建预设时预填充"""
        return self._build_unified_system_prompt(self.default_prompts, {})

    def _fix_migrated_presets(self) -> None:
        """修复从旧custom_prompts迁移过来的不完整预设，补充完整的角色模板"""
        try:
            full_template = self.get_default_system_prompt()
            # 识别标志：完整模板必然包含"销冠核心法则"；旧迁移预设只有场景片段
            db_manager.fix_migrated_presets(full_template)
        except Exception as e:
            logger.warning(f"修复迁移预设失败（不影响启动）: {e}")

    def _build_unified_system_prompt(self, custom_prompts: dict, settings: dict) -> str:
        """
        构建统一的系统提示词
        将意图判断和回复生成整合到一个提示词中
        """
        # 获取各场景的指导（优先使用用户自定义）
        price_guide = custom_prompts.get('price', self.default_prompts['price'])
        tech_guide = custom_prompts.get('tech', self.default_prompts['tech'])
        default_guide = custom_prompts.get('default', self.default_prompts['default'])
        knowledge_base = custom_prompts.get('knowledge_base', '')

        unified_prompt = f"""你是一位顶尖销冠AI，负责在闲鱼平台上促成API额度的成交。你不仅仅是客服——你懂得销售心理学，善于制造稀缺感和紧迫感，同时真诚地帮客户做出最优选择。请根据用户消息、知识库和上下文，直接生成合适的回复。

## 销冠核心法则
1. **每句话都朝成交推进**：回答问题的同时，自然植入下单引导（"拍下秒发""现在入手正好"）
2. **制造稀缺和紧迫**：适时暗示库存/名额有限（"最近咨询量特别大""这批快分完了""今天已经出了好几单了"），但不要每句话都说，自然穿插
3. **强调独特优势**：官方1/10价格、国内直连CN2专线不限速、满血4.6模型、自动发货秒到、独立后台查用量
4. **社会认证**：适时提到其他客户的正面反馈（"回头客特别多""很多开发者都在用""老客户都是直接囤1800刀的"），自然不刻意
5. **严格基于知识库**：只回答知识库和商品信息中有的内容，绝不编造
6. **准确理解意图**：只根据用户实际说的内容判断，不过度解读
7. **不主动提及敏感话题**：用户没提到的（如退款、砍价）不要主动提
8. **避免重复**：结合对话历史，不重复之前说过的话
9. **语言简洁自然**：像真人聊天，简短友好，一般不超过30字，不要用markdown格式

## 场景处理指南

### 当用户明确要求降价/优惠/砍价时
{price_guide}

### 当用户询问产品技术/功能/使用问题时
{tech_guide}

### 售后问题处理
- 简单售后（教程在哪、怎么配置）：直接根据知识库回答，给出具体教程链接
- 疑难售后（报错、无法使用、账号异常、退款等复杂问题）：引导客户加V处理，回复类似"这个加V：greatluck2023，我帮你排查处理~"
- 判断标准：如果知识库里有明确答案就直接回答，如果没有或问题比较复杂就引导加V

### 其他一般咨询
{default_guide}

{f'## 产品知识库（回答问题的核心依据）{chr(10)}{knowledge_base}' if knowledge_base else ''}

## 回复示例（学习销冠的语气和节奏）

客户: 怎么收费的
销冠: 10元=128刀先体验，140元=1800刀囤货最划算，官方1/10的价格，最近拍的人挺多的~

客户: 能便宜点吗
销冠: 已经是官方十分之一了真没空间了，好多老客户都是直接囤1800刀的，先拍个128刀体验下？好用再囤~

客户: 支持Claude Code吗
销冠: 完美适配Claude Code，国内直连不限速，很多开发者都在用，拍下秒发~

客户: 怎么发货
销冠: 拍下后自动发API地址和Key，附带保姆级教程，秒到账~

客户: 可以用酒馆吗
销冠: 不支持酒馆哦，主要适配Claude Code、OpenCode、OpenClaw等开发工具

客户: 有后台看消耗吗
销冠: 有的，购买后给您专属后台，余额用量随时可查，用着很透明~

客户: 有网站可以看看吗
销冠: 有独立后台可以随时查余额和用量，使用方式是API地址+Key接入，拍下后自动发给您~

客户: 国内能用吗需要翻墙吗
销冠: 国内直连，走CN2 GIA专线，不需要翻墙，速度很快~

客户: 用不了报错了怎么办
销冠: 加一下V：greatluck2023，发我报错截图帮你看看~

客户: 教程在哪
销冠: Claude Code教程：Linux/Mac版 https://ucnkgui8gbgt.feishu.cn/wiki/AMKLwkOyYikFtrk62TPcbnYbnOe ，Windows版 https://ucnkgui8gbgt.feishu.cn/docx/VpPudlFqlozgadxtj2CcgwAQnkc ，有问题随时问~

客户: 能退款不
销冠: 额度还没到期呢，趁有效期内用完更划算哦。实在需要退款的话，您先提交退货申请，稍后帮您办理~

客户: 是逆向的吗
销冠: 我们走的官方Anthropic协议，满血4.6模型，高稳定性，很多客户用了大半年了非常稳~

客户: token数统计偏大/消耗怎么这么多
销冠: 我们性价比极高，官方1/10的价格，您可以随时在后台查询详细用量明细哦

客户: 有没有1m上下文
销冠: 暂时不支持1m上下文哦，目前是标准的200k上下文

客户: 好的我看看
销冠: 好的随时找我，最近咨询量比较大，想好了直接拍就行，自动发货秒到~

客户: 稳定吗
销冠: 非常稳定，CN2专线直连，很多客户用了大半年了，回头客特别多~

## 特别注意
- 用户只是问价格 ≠ 用户在砍价，正常回答价格并顺带推荐即可
- 用户说"好的""我看看""考虑一下"→ 不要追着推销，轻轻制造紧迫感即可（"想好了随时拍~"）
- 用户咨询简单售后（教程、配置方法）→ 直接根据知识库回答
- 用户遇到疑难问题（报错、异常、退款）→ 引导加V：greatluck2023
- 超出知识库范围的问题 → 引导加V：greatluck2023
- 用户问"是不是逆向"→ 强调走官方协议、满血模型、高稳定性，不要说"我们不是逆向"也不要承认逆向，突出比较优势和客户口碑
- 用户问退款→ 先挽留（强调趁有效期用完更划算），再告知提交退货申请流程
- 套餐是月度有效期，到期额度清零，绝对不要说"额度不会过期""用不完下个月继续"
- 用户问"有网站吗"→ 回答有独立后台查余额，使用方式是API+Key，不要说没有网站
- 制造稀缺感要自然，不要每条消息都说"快没了"，根据对话节奏适时穿插
- 不要输出分析过程，直接输出回复内容"""
        
        return unified_prompt

    def _call_dashscope_api(self, settings: dict, messages: list, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """调用DashScope API"""
        base_url = settings['base_url']
        if '/apps/' in base_url:
            app_id = base_url.split('/apps/')[-1].split('/')[0]
        else:
            raise ValueError("DashScope API URL中未找到app_id")

        url = f"https://dashscope.aliyuncs.com/api/v1/apps/{app_id}/completion"

        system_content = ""
        user_content = ""
        for msg in messages:
            if msg['role'] == 'system':
                system_content = msg['content']
            elif msg['role'] == 'user':
                user_content = msg['content'] # 假设 user prompt 已在 generate_reply 中构建好

        if system_content and user_content:
            prompt = f"{system_content}\n\n用户问题：{user_content}\n\n请直接回答用户的问题："
        elif user_content:
            prompt = user_content
        else:
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        data = {
            "input": {"prompt": prompt},
            "parameters": {"max_tokens": max_tokens, "temperature": temperature},
            "debug": {}
        }
        headers = {
            "Authorization": f"Bearer {settings['api_key']}",
            "Content-Type": "application/json"
        }

        logger.info(f"DashScope API请求: {url}")
        logger.info(f"发送的prompt: {prompt[:100]}...") # 避免 prompt 过长
        logger.debug(f"请求数据: {json.dumps(data, ensure_ascii=False)}")

        response = requests.post(url, headers=headers, json=data, timeout=30)

        if response.status_code != 200:
            logger.error(f"DashScope API请求失败: {response.status_code} - {response.text}")
            raise Exception(f"DashScope API请求失败: {response.status_code} - {response.text}")

        result = response.json()
        logger.debug(f"DashScope API响应: {json.dumps(result, ensure_ascii=False)}")

        if 'output' in result and 'text' in result['output']:
            return result['output']['text'].strip()
        else:
            raise Exception(f"DashScope API响应格式错误: {result}")

    def _call_gemini_api(self, settings: dict, messages: list, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """
        调用Google Gemini REST API (v1beta)
        """
        api_key = settings['api_key']
        model_name = settings['model_name'] 
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

        headers = {"Content-Type": "application/json"}

        # --- 转换消息格式 (修复 P1-3: 增强健壮性) ---
        system_instruction = ""
        user_content_parts = []

        # 遍历消息，找到 system 和所有的 user parts
        for msg in messages:
            if msg['role'] == 'system':
                system_instruction = msg['content']
            elif msg['role'] == 'user':
                # 我们只关心 user content
                user_content_parts.append(msg['content'])
        
        # 将所有 user parts 合并为最后的 user_content
        # 在我们的使用场景中 (generate_reply)，只会有一个 user part，但这样更安全
        user_content = "\n".join(user_content_parts)

        if not user_content:
            logger.warning(f"Gemini API 调用: 未在消息中找到 'user' 角色内容。Messages: {messages}")
            raise ValueError("未在消息中找到用户内容 (user content)")
        # --- 消息格式转换结束 ---

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_content}]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        logger.info(f"Calling Gemini REST API: {url.split('?')[0]}")
        logger.debug(f"Gemini Payload: {json.dumps(payload, ensure_ascii=False)}")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            logger.error(f"Gemini API 请求失败: {response.status_code} - {response.text}")
            raise Exception(f"Gemini API 请求失败: {response.status_code} - {response.text}")
            
        result = response.json()
        logger.debug(f"Gemini API 响应: {json.dumps(result, ensure_ascii=False)}")

        try:
            reply_text = result['candidates'][0]['content']['parts'][0]['text']
            return reply_text.strip()
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Gemini API 响应格式错误: {result} - {e}")
            raise Exception(f"Gemini API 响应格式错误: {result}")

    def _call_openai_api(self, client: OpenAI, settings: dict, messages: list, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """调用OpenAI兼容API"""
        response = client.chat.completions.create(
            model=settings['model_name'],
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    def is_ai_enabled(self, cookie_id: str) -> bool:
        """检查指定账号是否启用AI回复"""
        settings = db_manager.get_ai_reply_settings(cookie_id)
        return settings['ai_enabled']
    
    def _get_chat_lock(self, chat_id: str) -> threading.Lock:
        """获取指定chat_id的锁，如果不存在则创建"""
        with self._chat_locks_lock:
            if chat_id not in self._chat_locks:
                self._chat_locks[chat_id] = threading.Lock()
            return self._chat_locks[chat_id]
    
    def generate_reply(self, message: str, item_info: dict, chat_id: str,
                      cookie_id: str, user_id: str, item_id: str,
                      skip_wait: bool = False) -> Optional[str]:
        """
        生成AI回复 - 统一意图识别与回复生成
        AI会自动判断用户意图并生成合适的回复，避免关键词误判
        """
        if not self.is_ai_enabled(cookie_id):
            return None
        
        try:
            # 先保存用户消息到数据库（意图暂时设为None，后续可根据需要更新）
            message_created_at = self.save_conversation(
                chat_id, cookie_id, user_id, item_id, "user", message, intent=None
            )
            
            # 消息去抖处理
            if not skip_wait:
                logger.info(f"【{cookie_id}】消息已保存，等待10秒收集后续消息: {message[:20]}...")
                time.sleep(10)
            else:
                logger.info(f"【{cookie_id}】消息已保存（外部防抖已启用）: {message[:20]}...")
            
            # 获取该chat_id的锁，确保同一对话的消息串行处理
            chat_lock = self._get_chat_lock(chat_id)
            
            with chat_lock:
                # 检查是否有更新的消息
                query_seconds = 6 if skip_wait else 25
                recent_messages = self._get_recent_user_messages(chat_id, cookie_id, seconds=query_seconds)
                
                if recent_messages and len(recent_messages) > 0:
                    latest_message = recent_messages[-1]
                    if message_created_at != latest_message['created_at']:
                        logger.info(f"【{cookie_id}】检测到更新消息，跳过当前消息")
                        return None
                
                # 1. 获取AI设置
                settings = db_manager.get_ai_reply_settings(cookie_id)

                # 优先级：商品级预设 > 账号活跃预设 > custom_prompts字段（向后兼容）> 代码默认值
                preset = db_manager.get_preset_for_item(cookie_id, item_id)
                if not preset:
                    preset = db_manager.get_active_preset(cookie_id)
                if preset and preset.get('system_prompt'):
                    _override_system_prompt = preset['system_prompt']
                    custom_prompts = {}
                elif preset:
                    # 旧4字段预设（迁移期兼底层兼容）
                    _override_system_prompt = None
                    custom_prompts = {
                        'price': preset.get('price_prompt', ''),
                        'tech': preset.get('tech_prompt', ''),
                        'default': preset.get('default_prompt', ''),
                        'knowledge_base': preset.get('knowledge_base', ''),
                    }
                else:
                    _override_system_prompt = None
                    custom_prompts = json.loads(settings['custom_prompts']) if settings.get('custom_prompts') else {}

                # 2. 获取对话历史
                context = self.get_conversation_context(chat_id, cookie_id)

                # 3. 获取对话轮数和议价设置（供AI参考）
                conversation_rounds = self.get_conversation_rounds(chat_id, cookie_id)
                max_bargain_rounds = settings.get('max_bargain_rounds', 3)
                max_discount_percent = settings.get('max_discount_percent', 10)
                max_discount_amount = settings.get('max_discount_amount', 100)

                # 4. 构建统一的系统提示词（整合意图判断和回复生成）
                system_prompt = _override_system_prompt or self._build_unified_system_prompt(custom_prompts, settings)

                # 5. 构建商品信息
                item_desc = f"商品标题: {item_info.get('title', '未知')}\n"
                item_desc += f"商品价格: {item_info.get('price', '未知')}元\n"
                item_desc += f"商品描述: {item_info.get('desc', '无')}"

                # 6. 构建对话历史字符串
                context_str = ""
                if context:
                    context_str = "\n".join([
                        f"{'客户' if msg['role'] == 'user' else '客服'}: {msg['content']}" 
                        for msg in context[-10:]
                    ])

                # 7. 构建用户消息（包含所有上下文）
                user_prompt = f"""## 商品信息
{item_desc}

## 对话历史
{context_str if context_str else '(新对话，暂无历史)'}

## 对话状态
- 当前对话轮数：第{conversation_rounds + 1}轮
- 议价限制：最多{max_bargain_rounds}轮议价后需坚持底价
- 最大可优惠：{max_discount_percent}%或{max_discount_amount}元

## 当前用户消息
{message}

请根据以上信息，直接回复用户："""

                # 8. 构建消息列表
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                # 9. 调用AI生成回复
                reply = None

                if self._is_dashscope_api(settings):
                    logger.info("使用DashScope API生成回复")
                    reply = self._call_dashscope_api(settings, messages, max_tokens=1024, temperature=0.4)

                elif self._is_gemini_api(settings):
                    logger.info("使用Gemini API生成回复")
                    reply = self._call_gemini_api(settings, messages, max_tokens=1024, temperature=0.4)

                elif self._is_anthropic_api(settings):
                    logger.info("使用Anthropic API生成回复")
                    reply = self._call_anthropic_api(settings, messages, max_tokens=1024, temperature=0.4)

                else:
                    logger.info("使用OpenAI兼容API生成回复")
                    client = self._create_openai_client(cookie_id)
                    if not client:
                        return None
                    reply = self._call_openai_api(client, settings, messages, max_tokens=1024, temperature=0.4)

                # 10. 保存AI回复到对话记录
                self.save_conversation(chat_id, cookie_id, user_id, item_id, "assistant", reply, intent=None)
                
                logger.info(f"AI回复生成成功 (账号: {cookie_id}): {reply}")
                return reply
                
        except Exception as e:
            logger.error(f"AI回复生成失败 {cookie_id}: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'url'):
                logger.error(f"请求URL: {e.response.url}")
            if hasattr(e, 'request') and hasattr(e.request, 'url'):
                logger.error(f"请求URL: {e.request.url}")
            return None

    async def generate_reply_async(self, message: str, item_info: dict, chat_id: str,
                                   cookie_id: str, user_id: str, item_id: str,
                                   skip_wait: bool = False) -> Optional[str]:
        """
        异步包装器：在独立线程池中执行同步的 `generate_reply`，并返回结果。
        这样可以在异步代码中直接 await，而不阻塞事件循环。
        """
        try:
            import asyncio as _asyncio
            return await _asyncio.to_thread(self.generate_reply, message, item_info, chat_id, cookie_id, user_id, item_id, skip_wait)
        except Exception as e:
            logger.error(f"异步生成回复失败: {e}")
            return None
    
    def get_conversation_context(self, chat_id: str, cookie_id: str, limit: int = 20) -> List[Dict]:
        """获取对话上下文"""
        try:
            with db_manager.lock:
                cursor = db_manager.conn.cursor()
                cursor.execute('''
                SELECT role, content FROM ai_conversations 
                WHERE chat_id = ? AND cookie_id = ? 
                ORDER BY created_at DESC LIMIT ?
                ''', (chat_id, cookie_id, limit))
                
                results = cursor.fetchall()
                context = [{"role": row[0], "content": row[1]} for row in reversed(results)]
                return context
        except Exception as e:
            logger.error(f"获取对话上下文失败: {e}")
            return []
    
    def save_conversation(self, chat_id: str, cookie_id: str, user_id: str, 
                         item_id: str, role: str, content: str, intent: str = None) -> Optional[str]:
        """保存对话记录，返回创建时间"""
        try:
            with db_manager.lock:
                cursor = db_manager.conn.cursor()
                cursor.execute('''
                INSERT INTO ai_conversations 
                (cookie_id, chat_id, user_id, item_id, role, content, intent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (cookie_id, chat_id, user_id, item_id, role, content, intent))
                db_manager.conn.commit()
                
                # 获取刚插入记录的created_at
                cursor.execute('''
                SELECT created_at FROM ai_conversations 
                WHERE rowid = last_insert_rowid()
                ''')
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            logger.error(f"保存对话记录失败: {e}")
            return None
    def get_conversation_rounds(self, chat_id: str, cookie_id: str) -> int:
        """获取对话轮数（用户消息数量）"""
        try:
            with db_manager.lock:
                cursor = db_manager.conn.cursor()
                cursor.execute('''
                SELECT COUNT(*) FROM ai_conversations 
                WHERE chat_id = ? AND cookie_id = ? AND role = 'user'
                ''', (chat_id, cookie_id))
                
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"获取对话轮数失败: {e}")
            return 0
    
    def _get_recent_user_messages(self, chat_id: str, cookie_id: str, seconds: int = 2) -> List[Dict]:
        """获取最近seconds秒内的所有用户消息（包含内容和时间戳）"""
        try:
            with db_manager.lock:
                cursor = db_manager.conn.cursor()
                # 先查询所有该chat的user消息，用于调试
                cursor.execute('''
                SELECT content, created_at, 
                       julianday('now') - julianday(created_at) as time_diff_days,
                       (julianday('now') - julianday(created_at)) * 86400.0 as time_diff_seconds
                FROM ai_conversations 
                WHERE chat_id = ? AND cookie_id = ? AND role = 'user' 
                ORDER BY created_at DESC LIMIT 10
                ''', (chat_id, cookie_id))
                
                all_messages = cursor.fetchall()
                logger.info(f"【调试】chat_id={chat_id} 最近10条user消息: {[(msg[0][:10], msg[1], f'{msg[3]:.2f}秒前') for msg in all_messages]}")
                
                # 正式查询
                cursor.execute('''
                SELECT content, created_at FROM ai_conversations 
                WHERE chat_id = ? AND cookie_id = ? AND role = 'user' 
                AND julianday('now') - julianday(created_at) < (? / 86400.0)
                ORDER BY created_at ASC
                ''', (chat_id, cookie_id, seconds))
                
                results = cursor.fetchall()
                return [{"content": row[0], "created_at": row[1]} for row in results]
        except Exception as e:
            logger.error(f"获取最近用户消息列表失败: {e}")
            return []
    


# 全局AI回复引擎实例
ai_reply_engine = AIReplyEngine()