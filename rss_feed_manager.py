import feedparser
import requests
import time
import threading
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import logging

class RSSFeedManager:
    def __init__(self):
        self.feed_sources = {
            "官方媒体": [
                {"name": "人民网科技", "url": "http://scitech.people.com.cn/rss/it.xml", "priority": "高"},
                {"name": "新华网科技", "url": "http://www.xinhuanet.com/tech/rss/tech.xml", "priority": "高"},
                {"name": "中国科学报", "url": "http://news.sciencenet.cn/rss.aspx", "priority": "中"},
                {"name": "科技日报", "url": "http://digitalpaper.stdaily.com/http_www.kjrb.com/kjrb/rss/rss.xml", "priority": "中"}
            ],
            "专业科技媒体": [
                {"name": "36氪", "url": "https://36kr.com/feed", "priority": "高"},
                {"name": "机器之心", "url": "https://www.jiqizhixin.com/rss", "priority": "高"},
                {"name": "量子位", "url": "https://www.qbitai.com/feed", "priority": "高"},
                {"name": "雷锋网", "url": "https://www.leiphone.com/feed", "priority": "中"},
                {"name": "智东西", "url": "https://zhidx.com/feed", "priority": "中"},
                {"name": "AI科技评论", "url": "https://www.jiqizhixin.com/rss/categories/5", "priority": "高"},
                {"name": "DeepTech深科技", "url": "https://www.mittrchina.com/feed", "priority": "中"}
            ],
            "研究机构": [
                {"name": "百度研究院", "url": "https://research.baidu.com/feed", "priority": "中"},
                {"name": "阿里达摩院", "url": "https://damo.alibaba.com/feed", "priority": "中"},
                {"name": "腾讯AI Lab", "url": "https://ai.tencent.com/ailab/feed", "priority": "中"},
                {"name": "智谱AI", "url": "https://www.zhipuai.cn/feed", "priority": "中"}
            ],
            "国外媒体": [
                {"name": "MIT科技评论中文版", "url": "https://www.mittrchina.com/feed", "priority": "中"},
                {"name": "哈佛商业评论中文版", "url": "https://www.hbrchina.org/feed", "priority": "低"},
                {"name": "Synced AI中文版", "url": "https://syncedreview.com/feed", "priority": "中"}
            ]
        }
        self.update_intervals = {"高": 1, "中": 3, "低": 6}  # 小时为单位
        self.last_update = {}  # 记录每个源的最后更新时间
        self.news_cache = {}  # 缓存已获取的新闻
        self.lock = threading.Lock()  # 线程锁，用于多线程安全
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger("RSSFeedManager")
        logger.setLevel(logging.INFO)
        
        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 创建文件处理器
        fh = logging.FileHandler("rss_manager.log")
        fh.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        # 添加处理器到记录器
        logger.addHandler(ch)
        logger.addHandler(fh)
        
        return logger
        
    def fetch_feeds(self):
        """获取所有RSS源的最新内容"""
        all_news = []
        current_time = datetime.now()
        
        # 为每个需要更新的源创建线程
        threads = []
        for category, sources in self.feed_sources.items():
            for source in sources:
                # 检查是否需要更新
                if source["name"] not in self.last_update:
                    self.last_update[source["name"]] = datetime.min
                
                hours_since_update = (current_time - self.last_update[source["name"]]).total_seconds() / 3600
                if hours_since_update >= self.update_intervals[source["priority"]]:
                    # 创建线程获取该源的内容
                    thread = threading.Thread(
                        target=self._fetch_single_feed,
                        args=(source, category, all_news)
                    )
                    threads.append(thread)
                    thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 过滤AIGC相关内容
        aigc_news = []
        for news in all_news:
            if self.filter_aigc_content(news):
                aigc_news.append(news)
                
        self.logger.info(f"获取到 {len(all_news)} 条新闻，其中 {len(aigc_news)} 条与AIGC相关")
        return aigc_news
    
    def _fetch_single_feed(self, source, category, result_list):
        """获取单个RSS源的内容"""
        try:
            self.logger.info(f"开始获取 {source['name']} 的RSS内容")
            
            # 使用feedparser解析RSS源
            feed = feedparser.parse(source["url"])
            
            # 检查是否成功获取
            if hasattr(feed, 'status') and feed.status != 200:
                self.logger.warning(f"获取 {source['name']} 失败，状态码: {feed.status}")
                return
            
            # 更新最后更新时间
            with self.lock:
                self.last_update[source["name"]] = datetime.now()
            
            # 处理每个条目
            for entry in feed.entries:
                # 提取发布时间
                if hasattr(entry, 'published_parsed'):
                    published_time = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, 'updated_parsed'):
                    published_time = datetime(*entry.updated_parsed[:6])
                else:
                    published_time = datetime.now()
                
                # 创建新闻项
                news_item = {
                    "title": entry.title if hasattr(entry, 'title') else "无标题",
                    "link": entry.link if hasattr(entry, 'link') else "",
                    "summary": self._clean_html(entry.summary) if hasattr(entry, 'summary') else "",
                    "published": published_time,
                    "source": source["name"],
                    "category": category,
                    "priority": source["priority"],
                    "content": self._get_full_content(entry) if hasattr(entry, 'link') else "",
                    "id": entry.id if hasattr(entry, 'id') else entry.link
                }
                
                # 检查是否已经存在
                if news_item["id"] not in self.news_cache:
                    with self.lock:
                        self.news_cache[news_item["id"]] = news_item
                        result_list.append(news_item)
                        
            self.logger.info(f"成功获取 {source['name']} 的RSS内容，共 {len(feed.entries)} 条")
            
        except Exception as e:
            self.logger.error(f"获取 {source['name']} 时出错: {str(e)}")
    
    def _clean_html(self, html_text):
        """清理HTML标签"""
        soup = BeautifulSoup(html_text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    
    def _get_full_content(self, entry):
        """尝试获取完整内容"""
        if hasattr(entry, 'content'):
            content_text = ""
            for content in entry.content:
                if content.get('type') == 'text/html':
                    content_text += self._clean_html(content.value)
            return content_text
        
        # 如果entry中没有完整内容，尝试从链接获取
        try:
            if hasattr(entry, 'link'):
                response = requests.get(entry.link, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # 尝试找到文章主体（这里的选择器需要根据不同网站调整）
                    article = soup.find('article') or soup.find('div', class_='article') or soup.find('div', class_='content')
                    if article:
                        return article.get_text(separator=' ', strip=True)
        except Exception as e:
            self.logger.warning(f"获取完整内容失败: {str(e)}")
        
        # 如果无法获取完整内容，返回摘要
        return entry.summary if hasattr(entry, 'summary') else ""
    
    def filter_aigc_content(self, news):
        """过滤与AIGC相关的内容"""
        # AIGC相关关键词列表
        aigc_keywords = [
            # 基础术语
            "AIGC", "生成式AI", "生成式人工智能", "生成式模型", "大模型", "大语言模型", "LLM", 
            "人工智能生成内容", "AI生成内容", "AI创作",
            
            # 模型名称
            "GPT", "GPT-4", "GPT-5", "ChatGPT", "Claude", "Gemini", "Llama", "PaLM", 
            "文心一言", "通义千问", "讯飞星火", "混元", "盘古", "ChatGLM", "Stable Diffusion",
            "MidJourney", "DALL-E", "Sora", "Anthropic", "Bard",
            
            # 技术术语
            "Transformer", "注意力机制", "自回归", "自监督学习", "预训练", "微调", "迁移学习",
            "多模态", "跨模态", "语义理解", "深度学习", "神经网络", "强化学习", "扩散模型",
            "LoRA", "PEFT", "QLoRA", "RAG", "检索增强生成", "向量数据库", "知识图谱",
            
            # 应用领域
            "AI绘画", "AI写作", "AI音乐", "AI视频", "AI编程", "AI代码", "AI翻译", "AI对话",
            "文本生成", "图像生成", "语音生成", "视频生成", "代码生成", "内容创作",
            
            # 公司和研究机构
            "OpenAI", "Anthropic", "Google DeepMind", "百度", "阿里", "腾讯", "智谱AI",
            "华为", "科大讯飞", "商汤", "旷视", "清华", "北大", "智源研究院",
            
            # 相关概念
            "提示工程", "Prompt", "Token", "参数", "训练数据", "上下文窗口", "幻觉",
            "偏见", "伦理", "版权", "监管", "合规"
        ]
        
        # 创建正则表达式模式，匹配这些关键词
        pattern = r'\b(' + '|'.join(aigc_keywords) + r')\b'
        regex = re.compile(pattern, re.IGNORECASE)
        
        # 检查标题中是否包含关键词
        if regex.search(news["title"]):
            return True
        
        # 检查摘要中是否包含关键词
        if regex.search(news["summary"]):
            return True
        
        # 检查内容中是否包含关键词
        if news["content"] and regex.search(news["content"]):
            return True
        
        # 进行更深入的内容分析
        # 计算AIGC相关词汇在内容中的密度
        if news["content"]:
            matches = regex.findall(news["content"])
            content_length = len(news["content"])
            
            # 如果内容长度超过500字符，且包含至少3个关键词，或者关键词密度超过0.5%
            if (content_length > 500 and len(matches) >= 3) or (content_length > 0 and len(matches) / content_length > 0.005):
                return True
        
        # 特殊情况：检查某些高相关度的短语
        high_relevance_phrases = [
            "大模型应用", "AI大模型", "生成式AI应用", "AIGC技术", "LLM技术", 
            "GPT应用", "ChatGPT使用", "AI内容创作", "AI生成工具"
        ]
        
        for phrase in high_relevance_phrases:
            if phrase in news["title"] or phrase in news["summary"] or (news["content"] and phrase in news["content"]):
                return True
        
        # 如果没有匹配到任何AIGC相关内容，返回False
        return False
