import requests
from bs4 import BeautifulSoup
import logging
import threading
import time
from datetime import datetime, timedelta
import re
import hashlib
import random
from urllib.parse import urljoin
import json

class WebCrawler:
    def __init__(self):
        self.crawl_targets = [
            {
                "name": "OpenAI博客",
                "url": "https://openai.com/blog",
                "selector": "article.post",
                "title_selector": "h2",
                "link_selector": "a",
                "summary_selector": "p.post-excerpt",
                "date_selector": "time",
                "date_format": "%Y-%m-%d",
                "category": "公司动态",
                "priority": "高",
                "depth": 1  # 爬取深度，1表示只爬取列表页，2表示还爬取详情页
            },
            {
                "name": "智谱AI",
                "url": "https://www.zhipuai.cn/news",
                "selector": ".news-item",
                "title_selector": ".news-title",
                "link_selector": "a",
                "summary_selector": ".news-desc",
                "date_selector": ".news-date",
                "date_format": "%Y-%m-%d",
                "category": "公司动态",
                "priority": "高",
                "depth": 1
            },
            {
                "name": "百度研究院",
                "url": "https://research.baidu.com/Blog",
                "selector": ".post-item",
                "title_selector": ".post-title",
                "link_selector": "a",
                "summary_selector": ".post-excerpt",
                "date_selector": ".post-date",
                "date_format": "%Y-%m-%d",
                "category": "研究进展",
                "priority": "中",
                "depth": 2
            },
            {
                "name": "腾讯AI实验室",
                "url": "https://ai.tencent.com/ailab/zh/news/",
                "selector": ".news-item",
                "title_selector": "h3",
                "link_selector": "a",
                "summary_selector": "p",
                "date_selector": ".date",
                "date_format": "%Y-%m-%d",
                "category": "研究进展",
                "priority": "中",
                "depth": 1
            },
            {
                "name": "机器之心",
                "url": "https://www.jiqizhixin.com/categories/industry-news",
                "selector": ".article-item",
                "title_selector": "h4",
                "link_selector": "a",
                "summary_selector": ".article-content",
                "date_selector": ".article-info .date",
                "date_format": "%Y-%m-%d",
                "category": "行业新闻",
                "priority": "高",
                "depth": 2
            },
            {
                "name": "量子位",
                "url": "https://www.qbitai.com/category/ai-industry",
                "selector": "article",
                "title_selector": "h2.entry-title",
                "link_selector": "a",
                "summary_selector": ".entry-summary",
                "date_selector": ".entry-date",
                "date_format": "%Y-%m-%d",
                "category": "行业新闻",
                "priority": "高",
                "depth": 2
            },
            {
                "name": "清华大学智能产业研究院",
                "url": "https://www.riit.tsinghua.edu.cn/xwzx/index.htm",
                "selector": ".news_list li",
                "title_selector": "a",
                "link_selector": "a",
                "summary_selector": None,  # 没有摘要
                "date_selector": "span",
                "date_format": "%Y-%m-%d",
                "category": "学术研究",
                "priority": "中",
                "depth": 2
            },
            {
                "name": "北京智源人工智能研究院",
                "url": "https://www.baai.ac.cn/news.html",
                "selector": ".news-item",
                "title_selector": ".news-title",
                "link_selector": "a",
                "summary_selector": ".news-summary",
                "date_selector": ".news-date",
                "date_format": "%Y-%m-%d",
                "category": "学术研究",
                "priority": "中",
                "depth": 2
            }
        ]
        
        self.update_intervals = {"高": 6, "中": 12, "低": 24}  # 小时为单位
        self.last_update = {}  # 记录每个爬取目标的最后更新时间
        self.news_cache = {}  # 缓存已爬取的新闻
        self.lock = threading.Lock()  # 线程锁，用于多线程安全
        
        # 设置请求头列表，随机使用以减少被封风险
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59"
        ]
        
        # 设置代理列表（示例，实际应用中应使用有效的代理）
        self.proxies = [
            # {"http": "http://proxy1.example.com:8080", "https": "https://proxy1.example.com:8080"},
            # {"http": "http://proxy2.example.com:8080", "https": "https://proxy2.example.com:8080"}
        ]
        
        # 设置日志记录器
        self.logger = self._setup_logger()
        
        # 设置爬虫延迟（秒）
        self.min_delay = 3
        self.max_delay = 7
        
        # AIGC关键词列表
        self.aigc_keywords = [
            "AIGC", "生成式AI", "生成式人工智能", "生成式模型", "大模型", "大语言模型", "LLM", 
            "人工智能生成内容", "AI生成内容", "AI创作", "GPT", "ChatGPT", "Claude", "Gemini", 
            "Llama", "文心一言", "通义千问", "讯飞星火", "混元", "盘古", "ChatGLM", 
            "Stable Diffusion", "MidJourney", "DALL-E", "Sora", "多模态", "自回归", 
            "自监督学习", "预训练", "微调", "迁移学习", "语义理解", "深度学习", "神经网络", 
            "强化学习", "扩散模型", "LoRA", "PEFT", "QLoRA", "RAG", "检索增强生成", 
            "向量数据库", "知识图谱", "AI绘画", "AI写作", "AI音乐", "AI视频", "AI编程", 
            "AI代码", "AI翻译", "AI对话", "文本生成", "图像生成", "语音生成", "视频生成", 
            "代码生成", "内容创作", "OpenAI", "Anthropic", "Google DeepMind", "百度", 
            "阿里", "腾讯", "智谱AI", "华为", "科大讯飞", "商汤", "旷视", "提示
