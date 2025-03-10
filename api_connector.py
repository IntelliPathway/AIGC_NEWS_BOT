import requests
import json
import time
import threading
import logging
from datetime import datetime, timedelta
import hashlib
import re
from urllib.parse import urlencode

class APIConnector:
    def __init__(self):
        self.api_sources = [
            {
                "name": "HuggingFace",
                "endpoint": "https://huggingface.co/api/models",
                "params": {"tags": "text-generation", "sort": "lastModified", "limit": 30},
                "priority": "高",
                "type": "model_repository",
                "auth_required": False
            },
            {
                "name": "arXiv",
                "endpoint": "http://export.arxiv.org/api/query",
                "params": {
                    "search_query": "all:ai+OR+all:\"large language model\"+OR+all:\"generative ai\"",
                    "sortBy": "lastUpdatedDate",
                    "sortOrder": "descending",
                    "max_results": 50
                },
                "priority": "高",
                "type": "academic",
                "auth_required": False
            },
            {
                "name": "GitHub",
                "endpoint": "https://api.github.com/search/repositories",
                "params": {
                    "q": "large language model OR generative ai OR AIGC",
                    "sort": "updated",
                    "order": "desc",
                    "per_page": 30
                },
                "priority": "中",
                "type": "code_repository",
                "auth_required": False
            },
            {
                "name": "OpenAI博客",
                "endpoint": "https://openai.com/api/blog",
                "params": {},
                "priority": "高",
                "type": "company_blog",
                "auth_required": False,
                "fallback_to_crawler": True
            },
            {
                "name": "百度AI开放平台",
                "endpoint": "https://ai.baidu.com/forum/api/forumlist",
                "params": {"kw": "文心一言", "pn": 1, "rn": 20},
                "priority": "中",
                "type": "company_platform",
                "auth_required": False
            },
            {
                "name": "AI研习社",
                "endpoint": "https://api.yanxishe.com/api/article/list",
                "params": {"page": 1, "pageSize": 20, "categoryId": ""},
                "priority": "中",
                "type": "community",
                "auth_required": False
            }
        ]
        
        # API密钥配置（实际应用中应从安全的环境变量或配置文件中读取）
        self.api_keys = {
            # "API名称": "API密钥"
        }
        
        self.update_intervals = {"高": 3, "中": 6, "低": 12}  # 小时为单位
        self.last_update = {}  # 记录每个API源的最后更新时间
        self.news_cache = {}  # 缓存已获取的新闻
        self.lock = threading.Lock()  # 线程锁，用于多线程安全
        
        # 设置请求头
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json"
        }
        
        # 设置日志记录器
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger("APIConnector")
        logger.setLevel(logging.INFO)
        
        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 创建文件处理器
        fh = logging.FileHandler("api_connector.log")
        fh.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        # 添加处理器到记录器
        logger.addHandler(ch)
        logger.addHandler(fh)
        
        return logger
    
    def fetch_api_data(self):
        """从各API获取最新数据"""
        all_news = []
        current_time = datetime.now()
        
        # 为每个需要更新的API源创建线程
        threads = []
        for source in self.api_sources:
            # 检查是否需要更新
            if source["name"] not in self.last_update:
                self.last_update[source["name"]] = datetime.min
            
            hours_since_update = (current_time - self.last_update[source["name"]]).total_seconds() / 3600
            if hours_since_update >= self.update_intervals[source["priority"]]:
                # 创建线程获取该API的内容
                thread = threading.Thread(
                    target=self._fetch_single_api,
                    args=(source, all_news)
                )
                threads.append(thread)
                thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        self.logger.info(f"从API源获取到 {len(all_news)} 条新闻")
        return all_news
    
    def _fetch_single_api(self, source, result_list):
        """获取单个API源的内容"""
        try:
            self.logger.info(f"开始获取 {source['name']} 的API数据")
            
            # 准备请求头和参数
            headers = self.headers.copy()
            params = source["params"].copy()
            
            # 如果需要认证，添加API密钥
            if source["auth_required"] and source["name"] in self.api_keys:
                if source["name"] == "GitHub":
                    headers["Authorization"] = f"token {self.api_keys['GitHub']}"
                else:
                    params["api_key"] = self.api_keys[source["name"]]
            
            # 发送请求
            response = requests.get(
                source["endpoint"],
                headers=headers,
                params=params,
                timeout=30
            )
            
            # 检查响应状态
            if response.status_code != 200:
                self.logger.warning(f"获取 {source['name']} 失败，状态码: {response.status_code}")
                if source.get("fallback_to_crawler", False):
                    self.logger.info(f"尝试使用网页爬虫作为 {source['name']} 的备选方案")
                    # 这里可以调用网页爬虫模块的方法
                return
            
            # 更新最后更新时间
            with self.lock:
                self.last_update[source["name"]] = datetime.now()
            
            # 根据不同的API源类型解析响应
            news_items = self._parse_api_response(source, response)
            
            # 添加到结果列表
            with self.lock:
                for item in news_items:
                    if item["id"] not in self.news_cache:
                        self.news_cache[item["id"]] = item
                        result_list.append(item)
            
            self.logger.info(f"成功获取 {source['name']} 的API数据，共 {len(news_items)} 条")
            
        except Exception as e:
            self.logger.error(f"获取 {source['name']} 时出错: {str(e)}")
    
    def _parse_api_response(self, source, response):
        """根据不同的API源类型解析响应"""
        news_items = []
        
        try:
            if source["type"] == "model_repository":
                if source["name"] == "HuggingFace":
                    # 解析HuggingFace API响应
                    data = response.json()
                    for model in data:
                        # 检查是否与AIGC相关
                        if self._is_aigc_related_model(model):
                            news_item = {
                                "title": f"HuggingFace新模型: {model['modelId']}",
                                "link": f"https://huggingface.co/{model['modelId']}",
                                "summary": model.get('description', '无描述'),
                                "published": datetime.strptime(model['lastModified'], "%Y-%m-%dT%H:%M:%S.%fZ") if 'lastModified' in model else datetime.now(),
                                "source": source["name"],
                                "category": "模型发布",
                                "content": f"模型ID: {model['modelId']}\n"
                                           f"作者: {model.get('author', '未知')}\n"
                                           f"标签: {', '.join(model.get('tags', []))}\n"
                                           f"描述: {model.get('description', '无描述')}",
                                "id": f"huggingface_{model['modelId']}",
                                "priority": source["priority"]
                            }
                            news_items.append(news_item)
            
            elif source["type"] == "academic":
                if source["name"] == "arXiv":
                    # 解析arXiv API响应（XML格式）
                    from xml.etree import ElementTree as ET
                    root = ET.fromstring(response.content)
                    
                    # 定义命名空间
                    namespaces = {
                        'atom': 'http://www.w3.org/2005/Atom',
                        'arxiv': 'http://arxiv.org/schemas/atom'
                    }
                    
                    # 遍历所有条目
                    for entry in root.findall('.//atom:entry', namespaces):
                        title = entry.find('atom:title', namespaces).text
                        link = entry.find('./atom:link[@title="pdf"]', namespaces)
                        if link is not None:
                            link = link.attrib['href']
                        else:
                            link = entry.find('atom:id', namespaces).text
                        
                        summary = entry.find('atom:summary', namespaces).text
                        published = entry.find('atom:published', namespaces).text
                        authors = [author.find('atom:name', namespaces).text for author in entry.findall('atom:author', namespaces)]
                        
                        # 检查是否与AIGC相关
                        if self._is_aigc_related_paper(title, summary):
                            news_item = {
                                "title": title,
                                "link": link,
                                "summary": summary[:300] + "..." if len(summary) > 300 else summary,
                                "published": datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ"),
                                "source": source["name"],
                                "category": "学术研究",
                                "content": f"标题: {title}\n"
                                           f"作者: {', '.join(authors)}\n"
                                           f"摘要: {summary}\n",
                                "id": f"arxiv_{link.split('/')[-1]}",
                                "priority": source["priority"]
                            }
                            news_items.append(news_item)
            
            elif source["type"] == "code_repository":
                if source["name"] == "GitHub":
                    # 解析GitHub API响应
                    data = response.json()
                    for repo in data.get("items", []):
                        # 检查是否与AIGC相关
                        if self._is_aigc_related_repo(repo):
                            news_item = {
                                "title": f"GitHub项目更新: {repo['full_name']}",
                                "link": repo['html_url'],
                                "summary": repo['description'] if repo['description'] else "无描述",
                                "published": datetime.strptime(repo['updated_at'], "%Y-%m-%dT%H:%M:%SZ"),
                                "source": source["name"],
                                "category": "开源项目",
                                "content": f"项目名称: {repo['full_name']}\n"
                                           f"描述: {repo['description'] if repo['description'] else '无描述'}\n"
                                           f"星标数: {repo['stargazers_count']}\n"
                                           f"Fork数: {repo['forks_count']}\n"
                                           f"主要语言: {repo['language']}\n"
                                           f"最近更新: {repo['updated_at']}\n",
                                "id": f"github_{repo['id']}",
                                "priority": source["priority"]
                            }
                            news_items.append(news_item)
            
            elif source["type"] == "company_blog":
                # 针对不同公司博客的解析逻辑
                if source["name"] == "OpenAI博客":
                    try:
                        data = response.json()
                        for post in data.get("items", []):
                            news_item = {
                                "title": post.get("title", "无标题"),
                                "link": f"https://openai.com/blog/{post.get('slug', '')}",
                                "summary": post.get("summary", "无摘要"),
                                "published": datetime.strptime(post.get("published_at", ""), "%Y-%m-%dT%H:%M:%SZ") if post.get("published_at") else datetime.now(),
                                "source": source["name"],
                                "category": "公司动态",
                                "content": post.get("content", "无内容"),
                                "id": f"openai_blog_{post.get('slug', '')}",
                                "priority": source["priority"]
                            }
                            news_items.append(news_item)
                    except:
                        self.logger.warning(f"解析 {source['name']} 响应失败，可能不是标准JSON格式")
            
            elif source["type"] == "company_platform":
                if source["name"] == "百度AI开放平台":
                    try:
                        data = response.json()
                        for post in data.get("data", {}).get("list", []):
                            news_item = {
                                "title": post.get("title", "无标题"),
                                "link": f"https://ai.baidu.com/forum/detail?id={post.get('id', '')}",
                                "summary": post.get("abstract", "无摘要"),
                                "published": datetime.fromtimestamp(post.get("create_time", 0)),
                                "source": source["name"],
                                "category": "平台动态",
                                "content": post.get("content", "无内容"),
                                "id": f"baidu_ai_forum_{post.get('id', '')}",
                                "priority": source["priority"]
                            }
                            news_items.append(news_item)
                    except:
                        self.logger.warning(f"解析 {source['name']} 响应失败")
            
            elif source["type"] == "community":
                if source["name"] == "AI研习社":
                    try:
                        data = response.json()
                        for article in data.get("data", {}).get("list", []):
                            news_item = {
                                "title": article.get("title", "无标题"),
                                "link": f"https://yanxishe.com/articleDetail/{article.get('id', '')}",
                                "summary": article.get("summary", "无摘要"),
                                "published": datetime.fromtimestamp(article.get("publishTime", 0)/1000),
                                "source": source["name"],
                                "category": "社区内容",
                                "content": article.get("content", "无内容"),
                                "id": f"ai_yanxishe_{article.get('id', '')}",
                                "priority": source["priority"]
                            }
                            news_items.append(news_item)
                    except:
                        self.logger.warning(f"解析 {source['name']} 响应失败")
        
        except Exception as e:
            self.logger.error(f"解析 {source['name']} 响应时出错: {str(e)}")
        
        return news_items
    
    def _is_aigc_related_model(self, model):
        """判断HuggingFace模型是否与AIGC相关"""
        aigc_tags = [
            "text-generation", "text2text-generation", "text-to-image", "image-to-text",
            "image-generation", "text-to-audio", "text-to-video", "diffusion", "gpt",
            "llm", "large-language-model", "generative-ai", "aigc"
        ]
        
        # 检查标签
        if "tags" in model:
            for tag in model["tags"]:
                if tag.lower() in aigc_tags:
                    return True
        
        # 检查模型ID和描述
        model_id = model.get("modelId", "").lower()
        description = model.get("description", "").lower()
        
        aigc_keywords = [
            "gpt", "llm", "llama", "falcon", "bloom", "bert", "t5", "gpt-neo", "gpt-j",
            "stable-diffusion", "midjourney", "dall-e", "clip", "whisper", "wav2vec",
            "generative", "generation", "transformer", "diffusion"
        ]
        
        for keyword in aigc_keywords:
            if keyword in model_id or keyword in description:
                return True
        
        return False
    
    def _is_aigc_related_paper(self, title, abstract):
        """判断arXiv论文是否与AIGC相关"""
        title_lower = title.lower()
        abstract_lower = abstract.lower()
        
        aigc_keywords = [
            "large language model", "llm", "generative ai", "generative model",
            "diffusion model", "transformer", "attention mechanism", "gpt", "bert", "t5",
            "llama", "falcon", "bloom", "stable diffusion", "midjourney", "dall-e",
            "text-to-image", "text-to-video", "text-to-audio", "multimodal generation",
            "neural generation", "language generation", "image generation", "video generation",
            "audio generation", "self-supervised", "self-attention", "autoregressive",
            "foundation model", "prompt", "in-context learning", "few-shot learning",
            "zero-shot learning", "fine-tuning", "transfer learning", "reinforcement learning",
            "rlhf", "constitutional ai", "alignment", "hallucination", "bias", "ethics"
        ]
        
        # 检查标题
        for keyword in aigc_keywords:
            if keyword in title_lower:
                return True
        
        # 检查摘要
        keyword_count = 0
        for keyword in aigc_keywords:
            if keyword in abstract_lower:
                keyword_count += 1
                if keyword_count >= 2:  # 如果摘要中包含至少2个关键词，则认为相关
                    return True
        
        return False
    
    def _is_aigc_related_repo(self, repo):
        """判断GitHub仓库是否与AIGC相关"""
        name_lower = repo.get("name", "").lower()
        full_name_lower = repo.get("full_name", "").lower()
        description_lower = repo.get("description", "").lower() if repo.get("description") else ""
        
        aigc_keywords = [
            "llm", "gpt", "bert", "t5", "llama", "falcon", "bloom", "chatgpt", "claude",
            "stable-diffusion", "midjourney", "dall-e", "diffusion", "transformer",
            "attention", "generative", "generation", "text-to-image", "text-to-video",
            "whisper", "wav2vec", "huggingface", "langchain", "openai", "anthropic",
            "prompt-engineering", "fine-tuning", "rlhf", "aigc"
        ]
        
        # 检查仓库名称
        for keyword in aigc_keywords:
            if keyword in name_lower or keyword in full_name_lower:
                return True
        
        # 检查描述
        if description_lower:
            keyword_count = 0
            for keyword in aigc_keywords:
                if keyword in description_lower:
                    keyword_count += 1
                    if keyword_count >= 1:  # 如果描述中包含至少1个关键词，则认为相关
                        return True
        
        # 检查主题标签
        if "topics" in repo:
            for topic in repo["topics"]:
                if topic.lower() in aigc_keywords:
                    return True
        
        return False
