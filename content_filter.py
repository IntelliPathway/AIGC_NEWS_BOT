import re
import jieba
import jieba.analyse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime
import json

class ContentFilter:
    def __init__(self):
        # AIGC相关关键词列表
        self.aigc_keywords = [
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
        
        # 内容分类类别及关键词
        self.categories = {
            "技术突破": [
                "算法", "模型", "架构", "参数", "训练", "推理", "性能", "优化", "突破", "创新",
                "研究", "论文", "实验", "精度", "效率", "速度", "技术", "方法", "提升", "改进",
                "新技术", "新方法", "新算法", "新架构", "新模型", "新突破", "研发", "开发",
                "技术路线", "技术方案", "技术框架", "技术栈", "技术生态", "技术体系",
                "核心技术", "关键技术", "前沿技术", "尖端技术", "底层技术", "基础技术"
            ],
            "产品发布": [
                "发布", "上线", "测试", "版本", "更新", "迭代", "产品", "服务", "功能", "特性",
                "应用", "工具", "平台", "系统", "解决方案", "SDK", "API", "接口", "开放", "公测",
                "内测", "正式版", "测试版", "预览版", "体验版", "免费版", "专业版", "企业版",
                "新品", "新产品", "新服务", "新功能", "新特性", "新版本", "产品线", "产品矩阵",
                "产品生态", "产品体系", "产品策略", "产品规划", "产品路线图", "产品定位"
            ],
            "政策法规": [
                "政策", "法规", "合规", "监管", "标准", "伦理", "规范", "指南", "条例", "办法",
                "意见", "通知", "决定", "规定", "措施", "管理", "治理", "审查", "评估", "认证",
                "许可", "备案", "登记", "申报", "立法", "执法", "司法", "普法", "执行", "实施",
                "国家标准", "行业标准", "团体标准", "企业标准", "国际标准", "技术标准",
                "管理标准", "服务标准", "安全标准", "质量标准", "伦理标准", "道德标准"
            ],
            "投融资": [
                "融资", "投资", "估值", "收购", "IPO", "上市", "并购", "战略投资", "风投", "天使轮",
                "A轮", "B轮", "C轮", "D轮", "Pre-IPO", "私募", "公募", "股权", "债权", "基金",
                "资本", "市值", "募资", "退出", "回报", "收益", "利润", "营收", "财报", "财务",
                "投资人", "投资机构", "风险投资", "创业投资", "战略投资者", "财务投资者",
                "产业投资", "政府投资", "企业投资", "个人投资", "跨境投资", "海外投资"
            ],
            "行业应用": [
                "应用", "落地", "场景", "解决方案", "案例", "实践", "实施", "部署", "集成", "对接",
                "行业", "领域", "垂直", "细分", "专业", "定制", "客户", "用户", "企业", "机构",
                "政府", "医疗", "金融", "教育", "制造", "零售", "物流", "交通", "能源", "农业",
                "文娱", "媒体", "法律", "安防", "环保", "建筑", "地产", "旅游", "餐饮", "服装",
                "家居", "汽车", "电子", "通信", "互联网", "软件", "硬件", "云计算", "大数据", "物联网"
            ],
            "人才动态": [
                "人才", "招聘", "加盟", "跳槽", "离职", "任命", "晋升", "调动", "人事", "组织",
                "团队", "专家", "学者", "研究员", "工程师", "科学家", "教授", "博士", "硕士", "人才争夺",
                "人才流动", "人才引进", "人才培养", "人才储备", "人才梯队", "人才战略", "人才政策",
                "首席科学家", "首席技术官", "首席研究员", "技术专家", "研发主管", "研发总监",
                "技术总监", "技术负责人", "产品负责人", "项目负责人", "团队负责人", "部门负责人"
            ],
            "伦理与社会影响": [
                "伦理", "道德", "社会", "影响", "责任", "风险", "安全", "隐私", "公平", "透明",
                "可解释", "可信", "可控", "偏见", "歧视", "误导", "虚假", "真实", "真假", "鉴别",
                "识别", "辨别", "版权", "知识产权", "著作权", "专利权", "商标权", "肖像权", "名誉权",
                "隐私权", "数据权", "算法公平", "算法透明", "算法偏见", "算法歧视", "算法责任",
                "社会责任", "企业责任", "技术伦理", "AI伦理", "伦理准则", "伦理规范", "伦理框架"
            ]
        }
        
        # 加载停用词
        self.stopwords = self._load_stopwords()
        
        # 初始化TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=self._tokenize,
            stop_words=self.stopwords,
            max_features=5000
        )
        
        # 预先计算各类别的TF-IDF向量
        self._prepare_category_vectors()
        
        # 设置日志记录器
        self.logger = self._setup_logger()
        
        # 加载同义词词典
        self.synonyms = self._load_synonyms()
        
        # 加载jieba自定义词典
        self._load_custom_dict()
    
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger("ContentFilter")
        logger.setLevel(logging.INFO)
        
        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 创建文件处理器
        fh = logging.FileHandler("content_filter.log")
        fh.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        # 添加处理器到记录器
        logger.addHandler(ch)
        logger.addHandler(fh)
        
        return logger
    
    def _load_stopwords(self):
        """加载停用词表"""
        try:
            # 尝试从文件加载停用词
            with open("stopwords.txt", "r", encoding="utf-8") as f:
                stopwords = set([line.strip() for line in f])
            return stopwords
        except:
            # 如果文件不存在，返回一个基本的停用词集合
            return set(["的", "了", "和", "是", "在", "有", "为", "与", "等", "这", "那", "也", "中", "上", "下"])
    
    def _load_synonyms(self):
        """加载同义词词典"""
        try:
            with open("synonyms.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            # 如果文件不存在，返回一个基本的同义词词典
            return {
                "大模型": ["LLM", "大语言模型", "foundation model", "基础模型"],
                "ChatGPT": ["GPT-4", "GPT-3.5", "GPT"],
                "文心一言": ["ERNIE Bot", "百度大模型"],
                "通义千问": ["Qwen", "阿里大模型"],
                "讯飞星火": ["星火认知大模型", "讯飞大模型"],
                "生成式AI": ["AIGC", "生成式人工智能", "generative AI"],
                "多模态": ["multimodal", "跨模态"],
                "预训练": ["pre-training", "pretrain"],
                "微调": ["fine-tuning", "finetune"],
                "提示工程": ["prompt engineering", "prompt design"]
            }
    
    def _load_custom_dict(self):
        """加载jieba自定义词典"""
        # 将AIGC关键词和分类关键词添加到jieba词典
        all_keywords = set(self.aigc_keywords)
        for category, keywords in self.categories.items():
            all_keywords.update(keywords)
        
        # 将同义词也添加到jieba词典
        for key, values in self.synonyms.items():
            all_keywords.add(key)
            all_keywords.update(values)
        
        # 将关键词写入临时文件
        try:
            with open("custom_dict.txt", "w", encoding="utf-8") as f:
                for keyword in all_keywords:
                    f.write(f"{keyword} 10 n\n")  # 词 词频 词性
            
            # 加载自定义词典
            jieba.load_userdict("custom_dict.txt")
        except Exception as e:
            self.logger.error(f"加载自定义词典失败: {str(e)}")
    
    def _tokenize(self, text):
        """分词并去除停用词"""
        # 使用jieba进行分词
        words = jieba.cut(text)
        # 去除停用词
        return [word for word in words if word not in self.stopwords and len(word.strip()) > 1]
    
    def _prepare_category_vectors(self):
        """预先计算各类别的TF-IDF向量"""
        # 为每个类别创建一个文档
        category_docs = []
        self.category_names = []
        
        for category, keywords in self.categories.items():
            # 将类别关键词连接成一个文档
            category_docs.append(" ".join(keywords))
            self.category_names.append(category)
        
        # 计算TF-IDF矩阵
        self.category_matrix = self.vectorizer.fit_transform(category_docs)
    
    def is_aigc_related(self, news):
        """判断内容是否与AIGC相关"""
        # 创建正则表达式模式，匹配AIGC关键词
        pattern = r'\b(' + '|'.join(self.aigc_keywords) + r')\b'
        regex = re.compile(pattern, re.IGNORECASE)
        
        # 检查标题中是否包含关键词
        if regex.search(news["title"]):
            self.logger.info(f"标题包含AIGC关键词: {news['title']}")
            return True
        
        # 检查摘要中是否包含关键词
        if "summary" in news and news["summary"] and regex.search(news["summary"]):
            self.logger.info(f"摘要包含AIGC关键词: {news['summary'][:100]}...")
            return True
        
        # 检查内容中是否包含关键词
        if "content" in news and news["content"]:
            matches = regex.findall(news["content"])
            content_length = len(news["content"])
            
            # 如果内容长度超过500字符，且包含至少3个关键词，或者关键词密度超过0.5%
            if (content_length > 500 and len(matches) >= 3) or (content_length > 0 and len(matches) / content_length > 0.005):
                self.logger.info(f"内容包含AIGC关键词: 找到{len(matches)}个关键词")
                return True
        
        # 检查同义词
        for keyword, synonyms in self.synonyms.items():
            # 如果关键词本身是AIGC关键词
            if keyword in self.aigc_keywords:
                # 检查标题、摘要和内容中是否包含同义词
                for synonym in synonyms:
                    if synonym in news["title"]:
                        self.logger.info(f"标题包含AIGC同义词: {synonym}")
                        return True
                    if "summary" in news and news["summary"] and synonym in news["summary"]:
                        self.logger.info(f"摘要包含AIGC同义词: {synonym}")
                        return True
                    if "content" in news and news["content"] and synonym in news["content"]:
                        self.logger.info(f"内容包含AIGC同义词: {synonym}")
                        return True
        
        # 特殊情况：检查某些高相关度的短语
        high_relevance_phrases = [
            "大模型应用", "AI大模型", "生成式AI应用", "AIGC技术", "LLM技术", 
            "GPT应用", "ChatGPT使用", "AI内容创作", "AI生成工具"
        ]
        
        for phrase in high_relevance_phrases:
            if phrase in news["title"] or ("summary" in news and news["summary"] and phrase in news["summary"]) or ("content" in news and news["content"] and phrase in news["content"]):
                self.logger.info(f"包含高相关度短语: {phrase}")
                return True
        
        # 如果没有匹配到任何AIGC相关内容，返回False
        return False
    
    def categorize_content(self, news):
        """对内容进行分类"""
        # 准备新闻文本
        news_text = news["title"]
        if "summary" in news and news["summary"]:
            news_text += " " + news["summary"]
        if "content" in news and news["content"]:
            news_text += " " + news["content"]
        
        # 转换为TF-IDF向量
        try:
            news_vector = self.vectorizer.transform([news_text])
        except:
            # 如果向量化失败（可能是因为词汇表不匹配），重新训练向量化器
            self._prepare_category_vectors()
            news_vector = self.vectorizer.transform([news_text])
        
        # 计算与各类别的余弦相似度
        similarities = cosine_similarity(news_vector, self.category_matrix)[0]
        
        # 找出相似度最高的类别
        max_similarity_index = np.argmax(similarities)
        max_similarity = similarities[max_similarity_index]
        
        # 如果最高相似度低于阈值，则分类为"其他"
        if max_similarity < 0.1:
            self.logger.info(f"新闻 '{news['title']}' 分类为: 其他 (最高相似度: {max_similarity:.4f})")
            return "其他"
        
        # 获取最相似的类别
        category = self.category_names[max_similarity_index]
        self.logger.info(f"新闻 '{news['title']}' 分类为: {category} (相似度: {max_similarity:.4f})")
        
        # 返回分类结果
        return category
    
    def extract_keywords(self, news, top_n=5):
        """提取新闻的关键词"""
        # 准备新闻文本
        news_text = news["title"]
        if "summary" in news and news["summary"]:
            news_text += " " + news["summary"]
        if "content" in news and news["content"]:
            news_text += " " + news["content"]
        
        # 使用jieba的TF-IDF算法提取关键词
        keywords = jieba.analyse.extract_tags(news_text, topK=top_n)
        
        self.logger.info(f"从新闻 '{news['title']}' 提取的关键词: {', '.join(keywords)}")
        return keywords
    
    def filter_sensitive_content(self, news):
        """过滤敏感内容"""
        # 敏感词列表（示例）
        sensitive_words = [
            "国家机密", "军事机密", "保密", "绝密", "机密", "内部文件", "泄密",
            "违法", "违规", "犯罪", "贿赂", "腐败", "欺诈", "造假", "虚假宣传",
            "歧视", "种族", "性别", "宗教", "政治倾向"
        ]
        
        # 检查内容是否包含敏感词
        for word in sensitive_words:
            if word in news["title"] or ("summary" in news and news["summary"] and word in news["summary"]) or ("content" in news and news["content"] and word in news["content"]):
                self.logger.warning(f"新闻 '{news['title']}' 包含敏感词: {word}")
                return True
        
        return False
    
    def enrich_content(self, news):
        """丰富内容，如添加关键词、分类等"""
        # 提取关键词
        keywords = self.extract_keywords(news)
        news["keywords"] = keywords
        
        # 分类
        if "category" not in news:
            news["category"] = self.categorize_content(news)
        
        # 添加处理时间戳
        news["processed_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return news
