import numpy as np
import logging
from datetime import datetime, timedelta
import re
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
import math

class ImportanceScorer:
    def __init__(self):
        # 设置各评分因素的权重
        self.weights = {
            "source_credibility": 0.20,  # 来源可信度
            "timeliness": 0.15,          # 时效性
            "content_quality": 0.15,     # 内容质量
            "relevance": 0.20,           # 相关性
            "influence": 0.15,           # 影响力
            "engagement": 0.10,          # 互动度
            "novelty": 0.05              # 新颖性
        }
        
        # 来源可信度评分表
        self.source_credibility_scores = {
            # 官方媒体
            "人民网科技": 0.95, "新华网科技": 0.95, "中国科学报": 0.90, "科技日报": 0.90,
            # 专业科技媒体
            "36氪": 0.85, "机器之心": 0.90, "量子位": 0.85, "雷锋网": 0.80, 
            "智东西": 0.80, "AI科技评论": 0.85, "DeepTech深科技": 0.80,
            # 研究机构
            "百度研究院": 0.90, "阿里达摩院": 0.90, "腾讯AI Lab": 0.90, "智谱AI": 0.85,
            # 国外媒体
            "MIT科技评论中文版": 0.90, "哈佛商业评论中文版": 0.85, "Synced AI中文版": 0.85,
            # 公司官方
            "OpenAI博客": 0.95, "百度AI开放平台": 0.85, "智谱AI": 0.85,
            # 学术机构
            "清华大学智能产业研究院": 0.90, "北京智源人工智能研究院": 0.90,
            # 社区平台
            "AI研习社": 0.75, "GitHub": 0.80, "HuggingFace": 0.85, "arXiv": 0.90
        }
        
        # 默认来源可信度得分
        self.default_credibility_score = 0.60
        
        # AIGC关键词列表，按重要性排序
        self.aigc_keywords = [
            # 核心技术和模型
            {"term": "大模型", "weight": 1.0},
            {"term": "大语言模型", "weight": 1.0},
            {"term": "LLM", "weight": 1.0},
            {"term": "GPT-4", "weight": 0.95},
            {"term": "GPT-5", "weight": 0.95},
            {"term": "ChatGPT", "weight": 0.9},
            {"term": "Claude", "weight": 0.9},
            {"term": "Gemini", "weight": 0.9},
            {"term": "Llama", "weight": 0.9},
            {"term": "文心一言", "weight": 0.9},
            {"term": "通义千问", "weight": 0.9},
            {"term": "讯飞星火", "weight": 0.9},
            {"term": "Stable Diffusion", "weight": 0.85},
            {"term": "MidJourney", "weight": 0.85},
            {"term": "DALL-E", "weight": 0.85},
            {"term": "Sora", "weight": 0.9},
            
            # 基础概念
            {"term": "AIGC", "weight": 0.8},
            {"term": "生成式AI", "weight": 0.8},
            {"term": "生成式人工智能", "weight": 0.8},
            {"term": "生成式模型", "weight": 0.8},
            {"term": "AI生成内容", "weight": 0.75},
            
            # 技术术语
            {"term": "Transformer", "weight": 0.7},
            {"term": "注意力机制", "weight": 0.7},
            {"term": "自回归", "weight": 0.7},
            {"term": "自监督学习", "weight": 0.7},
            {"term": "预训练", "weight": 0.7},
            {"term": "微调", "weight": 0.7},
            {"term": "多模态", "weight": 0.75},
            {"term": "跨模态", "weight": 0.75},
            {"term": "扩散模型", "weight": 0.75},
            {"term": "LoRA", "weight": 0.7},
            {"term": "PEFT", "weight": 0.7},
            {"term": "QLoRA", "weight": 0.7},
            {"term": "RAG", "weight": 0.8},
            {"term": "检索增强生成", "weight": 0.8},
            
            # 应用领域
            {"term": "AI绘画", "weight": 0.65},
            {"term": "AI写作", "weight": 0.65},
            {"term": "AI音乐", "weight": 0.65},
            {"term": "AI视频", "weight": 0.7},
            {"term": "AI编程", "weight": 0.7},
            {"term": "AI代码", "weight": 0.7},
            
            # 公司和研究机构
            {"term": "OpenAI", "weight": 0.85},
            {"term": "Anthropic", "weight": 0.8},
            {"term": "Google DeepMind", "weight": 0.8},
            {"term": "百度", "weight": 0.75},
            {"term": "阿里", "weight": 0.75},
            {"term": "腾讯", "weight": 0.75},
            {"term": "智谱AI", "weight": 0.75},
            
            # 相关概念
            {"term": "提示工程", "weight": 0.7},
            {"term": "Prompt", "weight": 0.7},
            {"term": "幻觉", "weight": 0.65},
            {"term": "伦理", "weight": 0.6},
            {"term": "版权", "weight": 0.6},
            {"term": "监管", "weight": 0.65}
        ]
        
        # 将关键词转换为字典，方便查找
        self.keyword_weights = {item["term"]: item["weight"] for item in self.aigc_keywords}
        
        # 设置日志记录器
        self.logger = self._setup_logger()
        
        # 初始化TF-IDF向量化器
        self.vectorizer = None
        self.corpus_vectors = None
        self.corpus_news = []
    
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger("ImportanceScorer")
        logger.setLevel(logging.INFO)
        
        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 创建文件处理器
        fh = logging.FileHandler("importance_scorer.log")
        fh.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        # 添加处理器到记录器
        logger.addHandler(ch)
        logger.addHandler(fh)
        
        return logger
    
    def score_news(self, news):
        """计算新闻的重要性得分"""
        scores = {}
        
        # 1. 来源可信度评分
        scores["source_credibility"] = self._score_source_credibility(news)
        
        # 2. 时效性评分
        scores["timeliness"] = self._score_timeliness(news)
        
        # 3. 内容质量评分
        scores["content_quality"] = self._score_content_quality(news)
        
        # 4. 相关性评分
        scores["relevance"] = self._score_relevance(news)
        
        # 5. 影响力评分
        scores["influence"] = self._score_influence(news)
        
        # 6. 互动度评分
        scores["engagement"] = self._score_engagement(news)
        
        # 7. 新颖性评分
        scores["novelty"] = self._score_novelty(news)
        
        # 计算加权总分
        total_score = 0
        for factor, score in scores.items():
            total_score += score * self.weights[factor]
        
        # 确保分数在0-100范围内
        total_score = min(max(total_score * 100, 0), 100)
        
        # 记录评分结果
        self.logger.info(f"新闻 '{news['title']}' 评分: {total_score:.2f}")
        self.logger.debug(f"详细评分: {scores}")
        
        # 添加评分到新闻对象
        news["importance_score"] = total_score
        news["factor_scores"] = scores
        
        return total_score
    
    def _score_source_credibility(self, news):
        """评估新闻来源的可信度"""
        # 获取新闻来源
        source = news.get("source", "")
        
        # 如果是合并的新闻，考虑多个来源
        if "sources" in news and isinstance(news["sources"], list):
            # 计算所有来源的平均可信度
            source_scores = []
            for src in news["sources"]:
                if src in self.source_credibility_scores:
                    source_scores.append(self.source_credibility_scores[src])
                else:
                    source_scores.append(self.default_credibility_score)
            
            if source_scores:
                # 使用最高可信度作为基础，每增加一个来源提高5%，但最高不超过0.95
                base_score = max(source_scores)
                bonus = min(0.05 * (len(source_scores) - 1), 0.15)
                return min(base_score + bonus, 0.95)
            else:
                return self.default_credibility_score
        
        # 单一来源
        if source in self.source_credibility_scores:
            return self.source_credibility_scores[source]
        else:
            # 尝试部分匹配
            for known_source, score in self.source_credibility_scores.items():
                if known_source in source or source in known_source:
                    return score * 0.9  # 部分匹配给予90%的分数
            
            return self.default_credibility_score
    
    def _score_timeliness(self, news):
        """评估新闻的时效性"""
        # 获取当前时间
        current_time = datetime.now()
        
        # 获取新闻发布时间
        published_time = news.get("published", None)
        
        # 如果没有发布时间，给予中等时效性分数
        if published_time is None or not isinstance(published_time, datetime):
            return 0.5
        
        # 计算时间差（小时）
        time_diff_hours = (current_time - published_time).total_seconds() / 3600
        
        # 时效性评分函数：24小时内为高分，之后逐渐降低
        if time_diff_hours <= 6:
            # 6小时内，满分
            return 1.0
        elif time_diff_hours <= 24:
            # 6-24小时，线性降低到0.8
            return 1.0 - 0.2 * (time_diff_hours - 6) / 18
        elif time_diff_hours <= 72:
            # 24-72小时，线性降低到0.6
            return 0.8 - 0.2 * (time_diff_hours - 24) / 48
        elif time_diff_hours <= 168:
            # 72小时-7天，线性降低到0.4
            return 0.6 - 0.2 * (time_diff_hours - 72) / 96
        else:
            # 7天以上，固定为0.3
            return 0.3
    
    def _score_content_quality(self, news):
        """评估新闻内容的质量"""
        # 获取新闻内容
        title = news.get("title", "")
        summary = news.get("summary", "")
        content = news.get("content", "")
        
        # 计算内容长度
        title_length = len(title)
        summary_length = len(summary)
        content_length = len(content)
        
        # 1. 长度评分
        length_score = 0
        
        # 标题长度评分（10-30字为佳）
        if 10 <= title_length <= 30:
            length_score += 0.2
        elif 5 <= title_length < 10 or 30 < title_length <= 50:
            length_score += 0.1
        else:
            length_score += 0.05
        
        # 摘要长度评分（50-200字为佳）
        if 100 <= summary_length <= 300:
            length_score += 0.3
        elif 50 <= summary_length < 100 or 300 < summary_length <= 500:
            length_score += 0.2
        elif summary_length > 0:
            length_score += 0.1
        
        # 内容长度评分（500字以上为佳）
        if content_length >= 1000:
            length_score += 0.5
        elif content_length >= 500:
            length_score += 0.3
        elif content_length >= 200:
            length_score += 0.2
        elif content_length > 0:
            length_score += 0.1
        
        # 2. 信息密度评分
        density_score = 0
        
        # 提取关键词
        all_text = title + " " + summary + " " + content
        keywords = jieba.analyse.extract_tags(all_text, topK=20)
        
        # 计算关键词密度
        if all_text:
            keyword_density = len(keywords) / (len(all_text) / 100)  # 每100字的关键词数量
            
            if keyword_density >= 2.0:
                density_score = 1.0
            elif keyword_density >= 1.5:
                density_score = 0.8
            elif keyword_density >= 1.0:
                density_score = 0.6
            elif keyword_density >= 0.5:
                density_score = 0.4
            else:
                density_score = 0.2
        
        # 3. 结构评分
        structure_score = 0
        
        # 检查是否有标题、摘要和内容
        if title and summary and content:
            structure_score = 1.0
        elif title and content:
            structure_score = 0.8
        elif title and summary:
            structure_score = 0.7
        elif title:
            structure_score = 0.4
        
        # 4. 语言质量评分（简单启发式方法）
        language_score = 0
        
        # 检查标点符号使用
        punctuation_ratio = len(re.findall(r'[，。！？；：""''（）【】《》]', all_text)) / (len(all_text) + 1)
        if 0.05 <= punctuation_ratio <= 0.15:
            language_score += 0.5
        elif 0.02 <= punctuation_ratio < 0.05 or 0.15 < punctuation_ratio <= 0.2:
            language_score += 0.3
        else:
            language_score += 0.1
        
        # 检查段落结构（简单估计）
        if content:
            paragraphs = content.split('\n')
            if len(paragraphs) >= 3:
                language_score += 0.5
            elif len(paragraphs) >= 2:
                language_score += 0.3
            else:
                language_score += 0.1
        
        # 综合各项评分
        quality_score = 0.3 * length_score + 0.3 * density_score + 0.2 * structure_score + 0.2 * language_score
        
        return quality_score
    
    def _score_relevance(self, news):
        """评估新闻与AIGC主题的相关性"""
        # 获取新闻文本
        title = news.get("title", "")
        summary = news.get("summary", "")
        content = news.get("content", "")
        all_text = title + " " + summary + " " + content
        
        # 计算AIGC关键词出现的频率和权重
        relevance_score = 0
        matched_keywords = []
        
        for keyword, weight in self.keyword_weights.items():
            # 计算关键词在文本中出现的次数
            count = all_text.lower().count(keyword.lower())
            if count > 0:
                matched_keywords.append((keyword, count, weight))
        
        # 如果没有匹配到关键词，给予低分
        if not matched_keywords:
            return 0.3
        
        # 计算加权得分
        total_weight = 0
        weighted_count = 0
        
        for keyword, count, weight in matched_keywords:
            # 对出现次数进行对数缩放，避免单一关键词重复出现导致得分过高
            log_count = 1 + math.log(count) if count > 1 else 1
            weighted_count += log_count * weight
            total_weight += weight
        
        # 计算最终相关性得分，确保在0-1范围内
        if total_weight > 0:
            # 标准化得分，最多匹配10个加权关键词视为满分
            normalized_score = min(weighted_count / 10, 1.0)
            
            # 根据匹配关键词的数量给予额外加分
            keyword_count_bonus = min(len(matched_keywords) / 15, 0.3)
            
            # 如果标题中包含高权重关键词，给予额外加分
            title_bonus = 0
            for keyword, _, weight in matched_keywords:
                if keyword.lower() in title.lower() and weight >= 0.8:
                    title_bonus = 0.1
                    break
            
            relevance_score = normalized_score * 0.6 + keyword_count_bonus + title_bonus
            
            # 确保得分不超过1
            relevance_score = min(relevance_score, 1.0)
        else:
            relevance_score = 0.3
        
        return relevance_score
    
     def _score_influence(self, news):
        """评估新闻的影响力"""
        # 影响力评分考虑以下因素：
        # 1. 新闻来源的影响力
        # 2. 新闻内容涉及的主体的重要性
        # 3. 新闻事件的规模和影响范围
        
        # 1. 来源影响力（部分基于来源可信度）
        source_influence = self._score_source_credibility(news) * 0.8
        
        # 2. 主体重要性
        entity_importance = 0
        
        # 检查标题和内容中是否提到重要实体
        important_entities = {
            # 重要公司
            "OpenAI": 1.0, "Google": 0.9, "微软": 0.9, "Microsoft": 0.9, "百度": 0.8, 
            "阿里巴巴": 0.8, "腾讯": 0.8, "华为": 0.8, "Meta": 0.8, "Facebook": 0.7,
            "Anthropic": 0.8, "Amazon": 0.8, "亚马逊": 0.8, "Apple": 0.8, "苹果": 0.8,
            "字节跳动": 0.7, "商汤": 0.7, "旷视": 0.7, "智谱AI": 0.7, "科大讯飞": 0.7,
            
            # 重要模型和产品
            "GPT-4": 1.0, "GPT-5": 1.0, "ChatGPT": 0.9, "Claude": 0.8, "Gemini": 0.8,
            "文心一言": 0.8, "通义千问": 0.8, "讯飞星火": 0.8, "Llama": 0.8, "Llama 3": 0.9,
            "Stable Diffusion": 0.7, "DALL-E": 0.7, "MidJourney": 0.7, "Sora": 0.9,
            
            # 重要人物
            "Sam Altman": 0.9, "Demis Hassabis": 0.8, "李飞飞": 0.8, "李开复": 0.8,
            "吴恩达": 0.8, "Andrew Ng": 0.8, "Yann LeCun": 0.8, "Geoffrey Hinton": 0.8,
            "Yoshua Bengio": 0.8, "Robin Li": 0.7, "李彦宏": 0.7, "马化腾": 0.7, "任正非": 0.7,
            
            # 重要机构
            "清华大学": 0.8, "北京大学": 0.8, "中国科学院": 0.8, "中国工程院": 0.8,
            "智源研究院": 0.7, "MIT": 0.8, "Stanford": 0.8, "斯坦福": 0.8, "哈佛": 0.7,
            "Harvard": 0.7, "DeepMind": 0.8, "FAIR": 0.7, "OpenAI研究院": 0.8
        }
        
        title = news.get("title", "")
        content = news.get("content", "")
        all_text = title + " " + content
        
        matched_entities = []
        for entity, importance in important_entities.items():
            if entity in all_text:
                matched_entities.append(importance)
        
        if matched_entities:
            # 使用最重要实体的重要性作为基础分数
            entity_importance = max(matched_entities)
            
            # 根据匹配到的实体数量给予额外加分
            entity_count_bonus = min(0.1 * (len(matched_entities) - 1), 0.3)
            entity_importance = min(entity_importance + entity_count_bonus, 1.0)
        else:
            entity_importance = 0.4  # 默认值
        
        # 3. 事件规模和影响
        event_scale = 0.5  # 默认中等规模
        
        # 检查是否涉及重大事件的关键词
        major_event_keywords = [
            "重大突破", "重大进展", "重大发现", "重大成果", "重大创新", "重大更新", "重大升级",
            "行业首创", "全球首个", "全球首款", "全球领先", "国内首个", "国内首款", "国内领先",
            "颠覆性", "革命性", "里程碑", "新纪元", "新时代", "新篇章", "新高度", "新阶段",
            "重磅", "爆款", "爆发式增长", "指数级增长", "爆炸式增长", "井喷式增长",
            "亿级用户", "千万级用户", "百万级用户", "融资", "IPO", "上市", "收购", "并购",
            "战略合作", "战略伙伴", "战略投资", "独家", "专访", "深度解析", "深度分析"
        ]
        
        for keyword in major_event_keywords:
            if keyword in all_text:
                event_scale = 0.8
                break
        
        # 检查是否涉及特别重大事件
        critical_event_keywords = [
            "重大突破性进展", "划时代", "改变世界", "改变人类", "人类历史上首次",
            "百亿级", "千亿级", "万亿级", "百亿美元", "千亿美元", "万亿美元",
            "重大安全漏洞", "重大安全事件", "重大伦理问题", "重大政策变化", "重大监管措施"
        ]
        
        for keyword in critical_event_keywords:
            if keyword in all_text:
                event_scale = 1.0
                break
        
        # 综合各项评分
        influence_score = 0.3 * source_influence + 0.4 * entity_importance + 0.3 * event_scale
        
        return influence_score
    
    def _score_engagement(self, news):
        """评估新闻的互动度"""
        # 互动度评分考虑以下因素：
        # 1. 新闻的阅读量/点击量
        # 2. 评论数
        # 3. 分享/转发数
        # 4. 点赞数
        
        # 获取互动数据
        views = news.get("views", 0)
        comments = news.get("comments", 0)
        shares = news.get("shares", 0)
        likes = news.get("likes", 0)
        
        # 如果没有互动数据，给予中等分数
        if views == 0 and comments == 0 and shares == 0 and likes == 0:
            # 检查是否是合并的新闻，合并的新闻可能有更高的互动潜力
            if news.get("is_merged", False) and news.get("merged_count", 0) > 1:
                return 0.7
            else:
                return 0.5
        
        # 计算互动度得分
        engagement_score = 0
        
        # 阅读量/点击量评分
        if views > 10000:
            views_score = 1.0
        elif views > 5000:
            views_score = 0.8
        elif views > 1000:
            views_score = 0.6
        elif views > 100:
            views_score = 0.4
        else:
            views_score = 0.2
        
        # 评论数评分
        if comments > 100:
            comments_score = 1.0
        elif comments > 50:
            comments_score = 0.8
        elif comments > 20:
            comments_score = 0.6
        elif comments > 5:
            comments_score = 0.4
        else:
            comments_score = 0.2
        
        # 分享/转发数评分
        if shares > 100:
            shares_score = 1.0
        elif shares > 50:
            shares_score = 0.8
        elif shares > 20:
            shares_score = 0.6
        elif shares > 5:
            shares_score = 0.4
        else:
            shares_score = 0.2
        
        # 点赞数评分
        if likes > 500:
            likes_score = 1.0
        elif likes > 200:
            likes_score = 0.8
        elif likes > 50:
            likes_score = 0.6
        elif likes > 10:
            likes_score = 0.4
        else:
            likes_score = 0.2
        
        # 综合各项评分，权重可以根据实际情况调整
        if views > 0 or comments > 0 or shares > 0 or likes > 0:
            # 计算有效指标的数量
            valid_metrics = sum(1 for x in [views, comments, shares, likes] if x > 0)
            
            # 计算总分
            total_score = 0
            if views > 0:
                total_score += views_score * 0.4
            if comments > 0:
                total_score += comments_score * 0.3
            if shares > 0:
                total_score += shares_score * 0.2
            if likes > 0:
                total_score += likes_score * 0.1
            
            # 根据有效指标数量调整最终分数
            engagement_score = total_score * (valid_metrics / 4)
        else:
            engagement_score = 0.5
        
        return engagement_score
    
    def _score_novelty(self, news):
        """评估新闻的新颖性"""
        # 新颖性评分考虑以下因素：
        # 1. 与已有新闻的相似度
        # 2. 内容中的创新点
        # 3. 是否包含新的信息或观点
        
        # 1. 与已有新闻的相似度评分
        similarity_score = 0
        
        # 如果是更新类新闻，新颖性较低
        if news.get("is_update", False):
            similarity_score = 0.3
        # 如果是合并的新闻，新颖性中等
        elif news.get("is_merged", False):
            similarity_score = 0.5
        else:
            # 如果有足够的语料库，计算与已有新闻的相似度
            if len(self.corpus_news) > 10:
                # 准备新闻文本
                text = news.get("title", "") + " " + news.get("summary", "")
                
                try:
                    # 如果向量化器未初始化，初始化它
                    if self.vectorizer is None:
                        self.vectorizer = TfidfVectorizer(analyzer='word', tokenizer=lambda x: jieba.cut(x))
                        corpus_texts = [n.get("title", "") + " " + n.get("summary", "") for n in self.corpus_news]
                        self.corpus_vectors = self.vectorizer.fit_transform(corpus_texts)
                    
                    # 计算新闻向量
                    news_vector = self.vectorizer.transform([text])
                    
                    # 计算与语料库中所有新闻的相似度
                    similarities = cosine_similarity(news_vector, self.corpus_vectors)[0]
                    
                    # 取最大相似度
                    max_similarity = max(similarities) if len(similarities) > 0 else 0
                    
                    # 新颖性与相似度成反比
                    similarity_score = 1.0 - max_similarity
                except:
                    # 如果向量化失败，给予中等分数
                    similarity_score = 0.5
            else:
                # 如果语料库太小，给予较高的新颖性分数
                similarity_score = 0.8
        
        # 2. 内容创新点评分
        innovation_score = 0
        
        # 检查是否包含创新相关关键词
        innovation_keywords = [
            "首次", "首创", "首个", "首款", "独创", "原创", "创新", "突破", "新技术", "新方法",
            "新算法", "新模型", "新架构", "新应用", "新场景", "新功能", "新特性", "新版本",
            "升级", "改进", "优化", "提升", "增强", "扩展", "拓展", "深化", "革新", "变革",
            "颠覆", "重构", "重塑", "重新定义", "开创", "开辟", "开拓", "探索", "实验", "尝试"
        ]
        
        title = news.get("title", "")
        summary = news.get("summary", "")
        content = news.get("content", "")
        all_text = title + " " + summary + " " + content
        
        innovation_count = 0
        for keyword in innovation_keywords:
            if keyword in all_text:
                innovation_count += 1
        
        # 根据创新关键词数量计算创新分数
        if innovation_count >= 5:
            innovation_score = 1.0
        elif innovation_count >= 3:
            innovation_score = 0.8
        elif innovation_count >= 1:
            innovation_score = 0.6
        else:
            innovation_score = 0.4
        
        # 3. 新信息或观点评分
        information_score = 0
        
        # 检查是否包含新信息或观点相关关键词
        new_info_keywords = [
            "最新", "最近", "刚刚", "今日", "今天", "昨日", "昨天", "本周", "本月", "近期",
            "最新消息", "最新进展", "最新动态", "最新报道", "最新研究", "最新成果", "最新发现",
            "独家", "独家报道", "独家消息", "独家揭秘", "独家专访", "独家解析", "独家分析",
            "深度", "深度报道", "深度分析", "深度解析", "深度揭秘", "深度专访", "深度观察",
            "观点", "视角", "见解", "洞察", "解读", "分析", "评论", "评价", "点评", "解析"
        ]
        
        new_info_count = 0
        for keyword in new_info_keywords:
            if keyword in all_text:
                new_info_count += 1
        
        # 根据新信息关键词数量计算新信息分数
        if new_info_count >= 5:
            information_score = 1.0
        elif new_info_count >= 3:
            information_score = 0.8
        elif new_info_count >= 1:
            information_score = 0.6
        else:
            information_score = 0.4
        
        # 综合各项评分
        novelty_score = 0.4 * similarity_score + 0.3 * innovation_score + 0.3 * information_score
        
        # 更新语料库
        self._update_corpus(news)
        
        return novelty_score
    
    def _update_corpus(self, news):
        """更新新闻语料库"""
        # 将新闻添加到语料库
        self.corpus_news.append(news)
        
        # 限制语料库大小，保留最新的100条新闻
        if len(self.corpus_news) > 100:
            self.corpus_news = self.corpus_news[-100:]
            
            # 重置向量化器，下次使用时会重新计算
            self.vectorizer = None
            self.corpus_vectors = None
    
    def score_news_batch(self, news_list):
        """批量评分新闻"""
        scored_news = []
        
        for news in news_list:
            score = self.score_news(news)
            scored_news.append((news, score))
        
        # 按评分降序排序
        scored_news.sort(key=lambda x: x[1], reverse=True)
        
        return scored_news
    
    def get_top_news(self, news_list, top_n=10):
        """获取评分最高的前N条新闻"""
        # 批量评分
        scored_news = self.score_news_batch(news_list)
        
        # 返回前N条
        return [news for news, score in scored_news[:top_n]]
    
    def analyze_score_factors(self, news):
        """分析影响新闻评分的因素"""
        # 计算评分
        self.score_news(news)
        
        # 获取各因素的得分
        factor_scores = news.get("factor_scores", {})
        
        # 分析主要影响因素
        if not factor_scores:
            return "无法分析评分因素，未找到详细评分数据。"
        
        # 按得分降序排序
        sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 生成分析报告
        report = f"新闻 '{news['title']}' 评分分析：\n"
        report += f"总评分: {news.get('importance_score', 0):.2f}/100\n\n"
        report += "主要影响因素:\n"
        
        for factor, score in sorted_factors:
            factor_name = {
                "source_credibility": "来源可信度",
                "timeliness": "时效性",
                "content_quality": "内容质量",
                "relevance": "相关性",
                "influence": "影响力",
                "engagement": "互动度",
                "novelty": "新颖性"
            }.get(factor, factor)
            
            report += f"- {factor_name}: {score:.2f} (权重: {self.weights.get(factor, 0):.2f})\n"
        
        # 提供改进建议
        report += "\n改进建议:\n"
        
        # 找出得分较低的因素
        low_factors = [factor for factor, score in factor_scores.items() if score < 0.6]
        
        if low_factors:
            for factor in low_factors:
                if factor == "source_credibility":
                    report += "- 来源可信度较低，考虑使用更权威的信息来源或多来源交叉验证\n"
                elif factor == "timeliness":
                    report += "- 时效性不足，考虑更及时地报道最新进展或更新内容\n"
                elif factor == "content_quality":
                    report += "- 内容质量有待提高，可以增加内容深度、丰富细节或改进结构\n"
                elif factor == "relevance":
                    report += "- 与AIGC主题相关性不够强，可以更聚焦于核心技术和应用\n"
                elif factor == "influence":
                    report += "- 影响力不足，可以关注更重要的实体或更具影响力的事件\n"
                elif factor == "engagement":
                    report += "- 互动度较低，可以增加内容的吸引力或提高用户参与度\n"
                elif factor == "novelty":
                    report += "- 新颖性不足，可以提供更独特的视角或新的信息点\n"
        else:
            report += "- 各项指标表现良好，建议保持当前内容质量和报道方向\n"
        
        return report
