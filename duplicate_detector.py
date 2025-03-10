import hashlib
import difflib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime, timedelta
import jieba
import re
from collections import defaultdict

class DuplicateDetector:
    def __init__(self):
        self.news_database = {}  # 存储已处理的新闻
        self.similarity_threshold = 0.8  # 相似度阈值
        self.time_window = timedelta(days=7)  # 时间窗口，只比较7天内的新闻
        self.vectorizer = TfidfVectorizer(analyzer='word', tokenizer=self._tokenize)
        self.vectors = None  # 存储新闻向量
        self.news_ids = []  # 存储新闻ID，与向量对应
        
        # 设置日志记录器
        self.logger = self._setup_logger()
        
        # 加载停用词
        self.stopwords = self._load_stopwords()
        
        # 聚类相关参数
        self.cluster_similarity_threshold = 0.7  # 聚类相似度阈值
        self.clusters = defaultdict(list)  # 存储新闻聚类
        self.cluster_vectors = {}  # 存储聚类中心向量
    
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger("DuplicateDetector")
        logger.setLevel(logging.INFO)
        
        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 创建文件处理器
        fh = logging.FileHandler("duplicate_detector.log")
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
    
    def _tokenize(self, text):
        """分词并去除停用词"""
        # 使用jieba进行分词
        words = jieba.cut(text)
        # 去除停用词
        return [word for word in words if word not in self.stopwords and len(word.strip()) > 1]
    
    def _compute_hash(self, text):
        """计算文本的哈希值"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _clean_text(self, text):
        """清理文本，去除标点符号、空格等"""
        # 去除HTML标签
        clean_text = re.sub(r'<.*?>', '', text)
        # 去除URL
        clean_text = re.sub(r'https?://\S+|www\.\S+', '', clean_text)
        # 去除标点符号
        clean_text = re.sub(r'[^\w\s]', '', clean_text)
        # 去除多余空格
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return clean_text
    
    def compute_similarity(self, news1, news2):
        """计算两条新闻的相似度"""
        # 准备新闻文本
        text1 = news1["title"]
        if "summary" in news1 and news1["summary"]:
            text1 += " " + news1["summary"]
        
        text2 = news2["title"]
        if "summary" in news2 and news2["summary"]:
            text2 += " " + news2["summary"]
        
        # 清理文本
        text1 = self._clean_text(text1)
        text2 = self._clean_text(text2)
        
        # 计算标题相似度
        title_similarity = difflib.SequenceMatcher(None, news1["title"], news2["title"]).ratio()
        
        # 如果标题完全相同，直接返回1.0
        if title_similarity > 0.9:
            return 1.0
        
        # 计算文本相似度
        try:
            # 使用TF-IDF向量计算余弦相似度
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # 综合考虑标题相似度和文本相似度
            combined_similarity = 0.6 * title_similarity + 0.4 * cosine_sim
            
            return combined_similarity
        except:
            # 如果向量化失败，退回到序列匹配
            return difflib.SequenceMatcher(None, text1, text2).ratio()
    
    def is_duplicate(self, news):
        """判断是否是重复新闻"""
        # 准备新闻文本
        text = news["title"]
        if "summary" in news and news["summary"]:
            text += " " + news["summary"]
        
        # 计算哈希值
        news_hash = self._compute_hash(text)
        
        # 检查是否有完全相同的哈希值
        if news_hash in self.news_database:
            self.logger.info(f"发现完全重复的新闻: {news['title']}")
            return True
        
        # 获取当前时间
        current_time = datetime.now()
        
        # 检查相似度
        for news_id, stored_news in list(self.news_database.items()):
            # 只比较时间窗口内的新闻
            if "published" in stored_news and isinstance(stored_news["published"], datetime):
                time_diff = current_time - stored_news["published"]
                if time_diff > self.time_window:
                    continue
            
            # 计算相似度
            similarity = self.compute_similarity(news, stored_news)
            
            # 如果相似度超过阈值，认为是重复新闻
            if similarity >= self.similarity_threshold:
                self.logger.info(f"发现相似新闻: '{news['title']}' 与 '{stored_news['title']}' (相似度: {similarity:.4f})")
                return True
        
        # 如果没有找到重复或相似的新闻，将当前新闻添加到数据库
        self.news_database[news_hash] = news
        
        # 更新向量
        self._update_vectors(news, news_hash)
        
        return False

    def _update_vectors(self, news, news_id):
        """更新新闻向量"""
        # 准备新闻文本
        text = news["title"]
        if "summary" in news and news["summary"]:
            text += " " + news["summary"]
        
        # 清理文本
        text = self._clean_text(text)
        
        # 如果是第一条新闻，初始化向量化器
        if self.vectors is None:
            self.vectors = self.vectorizer.fit_transform([text])
            self.news_ids = [news_id]
        else:
            # 重新训练向量化器以包含新词汇
            all_texts = [self._clean_text(self.news_database[nid]["title"] + 
                        (" " + self.news_database[nid]["summary"] if "summary" in self.news_database[nid] and self.news_database[nid]["summary"] else ""))
                        for nid in self.news_ids]
            all_texts.append(text)
            
            self.vectors = self.vectorizer.fit_transform(all_texts)
            self.news_ids.append(news_id)
    
    def cluster_news(self, news_list):
        """对新闻进行聚类"""
        self.logger.info(f"开始对 {len(news_list)} 条新闻进行聚类")
        
        # 如果新闻数量太少，不进行聚类
        if len(news_list) < 3:
            self.logger.info("新闻数量太少，不进行聚类")
            return {i: [news] for i, news in enumerate(news_list)}
        
        # 准备新闻文本
        texts = []
        for news in news_list:
            text = news["title"]
            if "summary" in news and news["summary"]:
                text += " " + news["summary"]
            texts.append(self._clean_text(text))
        
        # 计算TF-IDF向量
        try:
            vectors = self.vectorizer.fit_transform(texts)
        except:
            # 如果向量化失败，可能是因为词汇表变化，重新初始化向量化器
            self.vectorizer = TfidfVectorizer(analyzer='word', tokenizer=self._tokenize)
            vectors = self.vectorizer.fit_transform(texts)
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(vectors)
        
        # 使用层次聚类
        clusters = {}
        assigned = set()
        
        # 首先找出相似度高的对
        for i in range(len(news_list)):
            if i in assigned:
                continue
                
            # 创建新簇
            cluster_id = len(clusters)
            clusters[cluster_id] = [news_list[i]]
            assigned.add(i)
            
            # 找出与当前新闻相似度高的其他新闻
            for j in range(i+1, len(news_list)):
                if j in assigned:
                    continue
                    
                if similarity_matrix[i, j] >= self.cluster_similarity_threshold:
                    clusters[cluster_id].append(news_list[j])
                    assigned.add(j)
        
        # 处理未分配的新闻
        for i in range(len(news_list)):
            if i not in assigned:
                # 创建单条新闻的簇
                cluster_id = len(clusters)
                clusters[cluster_id] = [news_list[i]]
        
        self.logger.info(f"聚类完成，共形成 {len(clusters)} 个簇")
        
        # 保存聚类结果
        self.clusters = clusters
        
        return clusters
    
    def merge_similar_news(self, news_list):
        """合并相似新闻，生成摘要"""
        # 首先进行聚类
        clusters = self.cluster_news(news_list)
        
        merged_news = []
        
        # 处理每个簇
        for cluster_id, cluster_news in clusters.items():
            # 如果簇中只有一条新闻，直接添加
            if len(cluster_news) == 1:
                merged_news.append(cluster_news[0])
                continue
            
            # 如果簇中有多条新闻，合并它们
            # 按发布时间排序
            sorted_news = sorted(cluster_news, 
                                key=lambda x: x["published"] if "published" in x and isinstance(x["published"], datetime) else datetime.min)
            
            # 使用最新的新闻作为基础
            base_news = sorted_news[-1].copy()
            
            # 合并标题，使用最新的新闻标题
            base_news["title"] = sorted_news[-1]["title"]
            
            # 合并摘要
            summaries = [news["summary"] for news in sorted_news if "summary" in news and news["summary"]]
            if summaries:
                # 选择最长的摘要
                base_news["summary"] = max(summaries, key=len)
            
            # 合并内容
            contents = [news["content"] for news in sorted_news if "content" in news and news["content"]]
            if contents:
                # 选择最长的内容
                base_news["content"] = max(contents, key=len)
            
            # 合并来源
            sources = set([news["source"] for news in sorted_news if "source" in news])
            base_news["sources"] = list(sources)
            
            # 合并关键词
            all_keywords = []
            for news in sorted_news:
                if "keywords" in news and news["keywords"]:
                    all_keywords.extend(news["keywords"])
            
            # 去重并取频率最高的关键词
            keyword_freq = {}
            for keyword in all_keywords:
                if keyword in keyword_freq:
                    keyword_freq[keyword] += 1
                else:
                    keyword_freq[keyword] = 1
            
            # 选择频率最高的5个关键词
            top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            base_news["keywords"] = [k for k, v in top_keywords]
            
            # 添加相关新闻
            base_news["related_news"] = [{"title": news["title"], "link": news["link"], "source": news["source"]} 
                                         for news in sorted_news if news != sorted_news[-1]]
            
            # 添加聚合信息
            base_news["is_merged"] = True
            base_news["merged_count"] = len(cluster_news)
            
            # 添加到结果列表
            merged_news.append(base_news)
        
        self.logger.info(f"合并完成，从 {len(news_list)} 条新闻合并为 {len(merged_news)} 条")
        
        return merged_news
    
    def detect_updates(self, news, time_window=None):
        """检测是否是对已有新闻的更新"""
        if time_window is None:
            time_window = self.time_window
        
        # 获取当前时间
        current_time = datetime.now()
        
        # 检查是否是对已有新闻的更新
        for news_id, stored_news in list(self.news_database.items()):
            # 只比较时间窗口内的新闻
            if "published" in stored_news and isinstance(stored_news["published"], datetime):
                time_diff = current_time - stored_news["published"]
                if time_diff > time_window:
                    continue
            
            # 计算标题相似度
            title_similarity = difflib.SequenceMatcher(None, news["title"], stored_news["title"]).ratio()
            
            # 如果标题相似度高，但不是完全相同
            if title_similarity >= 0.7 and title_similarity < 0.9:
                # 检查是否包含更新关键词
                update_keywords = ["更新", "最新", "进展", "后续", "补充", "修正", "update", "latest"]
                for keyword in update_keywords:
                    if keyword in news["title"] and keyword not in stored_news["title"]:
                        self.logger.info(f"发现对已有新闻的更新: '{news['title']}' 更新自 '{stored_news['title']}'")
                        
                        # 添加更新信息
                        news["is_update"] = True
                        news["update_from"] = stored_news["title"]
                        news["update_from_link"] = stored_news["link"]
                        
                        return True
        
        return False
    
    def process_news_batch(self, news_list):
        """处理一批新闻，去重并聚合"""
        # 去重
        unique_news = []
        for news in news_list:
            if not self.is_duplicate(news):
                unique_news.append(news)
        
        self.logger.info(f"去重后剩余 {len(unique_news)}/{len(news_list)} 条新闻")
        
        # 检测更新
        for news in unique_news:
            self.detect_updates(news)
        
        # 聚合相似新闻
        merged_news = self.merge_similar_news(unique_news)
        
        return merged_news

