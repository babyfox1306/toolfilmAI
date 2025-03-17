import os
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import torch
import nltk
import networkx as nx

# Kiểm tra và tải dữ liệu NLTK nếu cần
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        print("✅ Đã tải dữ liệu NLTK punkt")
    except:
        print("⚠️ Không thể tải dữ liệu NLTK, sẽ sử dụng phương pháp tách câu đơn giản")

def clean_transcript(text):
    """Làm sạch văn bản transcript"""
    if not text:
        return ""
        
    # Xóa các khoảng trắng dư thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Xóa các từ lặp lại liên tiếp (thường là lỗi nhận dạng)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    
    # Thêm dấu chấm nếu câu kết thúc không có dấu
    text = re.sub(r'([a-zA-Z0-9])\s+([A-Z])', r'\1. \2', text)
    
    # Sửa các lỗi phổ biến trong nhận dạng tiếng Việt
    corrections = {
        'cua': 'của',
        'dc': 'được',
        'ko': 'không',
        'k ': 'không ',
        'j ': 'gì ',
        'ntn': 'như thế nào',
    }
    
    for wrong, correct in corrections.items():
        text = re.sub(r'\b' + wrong + r'\b', correct, text, flags=re.IGNORECASE)
    
    return text

def filter_irrelevant_content(text):
    """Lọc nội dung không liên quan trong văn bản"""
    if not text:
        return ""
        
    # Loại bỏ các từ, cụm từ dư thừa hoặc không liên quan
    irrelevant_phrases = [
        r"\bừm\b", r"\bà\b", r"\bậy\b", r"\bhả\b", r"\buhm\b", r"\buh\b", r"\bhmm\b",
        r"((xin )?chào (các )?(bạn|anh|chị|quý vị))",
        r"cảm ơn (các )?(bạn|anh|chị|quý vị) đã xem video",
        r"đừng quên like và subscribe",
        r"hãy bấm like và đăng ký kênh",
        r"nhớ đăng ký kênh nhé",
        r"chào mừng bạn đến với",
        r"đừng quên đăng ký kênh",
        r"hãy like và subscribe",
        r"chia sẻ video này",
        r"cảm ơn các bạn đã xem",
        r"mình sẽ gặp lại các bạn trong video sau"
    ]
    
    # Lọc bỏ những cụm từ không liên quan
    for phrase in irrelevant_phrases:
        text = re.sub(phrase, "", text, flags=re.IGNORECASE)
    
    # Xóa khoảng trắng dư thừa sau khi lọc
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tách và lọc từng câu
    try:
        sentences = sent_tokenize(text)
        relevant_sentences = []
        
        for sentence in sentences:
            # Bỏ qua các câu quá ngắn
            if len(sentence.strip()) < 10:
                continue
                
            # Bỏ qua câu không có thông tin đáng kể
            words = sentence.lower().split()
            if len(set(words)) < 3:  # Ít từ duy nhất
                continue
                
            relevant_sentences.append(sentence)
        
        return ". ".join(relevant_sentences)
    except:
        return text

def simple_sent_tokenize(text):
    """Hàm tách câu đơn giản thay thế cho NLTK sent_tokenize"""
    if not text:
        return []
        
    # Tách theo các dấu câu thường gặp
    text = text.replace('!', '.').replace('?', '.')
    # Tách thành các câu
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return sentences

def textrank_scores(similarity_matrix, d=0.85, iterations=50):
    """
    Tính điểm TextRank cho các câu dựa trên ma trận tương đồng
    d: hệ số giảm (damping factor)
    iterations: số vòng lặp để hội tụ
    """
    n = len(similarity_matrix)
    if n == 0:
        return np.array([])
    
    # Chuẩn hóa ma trận tương đồng
    for i in range(n):
        if sum(similarity_matrix[i]) != 0:
            similarity_matrix[i] = similarity_matrix[i] / sum(similarity_matrix[i])
    
    # Khởi tạo vector điểm ban đầu
    scores = np.ones(n) / n
    
    # Lặp cho đến khi hội tụ
    for _ in range(iterations):
        new_scores = (1 - d) / n + d * (similarity_matrix.T @ scores)
        # Kiểm tra hội tụ
        if np.allclose(new_scores, scores, rtol=1e-6):
            break
        scores = new_scores
    
    return scores

def summarize_transcript(transcript, ratio=0.3):
    """Tóm tắt văn bản sử dụng TF-IDF để xác định các câu quan trọng nhất"""
    print("Đang tóm tắt nội dung văn bản...")
    
    # Kiểm tra đầu vào
    if not transcript or len(transcript) < 10:
        return transcript
    
    # Làm sạch văn bản
    clean_text = clean_transcript(transcript)
    
    # Tách văn bản thành các câu
    try:
        sentences = nltk.sent_tokenize(clean_text)
    except Exception as e:
        print(f"Lỗi NLTK: {e}, sử dụng phương pháp tách câu đơn giản")
        sentences = simple_sent_tokenize(clean_text)
    
    if len(sentences) < 3:
        return clean_text
    
    # Tính điểm TF-IDF cho mỗi câu
    try:
        vectorizer = TfidfVectorizer(min_df=1)  # Đặt min_df=1 để tránh lỗi với văn bản ngắn
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
    except ValueError as e:
        print(f"Không thể phân tích TF-IDF: {e}, sử dụng phương pháp tóm tắt đơn giản...")
        num_sentences = max(1, int(len(sentences) * ratio))
        return ". ".join(sentences[:num_sentences])
    
    # Sắp xếp câu theo điểm số và chọn ra top N% câu quan trọng nhất
    num_sentences = max(1, int(len(sentences) * ratio))
    ranked_sentences = [(score, i) for i, score in enumerate(sentence_scores)]
    ranked_sentences.sort(reverse=True)
    selected_indices = [i for _, i in ranked_sentences[:num_sentences]]
    selected_indices.sort()  # Sắp xếp lại theo thứ tự gốc
    
    # Tạo văn bản tóm tắt
    summary = ". ".join([sentences[i] for i in selected_indices])
    print(f"📝 Đã tóm tắt từ {len(sentences)} câu xuống {num_sentences} câu ({ratio*100:.0f}%)")
    return summary

def enhanced_summarize_transcript(text, ratio=0.3):
    """Tóm tắt transcript với phương pháp TF-IDF và TextRank cải tiến"""
    if not text or len(text.split()) < 10:
        return text
        
    try:
        # Tách văn bản thành các câu
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            sentences = simple_sent_tokenize(text)
        
        # Nếu số câu quá ít, trả về nguyên văn bản
        if len(sentences) <= 3:
            return text
        
        # Sử dụng TF-IDF để trọng số từ
        try:
            # Thử với các tham số mặc định
            vectorizer = TfidfVectorizer(max_df=0.95, min_df=1)  # Giảm min_df xuống 1
            X = vectorizer.fit_transform(sentences)
        except ValueError as e:
            print(f"Lỗi TF-IDF: {e}")
            # Nếu vẫn lỗi, trả về phương pháp đơn giản
            return simple_summarize(text, ratio)
        
        # Tính toán similarity matrix
        similarity_matrix = cosine_similarity(X)
        
        # Áp dụng TextRank với networkx
        try:
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)
            
            # Sắp xếp câu theo điểm và chọn top N câu
            ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
            
            # Số lượng câu cần lấy
            num_sentences = max(1, int(len(sentences) * ratio))
            
            # Chọn top N câu và sắp xếp lại theo thứ tự xuất hiện gốc
            top_sentence_indices = [sentences.index(ranked_sentences[i][1]) for i in range(min(num_sentences, len(ranked_sentences)))]
            top_sentence_indices.sort()
            
            # Tạo tóm tắt
            summary = " ".join([sentences[i] for i in top_sentence_indices])
            
            return summary
        except Exception as e:
            print(f"Lỗi TextRank: {e}")
            # Fallback to simpler TF-IDF method
            return summarize_transcript(text, ratio)
    except Exception as e:
        print(f"Lỗi khi tóm tắt: {e}")
        # Fallback to simplest method
        return simple_summarize(text, ratio)

def simple_summarize(text, ratio=0.3):
    """Phương pháp tóm tắt đơn giản khi phương pháp chính gặp lỗi"""
    if not text:
        return ""
        
    sentences = text.split('.')
    num_sentences = max(1, int(len(sentences) * ratio))
    return '. '.join(sentences[:num_sentences]) + ('.' if not sentences[0].endswith('.') else '')

def summarize_transcript_with_textrank(text, ratio=0.3):
    """Tóm tắt văn bản sử dụng thuật toán TextRank"""
    if not text or len(text.split()) < 10:
        return text
        
    # Tách văn bản thành các câu
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception as e:
        print(f"Lỗi khi tách câu với NLTK: {e}")
        sentences = simple_sent_tokenize(text)
    
    if len(sentences) <= 3:
        return text
    
    # Tạo ma trận tương đồng giữa các câu
    try:
        vectorizer = TfidfVectorizer()
        sentence_vectors = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(sentence_vectors)
        
        # Áp dụng thuật toán PageRank (TextRank)
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)
        
        # Sắp xếp các câu theo điểm số
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        
        # Chọn số câu dựa theo tỉ lệ
        num_sentences = max(1, int(len(sentences) * ratio))
        
        # Lấy các câu có điểm cao và sắp xếp lại theo thứ tự gốc
        top_sentence_indices = [sentences.index(ranked_sentences[i][1]) for i in range(min(num_sentences, len(ranked_sentences)))]
        top_sentence_indices.sort()
        
        # Tạo tóm tắt
        summary = " ".join([sentences[i] for i in top_sentence_indices])
        return summary
        
    except Exception as e:
        print(f"Lỗi khi áp dụng TextRank: {e}")
        # Sử dụng phương pháp dự phòng
        return summarize_transcript(text, ratio)

def validate_summary(summary):
    """Kiểm tra và làm sạch bản tóm tắt"""
    if not summary:
        return ""
    
    # Xóa các dòng trống
    summary = re.sub(r'\n\s*\n', '\n\n', summary)
    
    # Đảm bảo các câu được viết hoa chữ cái đầu
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    capitalized_sentences = []
    
    for sentence in sentences:
        if sentence:
            # Viết hoa chữ cái đầu tiên
            capitalized = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            capitalized_sentences.append(capitalized)
    
    # Gộp lại các câu
    clean_summary = " ".join(capitalized_sentences)
    
    # Đảm bảo kết thúc bằng dấu chấm
    if clean_summary and clean_summary[-1] not in ['.', '!', '?']:
        clean_summary += '.'
    
    return clean_summary
