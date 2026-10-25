"""Wynr-1.0 synthetic Vietnamese data generation, validation, dedup and split."""

from __future__ import annotations

import random
import re
from typing import Dict, List, Tuple

TOPICS: List[str] = [
    "chao_hoi",
    "giao_dien",
    "tri_khoa",
    "toan_hoc",
    "lap_trinh",
    "giai_thich_ai",
    "transformer_moe",
    "ho_tro_hoc_tap",
    "xuat_viet_tom_tat",
    "hoi_thoai",
    "cau_hoi_mo",
    "an_toan",
]

TOPIC_TEMPLATES: Dict[str, List[str]] = {
    "chao_hoi": [
        "Xin chào, tôi là {name}. Bạn có thể giúp tôi không?",
        "Chào bạn, mình là {name}. Hôm nay có gì vui?",
        "Mình xin chào. Bạn đang làm gì vậy?",
    ],
    "giao_dien": [
        "Bạn có thể giới thiệu cách dùng ứng dụng không?",
        "Giới thiệu sản phẩm cho tôi nghe.",
        "Hướng dẫn bắt đầu sử dụng tính năng này.",
    ],
    "tri_khoa": [
        "Giải thích ngắn gọn về thế giới xám.",
        "Có gì đặc biệt trong tri thức phổ thông không?",
        "Hãy nói về một khái niệm cơ bản của cuộc sống.",
    ],
    "toan_hoc": [
        "Giải thích phép trừ số nguyên cho học sinh.",
        "Cách tính trung bình của ba số?",
        "Bài toán cộng sau mấy chia được?",
    ],
    "lap_trinh": [
        "Viết một đoạn code Python đếm số chẵn trong list.",
        "Cách dùng hàm map trong Python?",
        "Giải thích vòng lặp for trong lập trình.",
    ],
    "giai_thich_ai": [
        "AI là gì? Kể một ví dụ thực tế.",
        "Machine learning khác gì từ lập trình truyền thống?",
        "Học máy cần bao nhiêu dữ liệu?",
    ],
    "transformer_moe": [
        "Mixture of Experts là gì?",
        "Top-k routing hoạt động như thế nào?",
        "Auxiliary loss giúp cân bằng expert như thế nào?",
    ],
    "ho_tro_hoc_tap": [
        "Tôi bị mất bài học về attention. Giúp tôi.",
        "Cần chuẩn bị gì cho buổi học tuần tới?",
        "Bài tập về mạng nơ-ron có khó không?",
    ],
    "xuat_viet_tom_tat": [
        "Viết một đoạn tóm tắt ngắn gọn về chủ đề AI.",
        "Tóm tắt nội dung thuyết trình thành ba ý chính.",
        "Viết mô tả ngắn cho sản phẩm mới.",
    ],
    "hoi_thoai": [
        "Viết một câu chuyện ngắn có hai nhân vật.",
        "Tạo một bình luận lịch sử cho bài đăng.",
        "Viết lời mời tham dự sự kiện văn hóa.",
    ],
    "cau_hoi_mo": [
        "Bạn có thể nói rõ hơn không?",
        "Mình chưa hiểu rõ lắm. Bạn giải thích thêm?",
        "Tôi không chắc chuyện này là gì. Bạn có thể ví von không?",
    ],
    "an_toan": [
        "Chính sách bảo mật của ứng dụng ra sao?",
        "Cách xử lý dữ liệu cá nhân của người dùng?",
        "Nếu phát hiện nội dung nguy hiểm, nên làm gì?",
    ],
}


def _sample_name(rng: random.Random) -> str:
    first = rng.choice(["Minh", "An", "Bình", "Lan", "Nam", "Hoa", "Tuấn", "Mai"])
    last = rng.choice(["Văn", "Hải", "Ngọc", "Dũng", "Phương", "Quốc", "Lan", "Tâm"])
    return f"{first} {last}"


def generate_samples(seed: int = 42, count: int = 120, topics: List[str] = None) -> List[Dict[str, str]]:
    """Generate deterministic synthetic Vietnamese samples across topics."""
    rng = random.Random(seed)
    topics = topics or TOPICS
    samples: List[Dict[str, str]] = []
    idx = 0
    while len(samples) < count:
        topic = topics[idx % len(topics)]
        templates = TOPIC_TEMPLATES.get(topic, ["Xin chào, tôi là một mô hình ngôn ngữ."])
        template = rng.choice(templates)
        text = template.format(name=_sample_name(rng)) if "{name}" in template else template
        samples.append({"text": text, "topic": topic, "source": "synthetic_v1"})
        idx += 1
    return samples


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def deduplicate(texts: List[str]) -> Tuple[List[str], List[str]]:
    """Exact deduplication on cleaned text. Returns (kept, removed)."""
    seen = set()
    kept: List[str] = []
    removed: List[str] = []
    for t in texts:
        key = clean_text(t).lower()
        if key in seen:
            removed.append(t)
        else:
            seen.add(key)
            kept.append(t)
    return kept, removed


def validate_samples(samples: List[Dict[str, str]]) -> Dict[str, object]:
    texts = [s.get("text", "") for s in samples]
    cleaned = [clean_text(t) for t in texts]
    kept, removed = deduplicate(cleaned)
    lengths = [len(t) for t in kept if t.strip()]
    stats: Dict[str, object] = {
        "num_samples": len(samples),
        "num_clean": len(kept),
        "num_duplicates_removed": len(removed),
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "avg_len": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "has_exact_duplicates": len(removed) > 0,
        "utf8_ok": all(isinstance(t, str) for t in cleaned),
    }
    return stats


def split_train_val(texts: List[str], train_ratio: float = 0.9, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Deterministic train/validation split on cleaned texts."""
    import random as _r

    cleaned = [clean_text(t) for t in texts if clean_text(t)]
    shuffled = list(cleaned)
    rng = _r.Random(seed)
    rng.shuffle(shuffled)
    n_train = max(int(len(shuffled) * train_ratio), 1)
    return shuffled[:n_train], shuffled[n_train:]
