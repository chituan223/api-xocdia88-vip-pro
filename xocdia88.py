import requests
import time
import threading
import math
from flask import Flask, jsonify
from collections import deque, Counter

app = Flask(__name__)

# ================= CONFIG =================
API_URL = "https://taixiumd5.system32-cloudfare-356783752985678522.monster/api/md5luckydice/GetSoiCau"
MIN_PHIEN = 15
MAX_PHIEN = 20

# ================= DATA =================
history_tx = deque(maxlen=MAX_PHIEN)
history_pt = deque(maxlen=MAX_PHIEN)
history_id = deque(maxlen=MAX_PHIEN)
history_dice = deque(maxlen=MAX_PHIEN)

stats = {"tong": 0, "dung": 0, "sai": 0, "max_cd": 0, "max_cs": 0, "cd": 0, "cs": 0}
lich_su = []
data_now = {}
last_phien = None
_prev_pred = None


# ================= UTILS =================
def encode(tx_list):
    return "".join(["T" if x == "Tài" else "X" for x in tx_list])

def decode(c):
    return "Tài" if c == "T" else "Xỉu"


# ================= AI MODELS =================

class MarkovChain:
    def predict(self, tx_list, order=3):
        if len(tx_list) < order + 5:
            return None, 0
        s = encode(tx_list)
        transitions = {}
        for i in range(len(s) - order):
            state = s[i:i + order]
            nxt = s[i + order]
            if state not in transitions:
                transitions[state] = Counter()
            transitions[state][nxt] += 1
        
        current = s[-order:]
        if current not in transitions:
            return None, 0
        
        counts = transitions[current]
        total = sum(counts.values())
        prob_t = counts.get("T", 0) / total
        prob_x = counts.get("X", 0) / total
        
        pred = "Tài" if prob_t > prob_x else "Xỉu"
        conf = max(prob_t, prob_x) * 100
        return pred, conf


class NGramModel:
    def predict(self, tx_list, max_n=5):
        if len(tx_list) < 10:
            return None, 0
        
        s = encode(tx_list)
        votes = Counter()
        confidences = []
        
        for n in range(2, min(max_n + 1, len(s) // 2 + 1)):
            if len(s) < n + 2:
                continue
            grams = Counter()
            for i in range(len(s) - n):
                gram = s[i:i + n]
                nxt = s[i + n]
                grams[(gram, nxt)] += 1
            
            current = s[-n:]
            t_total = sum(v for (g, c), v in grams.items() if g == current and c == "T")
            x_total = sum(v for (g, c), v in grams.items() if g == current and c == "X")
            total = t_total + x_total
            
            if total > 0:
                if t_total > x_total:
                    votes["Tài"] += n
                    confidences.append(t_total / total * 100)
                elif x_total > t_total:
                    votes["Xỉu"] += n
                    confidences.append(x_total / total * 100)
        
        if not votes:
            return None, 0
        
        pred = votes.most_common(1)[0][0]
        avg_conf = sum(confidences) / len(confidences) if confidences else 50
        return pred, min(avg_conf, 95)


class PatternModel:
    def __init__(self):
        self.patterns = {
            "bệt": self._detect_streak,
            "1-1": self._detect_alt,
            "2-2": self._detect_double,
            "3-3": self._detect_triple,
            "cân_bằng": self._detect_balance,
            "đảo_sau_bệt": self._detect_reversal,
            "lặp_chu_kỳ": self._detect_cycle
        }
    
    def _detect_streak(self, s):
        if len(s) < 3:
            return None
        last = s[-1]
        count = 0
        for c in reversed(s):
            if c == last:
                count += 1
            else:
                break
        if count >= 3:
            opposite = "X" if last == "T" else "T"
            return decode(opposite), min(50 + count * 8, 90), f"Bệt {decode(last)} x{count}"
        return None
    
    def _detect_alt(self, s):
        if len(s) < 4:
            return None
        recent = s[-6:]
        is_alt = all(recent[i] != recent[i + 1] for i in range(len(recent) - 1))
        if is_alt:
            next_c = "X" if recent[-1] == "T" else "T"
            return decode(next_c), 75, "Cầu 1-1"
        return None
    
    def _detect_double(self, s):
        if len(s) < 6:
            return None
        recent = s[-8:]
        for i in range(0, len(recent) - 3, 2):
            if not (recent[i] == recent[i + 1] and recent[i + 2] == recent[i + 3] and recent[i] != recent[i + 2]):
                return None
        next_expected = recent[0] if len(recent) % 4 == 0 else recent[2]
        return decode(next_expected), 80, "Cầu 2-2"
    
    def _detect_triple(self, s):
        if len(s) < 9:
            return None
        recent = s[-12:]
        for i in range(0, len(recent) - 5, 3):
            if not (recent[i] == recent[i + 1] == recent[i + 2] and recent[i + 3] == recent[i + 4] == recent[i + 5] and recent[i] != recent[i + 3]):
                return None
        return decode(recent[0]), 85, "Cầu 3-3"
    
    def _detect_balance(self, s):
        if len(s) < 20:
            return None
        recent = s[-20:]
        t_count = recent.count("T")
        t_pct = t_count / 20
        if t_pct > 0.7:
            return "Xỉu", min(50 + (t_pct - 0.5) * 100, 90), f"Tài thiên lệch {t_pct:.0%}"
        elif t_pct < 0.3:
            return "Tài", min(50 + (0.5 - t_pct) * 100, 90), f"Xỉu thiên lệch {1 - t_pct:.0%}"
        return None
    
    def _detect_reversal(self, s):
        if len(s) < 5:
            return None
        streak = 0
        last = s[-1]
        for c in reversed(s):
            if c == last:
                streak += 1
            else:
                break
        if streak >= 3:
            opposite = "X" if last == "T" else "T"
            return decode(opposite), 70, f"Đảo sau bệt {streak}"
        return None
    
    def _detect_cycle(self, s):
        if len(s) < 8:
            return None
        for cycle_len in range(2, min(6, len(s) // 2)):
            cycle = s[-cycle_len:]
            if s[-cycle_len * 2:-cycle_len] == cycle:
                next_in_cycle = cycle[0]
                return decode(next_in_cycle), 80, f"Chu kỳ {cycle_len}"
        return None
    
    def predict(self, tx_list):
        s = encode(tx_list)
        results = []
        for name, detector in self.patterns.items():
            result = detector(s)
            if result:
                pred, conf, reason = result
                results.append((pred, conf, reason, name))
        
        if not results:
            return None, 0, "Không nhận diện pattern"
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[0][0], results[0][1], results[0][2]


class StreakModel:
    def predict(self, tx_list):
        if len(tx_list) < 10:
            return None, 0
        
        s = encode(tx_list)
        streaks = []
        current = 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                current += 1
            else:
                streaks.append(current)
                current = 1
        streaks.append(current)
        
        avg_streak = sum(streaks) / len(streaks)
        current_streak = streaks[-1]
        
        if current_streak > avg_streak + 0.5:
            opposite = "Xỉu" if s[-1] == "T" else "Tài"
            conf = min(50 + (current_streak - avg_streak) * 15, 90)
            return opposite, conf, f"Streak {current_streak} > avg {avg_streak:.1f}"
        
        if current_streak <= avg_streak:
            cont = decode(s[-1])
            conf = min(50 + (avg_streak - current_streak) * 10, 85)
            return cont, conf, f"Streak {current_streak} <= avg {avg_streak:.1f}"
        
        return None, 0, "Streak trung lập"


class ReversalModel:
    def predict(self, tx_list):
        if len(tx_list) < 15:
            return None, 0
        
        s = encode(tx_list)
        recent = s[-15:]
        reversals = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])
        reversal_rate = reversals / (len(recent) - 1)
        
        if reversal_rate > 0.6:
            next_c = "X" if recent[-1] == "T" else "T"
            return decode(next_c), min(50 + reversal_rate * 30, 85), f"Đảo cao {reversal_rate:.0%}"
        
        if reversal_rate < 0.3:
            return decode(recent[-1]), min(50 + (0.3 - reversal_rate) * 100, 80), f"Ít đảo {reversal_rate:.0%}"
        
        return None, 0, f"Đảo trung bình {reversal_rate:.0%}"


class CycleModel:
    def predict(self, tx_list):
        if len(tx_list) < 20:
            return None, 0
        
        s = encode(tx_list)
        best_cycle = None
        best_score = 0
        
        for cycle in range(2, min(11, len(s) // 2)):
            matches = 0
            total = 0
            for i in range(cycle, len(s)):
                if s[i] == s[i - cycle]:
                    matches += 1
                total += 1
            
            score = matches / total if total > 0 else 0
            if score > best_score and score > 0.55:
                best_score = score
                best_cycle = cycle
        
        if best_cycle:
            predicted = s[-best_cycle]
            conf = min(best_score * 100, 90)
            return decode(predicted), conf, f"Chu kỳ {best_cycle} ({best_score:.0%})"
        
        return None, 0, "Không có chu kỳ rõ ràng"


class MomentumModel:
    def predict(self, tx_list):
        if len(tx_list) < 10:
            return None, 0
        
        s = encode(tx_list)
        windows = [5, 10, 15]
        momentums = []
        
        for w in windows:
            if len(s) >= w:
                recent = s[-w:]
                t_pct = recent.count("T") / w
                momentums.append(t_pct)
        
        if not momentums:
            return None, 0
        
        avg_momentum = sum(momentums) / len(momentums)
        
        if avg_momentum > 0.6:
            return "Tài", min(avg_momentum * 100, 90), f"Momentum Tài {avg_momentum:.0%}"
        elif avg_momentum < 0.4:
            return "Xỉu", min((1 - avg_momentum) * 100, 90), f"Momentum Xỉu {1 - avg_momentum:.0%}"
        
        return None, 0, f"Momentum trung lập {avg_momentum:.0%}"


class TrendStrengthModel:
    def predict(self, tx_list):
        if len(tx_list) < 15:
            return None, 0
        
        s = encode(tx_list)
        half = len(s) // 2
        first_half = s[:half].count("T") / half if half > 0 else 0.5
        second_half = s[half:].count("T") / (len(s) - half) if len(s) - half > 0 else 0.5
        
        trend_diff = abs(second_half - first_half)
        
        if trend_diff > 0.15:
            if second_half > first_half:
                return "Tài", min(50 + trend_diff * 200, 85), "Trend Tài mạnh"
            else:
                return "Xỉu", min(50 + trend_diff * 200, 85), "Trend Xỉu mạnh"
        
        return None, 0, f"Trend yếu {trend_diff:.0%}"


class MovingAverageModel:
    def predict(self, tx_list):
        if len(tx_list) < 20:
            return None, 0
        
        values = [1 if x == "Tài" else 0 for x in tx_list]
        ma5 = sum(values[-5:]) / 5
        ma10 = sum(values[-10:]) / 10
        ma20 = sum(values[-20:]) / 20 if len(values) >= 20 else ma10
        
        if ma5 > ma10 > ma20:
            return "Tài", 75, "MA Bullish"
        elif ma5 < ma10 < ma20:
            return "Xỉu", 75, "MA Bearish"
        
        if len(values) >= 6:
            prev_ma5 = sum(values[-6:-1]) / 5
            if ma5 > prev_ma5 and ma5 > 0.5:
                return "Tài", 70, "MA5 tăng"
            elif ma5 < prev_ma5 and ma5 < 0.5:
                return "Xỉu", 70, "MA5 giảm"
        
        return None, 0, f"MA5={ma5:.2f} MA10={ma10:.2f}"


class EnsembleModel:
    def __init__(self):
        self.models = {
            "Markov": MarkovChain(),
            "N-Gram": NGramModel(),
            "Pattern": PatternModel(),
            "Streak": StreakModel(),
            "Reversal": ReversalModel(),
            "Cycle": CycleModel(),
            "Momentum": MomentumModel(),
            "Trend": TrendStrengthModel(),
            "MA": MovingAverageModel()
        }
    
    def predict(self, tx_list):
        votes = Counter()
        confidences = {}
        reasons = {}
        
        for name, model in self.models.items():
            try:
                pred, conf, reason = model.predict(tx_list)
                if pred:
                    votes[pred] += conf
                    confidences[name] = round(conf, 1)
                    reasons[name] = reason
            except:
                continue
        
        if not votes:
            if tx_list:
                fallback = "Xỉu" if tx_list[-1] == "Tài" else "Tài"
                return fallback, 50, {"Fallback": "Đảo cầu"}, "Không có tín hiệu - đảo"
            return "Tài", 50, {}, "Mặc định"
        
        winner = votes.most_common(1)[0]
        pred = winner[0]
        total_conf = winner[1]
        avg_conf = min(total_conf / sum(votes.values()) * 100, 95) if sum(votes.values()) > 0 else 50
        
        return pred, round(avg_conf, 1), confidences, " | ".join([f"{k}:{v}" for k, v in reasons.items()])


class AdaptiveWeightModel:
    def __init__(self):
        self.model = EnsembleModel()
        self.weights = {name: 1.0 for name in self.model.models.keys()}
        self.performance = {name: {"dung": 0, "sai": 0} for name in self.model.models.keys()}
        self.history = {}
    
    def update_weights(self, prediction, actual, tx_list):
        if len(tx_list) < 5:
            return
        
        s = encode(tx_list[:-1])
        
        for name, model in self.model.models.items():
            try:
                pred, _, _ = model.predict(list(tx_list[:-1]))
                if pred:
                    if pred == actual:
                        self.performance[name]["dung"] += 1
                        self.weights[name] = min(self.weights[name] * 1.05, 3.0)
                    else:
                        self.performance[name]["sai"] += 1
                        self.weights[name] = max(self.weights[name] * 0.95, 0.3)
            except:
                continue
    
    def predict(self, tx_list):
        pred, conf, model_confs, reason = self.model.predict(tx_list)
        return pred, conf, model_confs, reason, self.weights.copy()


# Khởi tạo AI
ai_engine = AdaptiveWeightModel()


# ================= PREDICT NEXT =================
def predict_next(tx_list):
    if len(tx_list) < MIN_PHIEN:
        return "Chờ dữ liệu", 0, {}, "Chưa đủ 15 phiên", {}
    
    du_doan, do_tin_cay, model_confs, phan_tich, weights = ai_engine.predict(tx_list)
    return du_doan, do_tin_cay, model_confs, phan_tich, weights


# ================= CẬP NHẬT THỐNG KÊ =================
def update_stats(actual, phien_id):
    global _prev_pred, stats, lich_su
    
    if not _prev_pred or _prev_pred == "Chờ dữ liệu":
        return
    
    dung = (_prev_pred == actual)
    stats["tong"] += 1
    
    if dung:
        stats["dung"] += 1
        stats["cd"] += 1
        stats["cs"] = 0
        if stats["cd"] > stats["max_cd"]:
            stats["max_cd"] = stats["cd"]
    else:
        stats["sai"] += 1
        stats["cs"] += 1
        stats["cd"] = 0
        if stats["cs"] > stats["max_cs"]:
            stats["max_cs"] = stats["cs"]
    
    lich_su.append({
        "phien": phien_id,
        "du_doan": _prev_pred,
        "ket_qua": actual,
        "dung": "✅" if dung else "❌"
    })
    
    if len(lich_su) > 100:
        lich_su.pop(0)


# ================= TÍNH CHUỖI =================
def tinh_chuoi(tx_list):
    if not tx_list:
        return 0, None, 0, 0
    
    s = encode(tx_list)
    max_tai = max_xiu = cur_tai = cur_xiu = 0
    current_streak = 0
    current_type = None
    
    for i, c in enumerate(s):
        if c == "T":
            cur_tai += 1
            cur_xiu = 0
            max_tai = max(max_tai, cur_tai)
        else:
            cur_xiu += 1
            cur_tai = 0
            max_xiu = max(max_xiu, cur_xiu)
        
        if i == len(s) - 1:
            current_streak = cur_tai if c == "T" else cur_xiu
            current_type = decode(c)
    
    return current_streak, current_type, max_tai, max_xiu


# ================= FETCH DATA =================
def fetch_data():
    global data_now, last_phien, _prev_pred, history_tx, history_pt, history_id, history_dice
    
    while True:
        try:
            res = requests.get(API_URL, timeout=5)
            if res.status_code != 200:
                time.sleep(2)
                continue
            
            data = res.json()
            if not isinstance(data, list) or len(data) == 0:
                time.sleep(2)
                continue
            
            latest = data[0]
            phien = latest.get("SessionId")
            
            if phien == last_phien:
                time.sleep(1)
                continue
            
            d1 = latest.get("FirstDice", 0)
            d2 = latest.get("SecondDice", 0)
            d3 = latest.get("ThirdDice", 0)
            tong = latest.get("DiceSum", 0)
            
            ket = "Tài" if tong >= 11 else "Xỉu"
            
            # Update stats trước khi lưu history mới
            if len(history_tx) >= MIN_PHIEN and _prev_pred and _prev_pred != "Chờ dữ liệu":
                ai_engine.update_weights(ket, ket, list(history_tx))
            
            update_stats(ket, phien)
            
            
            history_tx.append(ket)
            history_pt.append(tong)
            history_id.append(phien)
            history_dice.append((d1, d2, d3))
            
            
            streak, stype, max_tai, max_xiu = tinh_chuoi(list(history_tx))
            
            
            tx_list = list(history_tx)
            du_doan, do_tin_cay, model_confs, phan_tich, weights = predict_next(tx_list)
            _prev_pred = du_doan
            
            
            pattern = encode(tx_list)[-20:] if len(tx_list) >= 20 else encode(tx_list)
            
            
            td = stats["tong"]
            ty_le = f"{stats['dung']/td*100:.1f}%" if td else "0%"
            
            data_now = {
                "Phien": phien,
                "Xuc_xac_1": d1,
                "Xuc_xac_2": d2,
                "Xuc_xac_3": d3,
                "Tong": tong,
                "Ket": ket,
                "Phien_hien_tai": phien + 1,
                "Du_doan": du_doan,
                "Do_tin_cay": f"{do_tin_cay}%",
                "Pattern": pattern,
                "Max_chuoi_Tai": max_tai,
                "Max_chuoi_Xiu": max_xiu,
                "Ty_le_dung": ty_le,
                "AI_Models": model_confs,
                "Trong_so": {k: round(v, 2) for k, v in weights.items()},
                "Phan_tich": phan_tich,
                
            }
            
            
            
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Lỗi:", e)
        
        time.sleep(2)


# ================= API =================
@app.route("/")
def home():
    return jsonify({"status": "ok", "data": data_now})

@app.route("/api/taixiumd5")
def api_main():
    return jsonify(data_now)

@app.route("/api/lichsu")
def api_history():
    td = stats["tong"]
    return jsonify({
        "tong": td,
        "dung": stats["dung"],
        "sai": stats["sai"],
        "ty_le": f"{stats['dung']/td*100:.1f}%" if td else "0%",
        "max_cd": stats["max_cd"],
        "max_cs": stats["max_cs"],
        "chuoi_hien_tai": f"{stats['cd']} đúng liên tiếp" if stats["cd"] > 0 else (f"{stats['cs']} sai liên tiếp" if stats["cs"] > 0 else "0"),
        "20_phien": lich_su[-20:],
        "lich_su_day_du": lich_su[-50:]
    })


# ================= RUN =================
threading.Thread(target=fetch_data, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
