from flask import Flask, jsonify
import requests
import time
import threading
import math
import json
import os
from collections import deque, Counter
from datetime import datetime

app = Flask(__name__)

# ================= CONFIG =================
MIN_PHIEN = 15
MAX_HISTORY = 20
DATA_FILE = "luckywin_ai_data.json"
WEIGHTS_FILE = "luckywin_ai_weights.json"

API_LIST = [
    "https://luck8bot.com/api/GetNewLottery/TaixiuMd5?id=",
    "http://luck8bot.com/api/GetNewLottery/TaixiuMd5?id="
]

# ================= DATA =================
history_tx = deque(maxlen=MAX_HISTORY)
history_pt = deque(maxlen=MAX_HISTORY)
history_id = deque(maxlen=MAX_HISTORY)

lich_su = []
stats = {
    "tong": 0, "dung": 0, "sai": 0,
    "cd": 0, "cs": 0, "max_cd": 0, "max_cs": 0
}

last_result = {}
_prev_pred = None
_last_phien_processed = None

# ================= UTILS =================
def encode(tx_list):
    return "".join(["T" if x == "Tài" else "X" for x in tx_list])

def decode(c):
    return "Tài" if c == "T" else "Xỉu"

def save_data():
    try:
        data = {
            "history": list(history_tx),
            "points": list(history_pt),
            "ids": list(history_id),
            "stats": stats,
            "lich_su": list(lich_su)[-200:],
            "last_phien": _last_phien_processed
        }
        with open(DATA_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[SAVE ERROR] {e}")

def load_data():
    global _last_phien_processed
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                # Load history
                for h, p, i in zip(data.get("history", [])[-100:],
                                   data.get("points", [])[-100:],
                                   data.get("ids", [])[-100:]):
                    history_tx.append(h)
                    history_pt.append(p)
                    history_id.append(i)
                # Load stats
                saved_stats = data.get("stats", {})
                for k in stats:
                    if k in saved_stats:
                        stats[k] = saved_stats[k]
                # Load lich_su
                saved_lich_su = data.get("lich_su", [])
                lich_su.extend(saved_lich_su)
                # Load last_phien
                _last_phien_processed = data.get("last_phien")
                print(f"[LOAD] Loaded {len(history_tx)} history, {len(lich_su)} records, last_phien={_last_phien_processed}")
    except Exception as e:
        print(f"[LOAD ERROR] {e}")

# ================= NHẬN DIỆN CẦU =================

class CauDetector:
    @staticmethod
    def detect(tx_list):
        if len(tx_list) < 5:
            return "Chưa đủ dữ liệu", []
        
        s = encode(tx_list)
        n = len(s)
        cau_list = []
        
        # 1. Cầu bệt
        streak = 0
        last = s[-1]
        for c in reversed(s):
            if c == last:
                streak += 1
            else:
                break
        if streak >= 3:
            cau_list.append(f"Bệt {decode(last)} x{streak}")
        
        # 2. Cầu 1-1
        if n >= 6:
            recent6 = s[-6:]
            if all(recent6[i] != recent6[i+1] for i in range(5)):
                cau_list.append("Cầu 1-1 (T-X-T-X-T-X)")
        
        # 3. Cầu 2-2
        if n >= 8:
            recent8 = s[-8:]
            is22 = True
            for i in range(0, 6, 2):
                if not (recent8[i] == recent8[i+1] and recent8[i+2] == recent8[i+3] and recent8[i] != recent8[i+2]):
                    is22 = False
                    break
            if is22:
                cau_list.append("Cầu 2-2 (TT-XX-TT-XX)")
        
        # 4. Cầu 3-3
        if n >= 12:
            recent12 = s[-12:]
            is33 = True
            for i in range(0, 9, 3):
                if not (recent12[i]==recent12[i+1]==recent12[i+2] and 
                        recent12[i+3]==recent12[i+4]==recent12[i+5] and 
                        recent12[i]!=recent12[i+3]):
                    is33 = False
                    break
            if is33:
                cau_list.append("Cầu 3-3 (TTT-XXX-TTT-XXX)")
        
        # 5. Cầu đảo sau bệt
        if streak >= 3 and n >= streak + 2:
            cau_list.append(f"Đảo sau bệt {streak}")
        
        # 6. Cầu cân bằng
        if n >= 20:
            recent20 = s[-20:]
            t_count = recent20.count("T")
            if t_count >= 14:
                cau_list.append(f"Tài thiên lệch {t_count}/20 -> chờ Xỉu")
            elif t_count <= 6:
                cau_list.append(f"Xỉu thiên lệch {20-t_count}/20 -> chờ Tài")
        
        # 7. Cầu chu kỳ
        if n >= 15:
            for cycle in range(2, 8):
                if n >= cycle * 3:
                    recent = s[-cycle*3:]
                    if all(recent[i] == recent[i+cycle] == recent[i+cycle*2] for i in range(cycle)):
                        cau_list.append(f"Chu kỳ lặp {cycle}")
                        break
        
        if not cau_list:
            return "Cầu hỗn loạn / Không nhận diện", []
        
        return " | ".join(cau_list), cau_list

# ================= 20 AI MODELS =================

class MarkovChain:
    def predict(self, tx_list, order=3):
        if len(tx_list) < order + 5:
            return None, 0
        s = encode(tx_list)
        trans = {}
        for i in range(len(s) - order):
            state = s[i:i+order]
            trans.setdefault(state, Counter())[s[i+order]] += 1
        cur = s[-order:]
        if cur not in trans:
            return None, 0
        counts = trans[cur]
        total = sum(counts.values())
        if total < 2:
            return None, 0
        p_t = counts.get("T", 0) / total
        pred = "Tài" if p_t > 0.5 else "Xỉu"
        conf = max(p_t, 1-p_t) * 100
        return pred, conf

class NGramModel:
    def predict(self, tx_list, max_n=5):
        if len(tx_list) < 10:
            return None, 0
        s = encode(tx_list)
        votes = Counter()
        confs = []
        for n in range(2, min(max_n+1, len(s)//2+1)):
            if len(s) < n + 2:
                continue
            grams = Counter()
            for i in range(len(s)-n):
                grams[(s[i:i+n], s[i+n])] += 1
            cur = s[-n:]
            t_score = sum(v for (g,c),v in grams.items() if g==cur and c=="T")
            x_score = sum(v for (g,c),v in grams.items() if g==cur and c=="X")
            total = t_score + x_score
            if total > 0:
                if t_score > x_score:
                    votes["Tài"] += n
                    confs.append(t_score/total*100)
                elif x_score > t_score:
                    votes["Xỉu"] += n
                    confs.append(x_score/total*100)
        if not votes:
            return None, 0
        pred = votes.most_common(1)[0][0]
        return pred, min(sum(confs)/len(confs) if confs else 50, 95)

class PatternModel:
    def predict(self, tx_list):
        if len(tx_list) < 5:
            return None, 0
        s = encode(tx_list)
        n = len(s)
        
        streak = 0
        last = s[-1]
        for c in reversed(s):
            if c == last:
                streak += 1
            else:
                break
        if streak >= 3:
            opp = "X" if last == "T" else "T"
            return decode(opp), min(50 + streak * 10, 90)
        
        if n >= 5:
            recent = s[-5:]
            if all(recent[i] != recent[i+1] for i in range(4)):
                return decode("X" if recent[-1] == "T" else "T"), 80
        
        if n >= 20:
            recent = s[-20:]
            t_pct = recent.count("T") / 20
            if t_pct >= 0.7:
                return "Xỉu", min(50 + (t_pct-0.5)*100, 90)
            elif t_pct <= 0.3:
                return "Tài", min(50 + (0.5-t_pct)*100, 90)
        
        return None, 0

class StreakModel:
    def predict(self, tx_list):
        if len(tx_list) < 10:
            return None, 0
        s = encode(tx_list)
        streaks = []
        cur = 1
        for i in range(1, len(s)):
            if s[i] == s[i-1]:
                cur += 1
            else:
                streaks.append(cur)
                cur = 1
        streaks.append(cur)
        avg = sum(streaks) / len(streaks)
        current = streaks[-1]
        
        if current > avg + 0.5:
            opp = "Xỉu" if s[-1] == "T" else "Tài"
            return opp, min(50 + (current-avg)*20, 90)
        elif current <= avg:
            return decode(s[-1]), min(50 + (avg-current)*15, 85)
        return None, 0

class ReversalModel:
    def predict(self, tx_list):
        if len(tx_list) < 15:
            return None, 0
        s = encode(tx_list)[-15:]
        rev = sum(1 for i in range(1, len(s)) if s[i] != s[i-1])
        rate = rev / (len(s)-1)
        
        if rate > 0.6:
            return decode("X" if s[-1] == "T" else "T"), min(50 + rate*30, 85)
        elif rate < 0.3:
            return decode(s[-1]), min(50 + (0.3-rate)*100, 80)
        return None, 0

class CycleModel:
    def predict(self, tx_list):
        if len(tx_list) < 20:
            return None, 0
        s = encode(tx_list)
        best_cycle = None
        best_score = 0
        
        for c_len in range(2, min(10, len(s)//2)):
            matches = sum(1 for i in range(c_len, len(s)) if s[i] == s[i-c_len])
            score = matches / (len(s)-c_len)
            if score > best_score and score > 0.55:
                best_score = score
                best_cycle = c_len
        
        if best_cycle:
            return decode(s[-best_cycle]), min(best_score*100, 90)
        return None, 0

class MomentumModel:
    def predict(self, tx_list):
        if len(tx_list) < 10:
            return None, 0
        s = encode(tx_list)
        momentums = []
        for w in [5, 10, 15]:
            if len(s) >= w:
                t_pct = s[-w:].count("T") / w
                momentums.append(t_pct)
        
        if not momentums:
            return None, 0
        avg = sum(momentums) / len(momentums)
        
        if avg > 0.6:
            return "Tài", min(avg*100, 90)
        elif avg < 0.4:
            return "Xỉu", min((1-avg)*100, 90)
        return None, 0

class TrendModel:
    def predict(self, tx_list):
        if len(tx_list) < 15:
            return None, 0
        s = encode(tx_list)
        half = len(s) // 2
        f = s[:half].count("T") / half if half else 0.5
        se = s[half:].count("T") / (len(s)-half) if len(s)-half else 0.5
        diff = abs(se - f)
        
        if diff > 0.15:
            return ("Tài" if se > f else "Xỉu"), min(50 + diff*200, 85)
        return None, 0

class MAModel:
    def predict(self, tx_list):
        if len(tx_list) < 20:
            return None, 0
        v = [1 if x == "Tài" else 0 for x in tx_list]
        ma5 = sum(v[-5:]) / 5
        ma10 = sum(v[-10:]) / 10
        ma20 = sum(v[-20:]) / 20 if len(v) >= 20 else ma10
        
        if ma5 > ma10 > ma20:
            return "Tài", 78
        elif ma5 < ma10 < ma20:
            return "Xỉu", 78
        
        if len(v) >= 6:
            pma5 = sum(v[-6:-1]) / 5
            if ma5 > pma5 and ma5 > 0.5:
                return "Tài", 72
            elif ma5 < pma5 and ma5 < 0.5:
                return "Xỉu", 72
        return None, 0

class BayesianModel:
    def predict(self, tx_list):
        if len(tx_list) < 20:
            return None, 0
        s = encode(tx_list)
        prior_t = s.count("T") / len(s)
        recent5 = s[-5:]
        t5 = recent5.count("T")
        
        if t5 >= 4:
            return "Xỉu", 70
        elif t5 <= 1:
            return "Tài", 70
        return None, 0

class EntropyModel:
    def predict(self, tx_list):
        if len(tx_list) < 20:
            return None, 0
        s = encode(tx_list)[-20:]
        t = s.count("T")
        x = len(s) - t
        if t == 0 or x == 0:
            return decode("X" if s[-1] == "T" else "T"), 75
        
        p_t = t / len(s)
        p_x = 1 - p_t
        entropy = -(p_t*math.log2(p_t) + p_x*math.log2(p_x))
        
        if entropy < 0.7:
            return decode(s[-1]), 75
        elif entropy > 0.95:
            return decode("X" if s[-1] == "T" else "T"), 60
        return None, 0

class MarkovHighOrder:
    def predict(self, tx_list):
        if len(tx_list) < 25:
            return None, 0
        s = encode(tx_list)
        best_pred = None
        best_conf = 0
        
        for order in [2, 3, 4, 5]:
            if len(s) < order + 5:
                continue
            trans = {}
            for i in range(len(s)-order):
                state = s[i:i+order]
                trans.setdefault(state, Counter())[s[i+order]] += 1
            cur = s[-order:]
            if cur not in trans:
                continue
            counts = trans[cur]
            total = sum(counts.values())
            if total < 3:
                continue
            p_t = counts.get("T", 0) / total
            conf = max(p_t, 1-p_t) * 100
            if conf > best_conf:
                best_conf = conf
                best_pred = "Tài" if p_t > 0.5 else "Xỉu"
        
        if best_pred and best_conf >= 55:
            return best_pred, min(best_conf, 95)
        return None, 0

class RegressionModel:
    def predict(self, tx_list):
        if len(tx_list) < 25:
            return None, 0
        v = [1 if x == "Tài" else 0 for x in tx_list[-20:]]
        n = len(v)
        sum_x = sum(range(n))
        sum_y = sum(v)
        sum_xy = sum(i*v[i] for i in range(n))
        sum_x2 = sum(i*i for i in range(n))
        denom = n*sum_x2 - sum_x*sum_x
        if denom == 0:
            return None, 0
        slope = (n*sum_xy - sum_x*sum_y) / denom
        
        if slope > 0.05:
            return "Tài", min(50+slope*500, 85)
        elif slope < -0.05:
            return "Xỉu", min(50+abs(slope)*500, 85)
        return None, 0

class FreqAdaptiveModel:
    def predict(self, tx_list):
        if len(tx_list) < 30:
            return None, 0
        s = encode(tx_list)
        signals = []
        for w in [10, 20, 30]:
            if len(s) < w:
                continue
            t_pct = s[-w:].count("T") / w
            if t_pct > 0.65:
                signals.append(("Xỉu", t_pct))
            elif t_pct < 0.35:
                signals.append(("Tài", t_pct))
        
        if not signals:
            return None, 0
        signals.sort(key=lambda x: x[1], reverse=True)
        pred, conf = signals[0]
        return pred, min(conf*100, 90)

class DeepPatternModel:
    def predict(self, tx_list):
        if len(tx_list) < 25:
            return None, 0
        s = encode(tx_list)
        
        for pat_len in range(3, min(7, len(s)//3)):
            for gap in range(1, min(5, (len(s)-pat_len)//2)):
                matches = 0
                total = 0
                for i in range(len(s)-pat_len-gap):
                    if s[i:i+pat_len] == s[i+gap:i+gap+pat_len]:
                        matches += 1
                    total += 1
                score = matches/total if total else 0
                if score > 0.6:
                    recent = s[-pat_len:]
                    for i in range(len(s)-pat_len-gap, -1, -1):
                        if s[i:i+pat_len] == recent and i+pat_len+gap < len(s):
                            return decode(s[i+pat_len+gap]), min(score*100, 90)
        
        for length in range(4, min(8, len(s)//2)):
            sub = s[-length:]
            positions = [i for i in range(len(s)-length) if s[i:i+length] == sub]
            if len(positions) >= 2:
                next_chars = [s[p+length] for p in positions if p+length < len(s)]
                if next_chars:
                    t_count = next_chars.count("T")
                    pred = "Tài" if t_count > len(next_chars)/2 else "Xỉu"
                    conf = max(t_count, len(next_chars)-t_count) / len(next_chars) * 100
                    return pred, min(conf, 90)
        return None, 0

class FibonacciModel:
    def predict(self, tx_list):
        if len(tx_list) < 8:
            return None, 0
        s = encode(tx_list)
        streak = 0
        last = s[-1]
        for c in reversed(s):
            if c == last:
                streak += 1
            else:
                break
        
        fib_streaks = [2, 3, 5, 8, 13]
        if streak in fib_streaks:
            opp = "X" if last == "T" else "T"
            return decode(opp), 70
        return None, 0

class GoldenRatioModel:
    def predict(self, tx_list):
        if len(tx_list) < 30:
            return None, 0
        s = encode(tx_list)
        phi = int(len(s) / 1.618)
        recent_phi = s[-phi:].count("T") / phi if phi else 0.5
        
        if recent_phi > 0.65:
            return "Xỉu", 70
        elif recent_phi < 0.35:
            return "Tài", 70
        return None, 0

class VolatilityModel:
    def predict(self, tx_list):
        if len(tx_list) < 15:
            return None, 0
        s = encode(tx_list)[-15:]
        changes = sum(1 for i in range(1, len(s)) if s[i] != s[i-1])
        vol = changes / (len(s)-1)
        
        if vol > 0.7:
            return decode("X" if s[-1] == "T" else "T"), 68
        elif vol < 0.2:
            return decode(s[-1]), 68
        return None, 0

class SupportResistanceModel:
    def predict(self, tx_list):
        if len(tx_list) < 20:
            return None, 0
        s = encode(tx_list)
        streaks = []
        cur = 1
        for i in range(1, len(s)):
            if s[i] == s[i-1]:
                cur += 1
            else:
                streaks.append((cur, s[i-1]))
                cur = 1
        streaks.append((cur, s[-1]))
        
        if len(streaks) >= 2 and streaks[-2][0] >= 4:
            return decode(streaks[-1][1]), 75
        return None, 0

class TimeSeriesModel:
    def predict(self, tx_list):
        if len(tx_list) < 15:
            return None, 0
        v = [1 if x == "Tài" else 0 for x in tx_list[-15:]]
        alpha = 0.3
        smoothed = v[0]
        for val in v[1:]:
            smoothed = alpha * val + (1-alpha) * smoothed
        
        if smoothed > 0.58:
            return "Tài", 70
        elif smoothed < 0.42:
            return "Xỉu", 70
        return None, 0

class ClusteringModel:
    def predict(self, tx_list):
        if len(tx_list) < 25:
            return None, 0
        s = encode(tx_list)
        recent8 = s[-8:]
        
        best_score = 0
        best_match = None
        for i in range(len(s)-16):
            segment = s[i:i+8]
            score = sum(a==b for a,b in zip(recent8, segment)) / 8
            if score > best_score and score > 0.75:
                best_score = score
                if i+8 < len(s):
                    best_match = s[i+8]
        
        if best_match:
            return decode(best_match), min(best_score*100, 85)
        return None, 0

# ================= ENSEMBLE =================

class SuperEnsemble:
    def __init__(self):
        self.models = {
            "Markov": MarkovChain(),
            "N-Gram": NGramModel(),
            "Pattern": PatternModel(),
            "Streak": StreakModel(),
            "Reversal": ReversalModel(),
            "Cycle": CycleModel(),
            "Momentum": MomentumModel(),
            "Trend": TrendModel(),
            "MA": MAModel(),
            "Bayes": BayesianModel(),
            "Entropy": EntropyModel(),
            "MarkovHO": MarkovHighOrder(),
            "Regression": RegressionModel(),
            "FreqAdapt": FreqAdaptiveModel(),
            "DeepPattern": DeepPatternModel(),
            "Fibonacci": FibonacciModel(),
            "GoldenRatio": GoldenRatioModel(),
            "Volatility": VolatilityModel(),
            "SupportResist": SupportResistanceModel(),
            "TimeSeries": TimeSeriesModel(),
            "Clustering": ClusteringModel()
        }
        self.weights = {name: 1.0 for name in self.models}
        self.performance = {name: {"dung": 0, "sai": 0} for name in self.models}
        self.load_weights()
    
    def load_weights(self):
        try:
            if os.path.exists(WEIGHTS_FILE):
                with open(WEIGHTS_FILE, "r") as f:
                    data = json.load(f)
                    for k, v in data.get("weights", {}).items():
                        if k in self.weights:
                            self.weights[k] = v
                    perf = data.get("performance", {})
                    for k, v in perf.items():
                        if k in self.performance:
                            self.performance[k] = v
        except:
            pass
    
    def save_weights(self):
        try:
            with open(WEIGHTS_FILE, "w") as f:
                json.dump({"weights": self.weights, "performance": self.performance}, f)
        except:
            pass
    
    def update(self, actual, tx_list_before):
        if len(tx_list_before) < 5:
            return
        for name, model in self.models.items():
            try:
                result = model.predict(list(tx_list_before))
                if result and len(result) >= 2 and result[0]:
                    pred = result[0]
                    if pred == actual:
                        self.performance[name]["dung"] += 1
                        self.weights[name] = min(self.weights[name] * 1.08, 5.0)
                    else:
                        self.performance[name]["sai"] += 1
                        self.weights[name] = max(self.weights[name] * 0.92, 0.2)
            except:
                pass
        self.save_weights()
    
    def predict(self, tx_list):
        votes = Counter()
        details = {}
        reasons = {}
        active_models = 0
        
        for name, model in self.models.items():
            try:
                result = model.predict(tx_list)
                if result and len(result) >= 2 and result[0]:
                    pred, conf = result[0], result[1]
                    if conf >= 55:
                        weight = self.weights.get(name, 1.0)
                        votes[pred] += conf * weight
                        details[name] = round(conf, 1)
                        reasons[name] = f"{name}:{conf:.0f}%"
                        active_models += 1
            except:
                pass
        
        if not votes:
            if tx_list:
                s = encode(tx_list)
                streak = 0
                last = s[-1]
                for c in reversed(s):
                    if c == last:
                        streak += 1
                    else:
                        break
                if streak >= 2:
                    fallback = "Xỉu" if tx_list[-1] == "Tài" else "Tài"
                    return fallback, 52, {}, {}, "Fallback đảo cầu"
                return decode(s[-1]), 52, {}, {}, "Fallback tiếp trend"
            return "Tài", 50, {}, {}, "Mặc định"
        
        winner = votes.most_common(1)[0]
        pred = winner[0]
        total = sum(votes.values())
        conf = min(winner[1]/total*100, 95) if total else 50
        
        top = sorted([(k, v) for k, v in reasons.items()],
                    key=lambda x: self.weights.get(x[0].split(":")[0], 1), reverse=True)[:5]
        reason_str = " | ".join([v for k, v in top])
        
        return pred, round(conf, 1), details, self.weights.copy(), reason_str

# Khởi tạo AI
ai_engine = SuperEnsemble()
load_data()

# ================= PREDICT =================
def predict(tx_list):
    if len(tx_list) < MIN_PHIEN:
        return "Chờ dữ liệu", "0%", {}, {}, "Chưa đủ 15 phiên"
    pred, conf, details, weights, reason = ai_engine.predict(tx_list)
    return pred, f"{conf}%", details, weights, reason

# ================= UPDATE STATS =================
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
        stats["max_cd"] = max(stats["max_cd"], stats["cd"])
    else:
        stats["sai"] += 1
        stats["cs"] += 1
        stats["cd"] = 0
        stats["max_cs"] = max(stats["max_cs"], stats["cs"])
    
    lich_su.append({
        "phien": phien_id,
        "du_doan": _prev_pred,
        "ket_qua": actual,
        "dung": "✅" if dung else "❌",
        "time": datetime.now().strftime("%H:%M:%S")
    })
    
    if len(lich_su) > 200:
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

# ================= NHẬN DIỆN CẦU =================
def nhan_dien_cau(tx_list):
    cau_text, cau_list = CauDetector.detect(tx_list)
    return cau_text, cau_list

# ================= GET DATA =================
def get_data():
    for url in API_LIST:
        try:
            res = requests.get(url, timeout=3)
            if res.status_code != 200:
                continue
            
            data = res.json()
            if "data" not in data:
                continue
            
            info = data["data"]
            phien = int(info.get("Expect", 0))
            dice = [int(x) for x in info.get("OpenCode", "0,0,0").split(",")]
            tong = sum(dice)
            
            return phien, dice, tong
        except:
            continue
    return None

# ================= BACKGROUND =================
def background():
    global last_result, _prev_pred, history_tx, history_pt, history_id, _last_phien_processed
    
    last_phien = None
    
    while True:
        data = get_data()
        
        if data:
            phien, dice, tong = data
            
            # Kiểm tra trùng phiên - FIX LỖI LỊCH SỬ
            if phien == last_phien or phien == _last_phien_processed:
                time.sleep(1)
                continue
            
            ket = "Tài" if tong >= 11 else "Xỉu"
            tx = "T" if ket == "Tài" else "X"
            
            # Huấn luyện AI với dữ liệu TRƯỚC KHI thêm phiên mới
            if len(history_tx) >= MIN_PHIEN and _prev_pred and _prev_pred != "Chờ dữ liệu":
                ai_engine.update(ket, list(history_tx))
            
            # Cập nhật stats (so sánh dự đoán trước với kết quả hiện tại)
            update_stats(tx, phien)
            
            # Lưu history mới
            history_tx.append(ket)
            history_pt.append(tong)
            history_id.append(phien)
            
            # Đánh dấu phiên đã xử lý
            _last_phien_processed = phien
            last_phien = phien
            
            # Tính chuỗi
            streak, stype, max_tai, max_xiu = tinh_chuoi(list(history_tx))
            
            # Nhận diện cầu
            cau_text, cau_list = nhan_dien_cau(list(history_tx))
            
            # AI Predict cho phiên TIẾP THEO
            tx_list = list(history_tx)
            du_doan, do_tin_cay, model_confs, weights, phan_tich = predict(tx_list)
            _prev_pred = du_doan
            
            # Pattern
            pattern = encode(tx_list)[-25:] if len(tx_list) >= 25 else encode(tx_list)
            
            # Tỷ lệ thật
            td = stats["tong"]
            ty_le = f"{stats['dung']/td*100:.1f}%" if td else "0%"
            
            # Build result
            last_result = {
                "status": "success",
                "data": {
                    "Phien": phien,
                    "Xuc_xac_1": dice[0],
                    "Xuc_xac_2": dice[1],
                    "Xuc_xac_3": dice[2],
                    "Tong": tong,
                    "Ket": ket,
                    "Phien_hien_tai": phien + 1,
                    "Du_doan": du_doan,
                    "Do_tin_cay": do_tin_cay,
                    "Pattern": pattern,
                    "Cau": cau_text,
                    "Chi_tiet_cau": cau_list,
                    "Chuoi_hien_tai": f"{stype} x{streak}" if stype else "N/A",
                    "Max_chuoi_Tai": max_tai,
                    "Max_chuoi_Xiu": max_xiu,
                    "Ty_le_dung": ty_le,
                    "Thong_ke": f"Dung:{stats['dung']} Sai:{stats['sai']} Tong:{td}",
                    "AI_Models": model_confs,
                    "Trong_so": {k: round(v, 2) for k, v in weights.items()},
                    "Phan_tich": phan_tich,
                    "So_phien_hoc": len(tx_list),
                  
                }
            }
            
            # Save data
            save_data()
            
            # Log console
            print("\n" + "=" * 75)
            print(f"🎲 PHIÊN {phien} | {dice[0]}-{dice[1]}-{dice[2]} = {tong} [{ket}]")
            print("-" * 75)
            print(f"Pattern  : {pattern}")
            print(f"Cầu      : {cau_text}")
            print(f"Chuỗi    : {stype} x{streak}" if stype else "Chuỗi    : N/A")
            print(f"🔮 Dự đoán: >>> {du_doan} <<< ({do_tin_cay})")
            print(f"📊 Tỷ lệ  : {ty_le} ({stats['dung']}/{td})")
            print(f"🧠 Phân tích: {phan_tich}")
            if model_confs:
                active = {k: v for k, v in model_confs.items() if v > 0}
                print(f"🤖 AI ({len(active)}/20 models): {active}")
            print("=" * 75)
        
        time.sleep(2)

# ================= API =================
@app.route("/api/taixiumd5")
def api_main():
    return jsonify(last_result if last_result else {"status": "waiting"})

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
        "chuoi_hien_tai_dung": stats["cd"],
        "chuoi_hien_tai_sai": stats["cs"],
        "lich_su_20": list(lich_su)[-20:],
        "lich_su_50": list(lich_su)[-50:],
        "lich_su_day_du": list(lich_su)[-100:],
        "ai_weights": {k: round(v, 2) for k, v in ai_engine.weights.items()},
        "ai_performance": ai_engine.performance
    })

@app.route("/")
def home():
    return jsonify({"status": "ok", "data": last_result.get("data", {})})

@app.route("/api/thongke")
def api_thongke():
    tx_list = list(history_tx)
    if len(tx_list) < 10:
        return jsonify({"error": "Chua du du lieu", "so_phien": len(tx_list)})
    
    s = encode(tx_list)
    cau_text, cau_list = nhan_dien_cau(tx_list)
    
    return jsonify({
        "tong_phien": len(tx_list),
        "tai": s.count("T"),
        "xiu": s.count("X"),
        "tai_rate": f"{s.count('T')/len(s)*100:.1f}%",
        "pattern_20": s[-20:],
        "pattern_50": s[-50:] if len(s) >= 50 else s,
        "max_tai": stats["max_cd"],
        "max_xiu": stats["max_cs"],
        "cau_hien_tai": cau_text,
        "ai_active": len([m for m in ai_engine.weights.values() if m > 0.5]),
        "models_performance": ai_engine.performance
    })

@app.route("/api/cau")
def api_cau():
    tx_list = list(history_tx)
    cau_text, cau_list = nhan_dien_cau(tx_list)
    return jsonify({
        "cau": cau_text,
        "chi_tiet": cau_list,
        "so_phien_phan_tich": len(tx_list),
        "pattern": encode(tx_list)[-30:] if len(tx_list) >= 30 else encode(tx_list)
    })

# ================= RUN =================
if __name__ == "__main__":
    threading.Thread(target=background, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
