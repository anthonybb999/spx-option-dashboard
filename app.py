import math
import warnings
from arch import arch_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="SPX 多空雙向期權邊界與風控 Dashboard",
    page_icon="📊",
    layout="wide",
)


# ======================================================================
# 0. 密碼登入驗證（使用 st.secrets，避免明文寫死喺 Code）
# ======================================================================
def check_password() -> bool:
  if st.session_state.get("authenticated"):
    return True

  st.markdown("## 🔒 SPX 期權分析 Dashboard — 私人存取登入")
  st.caption("請輸入存取密碼以進入分析面板。")

  pwd = st.text_input("存取密碼", type="password", key="login_pwd_input")
  col_btn, _ = st.columns([1, 5])
  with col_btn:
    login_clicked = st.button("登入", type="primary", use_container_width=True)

  if login_clicked:
    secret_pwd = str(st.secrets.get("APP_PASSWORD", "")).strip()
    if not secret_pwd:
      st.warning(
          "尚未於 Streamlit Secrets 設定 `APP_PASSWORD`。請先於 App"
          " Settings → Secrets 設定密碼。"
      )
    elif pwd == secret_pwd:
      st.session_state["authenticated"] = True
      st.rerun()
    else:
      st.error("密碼錯誤，請再試一次。")

  return False


if not check_password():
  st.stop()


# ======================================================================
# 1. 核心模型：雙核合適度 + 雙邊 0.16Δ 結構 + 75% 防線 + Long/Short 推薦
# ======================================================================
def calculate_dual_suitability_and_recommendation(
    bias_ema20,
    trend_ma_diff,
    rsi,
    ret_3d,
    vrp_ratio,
    vix_inversion,
    vix_spike,
    vix,
    spx_close,
    sma200,
    streak=0,
    streak_type="跌",
    put_016_strike=None,
    put_016_iv=None,
    put_016_price=None,
    call_016_strike=None,
    call_016_iv=None,
    call_016_price=None,
):
  f_bias = (bias_ema20 - 0.4012) / 2.0331
  f_trend = (trend_ma_diff - 0.6310) / 2.1736
  f_rsi = (rsi - 55.4287) / 15.9171
  f_ret3d = (ret_3d - 0.1494) / 1.8300
  f_vrp = (vrp_ratio - 1.3361) / 0.3615
  f_inversion = (vix_inversion - 0.8990) / 0.0622
  f_spike = (vix_spike - 0.2937) / 8.0915

  put_struct_adj = 0.0
  call_struct_adj = 0.0
  if put_016_iv is not None and call_016_iv is not None:
    avg_016_iv = (put_016_iv + call_016_iv) / 2.0
    iv_vs_vix_diff = (avg_016_iv - vix) / 100.0
    put_struct_adj += -0.15 * iv_vs_vix_diff
    call_struct_adj += -0.15 * iv_vs_vix_diff

  if (
      put_016_strike is not None
      and call_016_strike is not None
      and spx_close > 0
  ):
    down_dist_pct = (spx_close - put_016_strike) / spx_close * 100.0
    up_dist_pct = (call_016_strike - spx_close) / spx_close * 100.0
    asym_pct = (down_dist_pct - up_dist_pct) / 100.0
    put_struct_adj += 0.20 * asym_pct
    call_struct_adj += -0.20 * asym_pct

  z_put = (
      +1.0944
      + (+0.3463 * f_bias)
      + (-0.0038 * f_trend)
      + (+0.0489 * f_rsi)
      + (-0.1010 * f_ret3d)
      + (+0.0121 * f_vrp)
      + (-0.1510 * f_inversion)
      + (+0.0212 * f_spike)
      + put_struct_adj
  )
  short_put_win = round(1.0 / (1.0 + np.exp(-z_put)) * 100.0, 1)

  z_call = (
      +1.0532
      + (-0.2274 * f_bias)
      + (+0.1797 * f_trend)
      + (+0.4364 * f_rsi)
      + (+0.2464 * f_ret3d)
      + (+0.2533 * f_vrp)
      + (-0.4988 * f_inversion)
      + (+0.0533 * f_spike)
      + call_struct_adj
  )
  short_call_win = round(1.0 / (1.0 + np.exp(-z_call)) * 100.0, 1)

  long_put_touch = round(100.0 - short_put_win, 1)
  long_call_touch = round(100.0 - short_call_win, 1)

  THRESHOLD = 75.0
  put_pass = short_put_win >= THRESHOLD
  call_pass = short_call_win >= THRESHOLD
  put_gap = short_put_win - THRESHOLD

  recommend = ""
  reasons = []
  rejections = []

  if vix >= 32.0 or vix_spike >= 40.0 or vix_inversion >= 1.08:
    recommend = "🛑 【強制空倉 (EXTREME PANIC CASH)】"
    reasons = [
        (
            f"恐慌指標失控：VIX 達 {vix:.1f}，期限比"
            f" {vix_inversion:.3f}，市場流動性瀕臨斷裂。"
        ),
        "此環境下 IV 易出現極端膨脹，任何賣方 Spread 均有較高穿價風險。",
        "做買方亦面臨高溢價滑點磨損，空倉觀望為較審慎做法。",
    ]
    rejections = [
        "Short Put Spread: 接飛刀隨時穿價。",
        "Short Call Spread: 極端波動下容易出現急彈反抽夾死。",
        "Long Put Spread: 極端恐慌下期權溢價偏貴，急彈易快速磨損。",
        "Long Call Spread: 恐慌未止下盲目追反彈容易直接質低。",
        "Iron Condor / Butterfly: 雙向震幅過大，非區間市，易爆腳。",
    ]
  elif spx_close < sma200 or vix_inversion > 1.0:
    if rsi <= 35.0 and streak_type == "跌" and streak >= 4 and vix < 30.0:
      recommend = "🚀 【LONG CALL SPREAD (超跌反彈博弈)】"
      reasons = [
          (
              f"連跌 {streak} 日且 RSI 進入超賣區"
              f" ({rsi:.1f})，短線空頭回補或反彈動能升高。"
          ),
          (
              f"賣方 Put Spread 勝率未達 75% ({short_put_win:.1f}%)，改用有限風險"
              " Debit Spread 控死最大虧損。"
          ),
          (
              f"留意 200SMA ({sma200:.1f}) 與上方阻力，反彈達標宜分批鎖利。"
          ),
      ]
      rejections = [
          (
              f"Short Put Spread: 下方勝率不足 75% ({short_put_win:.1f}%)，賣方接飛刀風險高。"
          ),
          "Short Call Spread: 超賣臨界下逆勢造頂易被夾。",
          "Long Put Spread: 連跌多日且超賣，追沽易遇反抽倒蝕。",
          "Iron Condor / Butterfly: 趨勢破位且波幅高，不宜做中性盤整。",
          "空倉: 若已鎖定小注風險，可作為反彈博弈之替代方案。",
      ]
    elif spx_close < sma200 and rsi > 40.0 and vix < 30.0:
      recommend = "🐻 【LONG PUT SPREAD (順勢下行博弈)】"
      reasons = [
          (
              f"現價 ({spx_close:.1f}) 跌破 200SMA ({sma200:.1f}) 且 RSI 處於"
              f" {rsi:.1f}（未極度超賣），下行趨勢具備延續性。"
          ),
          (
              f"下方 Put 存活率未達 75% ({short_put_win:.1f}%)，直接做 Short Put"
              " 風險偏高，改用 Long Put Spread 鎖定最大成本。"
          ),
          "需留意隨時出現跌深反彈，設好止盈止損紀律。",
      ]
      rejections = [
          "Short Put Spread: 跌浪未止且勝率未達 75%，賣方風險高。",
          "Short Call Spread: 反彈波動大，上方防守易受壓。",
          "Long Call Spread: 尚未出現超賣反彈訊號，逆勢做多勝率較低。",
          "Iron Condor / Butterfly: 趨勢未穩，中性策略易爆腳。",
          "空倉: 適合不想承受方向波幅者維持觀望。",
      ]
    else:
      recommend = "🛑 【強制空倉 / 等絕望棒 (WAIT CAPITULATION)】"
      reasons = [
          (
              f"現價 ({spx_close:.1f}) 相對 200SMA ({sma200:.1f}) 與 VIX 期限比"
              f" ({vix_inversion:.3f}) 處於弱勢或不穩結構，且未見明確絕望棒或理想"
              " Long 切入點。"
          ),
          (
              f"下方 Put 存活率僅 {short_put_win:.1f}% (未達 75% 防線"
              f" {put_gap:+.1f}%)。"
          ),
          "追沽易遇反抽，撈底又面臨阻力，等形態走穩再定。",
      ]
      rejections = [
          (
              f"Short Put Spread: 下方勝率不足 75% ({short_put_win:.1f}%)，左側抄底勝率極低。"
          ),
          (
              f"Short Call Spread: 上方勝率僅 {short_call_win:.1f}%，易遇反彈夾死。"
          ),
          "Long Put Spread: 當前位置追跌盈虧比與確定性未夠理想。",
          "Long Call Spread: 尚未確認止跌絕望棒，撈底易反覆捱打。",
          "Iron Condor / Butterfly: 波動劇烈且趨勢破位，嚴禁做中性盤整策略。",
      ]
  elif put_pass and call_pass:
    if vix < 14.0 and (45.0 <= rsi <= 55.0) and abs(bias_ema20) < 0.5:
      recommend = "🦋 【BUTTERFLY SPREAD (蝶式價差)】"
      reasons = [
          f"VIX ({vix:.1f}) 處於低波水準，市場波幅收縮。",
          "SPX 與均線接近，RSI 居中，適合以較小成本博窄幅區間。",
      ]
      rejections = [
          "Iron Condor: 低波環境下賣方保費偏薄。",
          "Short Put / Call Spread: 單邊溢價較少。",
          "Long Put / Call Spread: 低波橫盤缺乏大單邊動能，易蝕時間值。",
          "空倉: 可作為低波觀望選項。",
      ]
    else:
      recommend = "🦅 【IRON CONDOR (鐵鷹策略)】"
      reasons = [
          (
              "雙邊勝率皆達 75% 防線 (Put:"
              f" {short_put_win:.1f}% / Call: {short_call_win:.1f}%)。"
          ),
          f"VIX 期限結構健康 ({vix_inversion:.3f} <= 1.0)，雙向收租條件較佳。",
      ]
      rejections = [
          "Short Put / Call Spread: 單邊局限，未善用雙邊區間。",
          "Long Put / Call Spread: 區間市況不利單邊突破買方。",
          "Butterfly: 容錯區間較窄。",
          "空倉: 雙邊防線已達標。",
      ]
  elif put_pass and not call_pass and spx_close > sma200:
    recommend = "📈 【SHORT PUT SPREAD (Bull Put Spread)】"
    reasons = [
        f"下方 Put 存活率達 {short_put_win:.1f}%（>= 75% 防線），支撐較穩。",
        "SPX 穩守 200SMA 上方，多頭格局未壞。",
    ]
    rejections = [
        f"Short Call Spread: Call 邊勝率不足 75% ({short_call_win:.1f}%)。",
        "Long Put Spread: 多頭格局未壞，逆勢造空勝率較低。",
        "Long Call Spread: Debit Spread 需較大升幅先回本。",
        "Iron Condor: 上方壓力邊存活率低。",
        "Butterfly: 缺乏窄幅 Pin 條件。",
        "空倉: 下方勝率已跨越 75% 防線。",
    ]
  elif call_pass and not put_pass and rsi > 40.0:
    recommend = "📉 【SHORT CALL SPREAD (Bear Call Spread)】"
    reasons = [
        f"上方 Call 存活率達 {short_call_win:.1f}%（>= 75% 防線），上方阻力較明顯。",
        f"RSI 處於 {rsi:.1f}（未進入超賣區），上方壓制具延續性。",
    ]
    rejections = [
        f"Short Put Spread: 下方勝率不足 75% ({short_put_win:.1f}%)。",
        "Long Put Spread: 慢跌或震盪市況下直買 Put Spread 易受反抽磨損。",
        "Long Call Spread: 上方阻力較重且動能不足。",
        "Iron Condor: 下方支撐脆弱。",
        "Butterfly: 趨勢未見窄幅盤整。",
        "空倉: 上方勝率已達 75% 防線。",
    ]
  else:
    recommend = "🛑 【強制空倉 (CASH / WAIT)】"
    reasons = [
        (
            "雙邊存活率均低於 75% 防線 (Put:"
            f" {short_put_win:.1f}% / Call: {short_call_win:.1f}%)。"
        ),
        "市場處於震盪混沌期，保持現金等待更清晰訊號最穩陣。",
    ]
    rejections = [
        f"Short Put Spread: 下方勝率僅 {short_put_win:.1f}%，未達 75% 防線。",
        f"Short Call Spread: 上方勝率僅 {short_call_win:.1f}%，未達 75% 防線。",
        "Long Put Spread: 混沌震盪市況下方向延續性差。",
        "Long Call Spread: 缺乏明確向上突破動能。",
        "Iron Condor: 兩側防守空間被壓縮。",
        "Butterfly: 無法確立 Pin 點區間。",
    ]

  return (
      short_call_win,
      short_put_win,
      long_call_touch,
      long_put_touch,
      recommend,
      reasons,
      rejections,
  )


def derive_simulated_metrics(df_spx: pd.DataFrame, new_price: float):
  df_temp = df_spx.copy()
  df_temp.iloc[-1, df_temp.columns.get_loc("close")] = new_price

  ema20 = df_temp["close"].ewm(span=20, adjust=False).mean().iloc[-1]
  sma50 = df_temp["close"].rolling(50).mean().iloc[-1]
  sma200 = df_temp["close"].rolling(200).mean().iloc[-1]
  bias_ema20 = (new_price - ema20) / ema20 * 100.0
  trend_ma_diff = (ema20 - sma50) / sma50 * 100.0

  delta = df_temp["close"].diff()
  gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
  loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
  rs = gain / (loss + 1e-9)
  rsi = 100.0 - (100.0 / (1.0 + rs))

  ret_3d = (
      (new_price - df_temp["close"].iloc[-4]) / df_temp["close"].iloc[-4]
  ) * 100.0

  streak = 0
  change = df_temp["close"].diff()
  sign = 1 if change.iloc[-1] > 0 else -1
  for val in reversed(change.iloc[1:]):
    if (val > 0 and sign == 1) or (val < 0 and sign == -1):
      streak += 1
    else:
      break
  streak_type = "升" if sign == 1 else "跌"
  return (
      float(bias_ema20),
      float(trend_ma_diff),
      float(rsi),
      float(ret_3d),
      float(sma200),
      int(streak),
      streak_type,
  )


def auto_evaluate_spreads_and_recommend(
    spread_width,
    down_spread_price,
    up_spread_price,
    short_put_win,
    short_call_win,
    long_put_touch,
    long_call_touch,
    base_recommend,
    base_reasons,
    base_rejections,
    vix,
    vix_inversion,
    spx_close,
    sma200,
    rsi,
    put_016_strike=None,
    call_016_strike=None,
):
  w = max(0.01, float(spread_width))
  candidates = []
  rows = []

  if down_spread_price is not None and down_spread_price > 0:
    px_dn = min(w - 0.01, float(down_spread_price))
    sp_profit, sp_loss = px_dn, w - px_dn
    sp_win_p = short_put_win / 100.0
    sp_ev = (sp_win_p * sp_profit) - ((1.0 - sp_win_p) * sp_loss)
    sp_odds = sp_profit / sp_loss
    sp_pass = (
        short_put_win >= 75.0
        and vix < 32.0
        and vix_inversion <= 1.0
        and sp_ev > 0
    )

    lp_profit, lp_loss = w - px_dn, px_dn
    lp_win_p = long_put_touch / 100.0
    lp_ev = (lp_win_p * lp_profit) - ((1.0 - lp_win_p) * lp_loss)
    lp_odds = lp_profit / lp_loss
    lp_pass = spx_close < sma200 and rsi > 38.0 and vix < 30.0 and lp_ev > 0

    be_dn = (
        f"{put_016_strike - px_dn:.2f}" if put_016_strike is not None else "-"
    )
    rows.append({
        "策略": "下方 Short Put (Credit)",
        "報價": f"${px_dn:.2f}",
        "賺 / 蝕": f"+${sp_profit:.2f} / -${sp_loss:.2f}",
        "賠率": f"1:{sp_odds:.2f}",
        "勝率": f"{short_put_win:.1f}%",
        "期望值 EV": f"${sp_ev:+.2f}",
        "打和位": be_dn,
    })
    rows.append({
        "策略": "下方 Long Put (Debit)",
        "報價": f"${px_dn:.2f}",
        "賺 / 蝕": f"+${lp_profit:.2f} / -${lp_loss:.2f}",
        "賠率": f"1:{lp_odds:.2f}",
        "勝率": f"{long_put_touch:.1f}%",
        "期望值 EV": f"${lp_ev:+.2f}",
        "打和位": be_dn,
    })

    candidates.append({
        "name": "📈 【SHORT PUT SPREAD (Credit)】",
        "short_name": "Short Put Spread (Credit)",
        "ev": sp_ev,
        "win_pct": short_put_win,
        "odds": sp_odds,
        "profit": sp_profit,
        "loss": sp_loss,
        "pass_rule": sp_pass,
        "fail_reason": (
            f"勝率 {short_put_win:.1f}% 未達 75% 或期限/EV (${sp_ev:+.2f}) 不足"
        ),
    })
    candidates.append({
        "name": "🐻 【LONG PUT SPREAD (Debit)】",
        "short_name": "Long Put Spread (Debit)",
        "ev": lp_ev,
        "win_pct": long_put_touch,
        "odds": lp_odds,
        "profit": lp_profit,
        "loss": lp_loss,
        "pass_rule": lp_pass,
        "fail_reason": f"趨勢/超賣條件未配合或 EV (${lp_ev:+.2f}) 不足",
    })

  if up_spread_price is not None and up_spread_price > 0:
    px_up = min(w - 0.01, float(up_spread_price))
    sc_profit, sc_loss = px_up, w - px_up
    sc_win_p = short_call_win / 100.0
    sc_ev = (sc_win_p * sc_profit) - ((1.0 - sc_win_p) * sc_loss)
    sc_odds = sc_profit / sc_loss
    sc_pass = (
        short_call_win >= 75.0
        and rsi > 40.0
        and vix < 32.0
        and vix_inversion <= 1.0
        and sc_ev > 0
    )

    lc_profit, lc_loss = w - px_up, px_up
    lc_win_p = long_call_touch / 100.0
    lc_ev = (lc_win_p * lc_profit) - ((1.0 - lc_win_p) * lc_loss)
    lc_odds = lc_profit / lc_loss
    lc_pass = vix < 30.0 and lc_ev > 0 and (spx_close >= sma200 or rsi <= 36.0)

    be_up = (
        f"{call_016_strike + px_up:.2f}"
        if call_016_strike is not None
        else "-"
    )
    rows.append({
        "策略": "上方 Short Call (Credit)",
        "報價": f"${px_up:.2f}",
        "賺 / 蝕": f"+${sc_profit:.2f} / -${sc_loss:.2f}",
        "賠率": f"1:{sc_odds:.2f}",
        "勝率": f"{short_call_win:.1f}%",
        "期望值 EV": f"${sc_ev:+.2f}",
        "打和位": be_up,
    })
    rows.append({
        "策略": "上方 Long Call (Debit)",
        "報價": f"${px_up:.2f}",
        "賺 / 蝕": f"+${lc_profit:.2f} / -${lc_loss:.2f}",
        "賠率": f"1:{lc_odds:.2f}",
        "勝率": f"{long_call_touch:.1f}%",
        "期望值 EV": f"${lc_ev:+.2f}",
        "打和位": be_up,
    })

    candidates.append({
        "name": "📉 【SHORT CALL SPREAD (Credit)】",
        "short_name": "Short Call Spread (Credit)",
        "ev": sc_ev,
        "win_pct": short_call_win,
        "odds": sc_odds,
        "profit": sc_profit,
        "loss": sc_loss,
        "pass_rule": sc_pass,
        "fail_reason": (
            f"勝率 {short_call_win:.1f}% 未達 75% 或反彈/EV (${sc_ev:+.2f}) 不足"
        ),
    })
    candidates.append({
        "name": "🚀 【LONG CALL SPREAD (Debit)】",
        "short_name": "Long Call Spread (Debit)",
        "ev": lc_ev,
        "win_pct": long_call_touch,
        "odds": lc_odds,
        "profit": lc_profit,
        "loss": lc_loss,
        "pass_rule": lc_pass,
        "fail_reason": f"趨勢/反彈結構未配合或 EV (${lc_ev:+.2f}) 不足",
    })

  if not candidates:
    return base_recommend, base_reasons, base_rejections, rows

  valid = [c for c in candidates if c["pass_rule"]]
  valid.sort(key=lambda x: x["ev"], reverse=True)

  if valid:
    best = valid[0]
    final_rec = best["name"]
    final_reasons = [
        (
            f"{best['short_name']} 通過風控且具備最高正 EV"
            f" (${best['ev']:+.2f}/股)。"
        ),
        (
            f"模型勝率 {best['win_pct']:.1f}%，賠率"
            f" 1:{best['odds']:.2f}（賺 ${best['profit']:.2f} / 蝕"
            f" ${best['loss']:.2f}）。"
        ),
    ]
    final_rejections = [
        f"{c['short_name']}: {c['fail_reason']}"
        for c in candidates
        if c["short_name"] != best["short_name"]
    ]
    return final_rec, final_reasons, final_rejections, rows
  else:
    final_rec = "🛑 【強制空倉 / 觀望 (未達風控或正 EV 門檻)】"
    final_reasons = [
        "上下方 Credit 與 Debit 均未同時通過「市況風控 + 正 EV」，維持觀望最穩陣。",
        f"現時 VIX {vix:.2f}、期限比 {vix_inversion:.3f}。",
    ]
    final_rejections = [
        f"{c['short_name']}: {c['fail_reason']}" for c in candidates
    ]
    return final_rec, final_reasons, final_rejections, rows


# ======================================================================
# 2. 快取市場數據與 GJR-GARCH 預測（TTL = 15 分鐘）
# ======================================================================
@st.cache_data(ttl=900, show_spinner="正在抓取 SPX、VIX、VIX3M 並計算 GJR-GARCH...")
def load_market_and_garch():
  tickers = ["^GSPC", "^VIX", "^VIX3M"]
  raw_data = yf.download(tickers, period="2y", interval="1d", progress=False)

  if isinstance(raw_data.columns, pd.MultiIndex):
    close_spx = raw_data["Close"]["^GSPC"]
    close_vix = raw_data["Close"]["^VIX"]
    close_vix3m = raw_data["Close"]["^VIX3M"]
  else:
    close_spx = raw_data["^GSPC"]
    close_vix = raw_data["^VIX"]
    close_vix3m = raw_data["^VIX3M"]

  spx = pd.DataFrame(
      {"close": close_spx, "vix": close_vix, "vix3m": close_vix3m}
  ).dropna()

  spx["ema20"] = spx["close"].ewm(span=20, adjust=False).mean()
  spx["sma50"] = spx["close"].rolling(50).mean()
  spx["sma200"] = spx["close"].rolling(200).mean()
  spx["bias_ema20"] = (spx["close"] - spx["ema20"]) / spx["ema20"] * 100
  spx["trend_ma_diff"] = (spx["ema20"] - spx["sma50"]) / spx["sma50"] * 100

  delta = spx["close"].diff()
  gain = delta.where(delta > 0, 0).rolling(14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
  rs = gain / (loss + 1e-9)
  spx["rsi"] = 100 - (100 / (1 + rs))

  spx["ret_3d"] = spx["close"].pct_change(3) * 100
  spx["ret_1d"] = spx["close"].pct_change()
  spx["rv20"] = spx["ret_1d"].rolling(20).std() * np.sqrt(252) * 100
  spx["vrp_ratio"] = spx["vix"] / (spx["rv20"] + 1e-5)
  spx["vix_inversion"] = spx["vix"] / spx["vix3m"]
  spx["vix_sma20"] = spx["vix"].rolling(20).mean()
  spx["vix_spike"] = (spx["vix"] - spx["vix_sma20"]) / spx["vix_sma20"] * 100

  change = spx["close"].diff()
  streak = 0
  sign = 1 if change.iloc[-1] > 0 else -1
  for val in reversed(change.iloc[1:]):
    if (val > 0 and sign == 1) or (val < 0 and sign == -1):
      streak += 1
    else:
      break
  streak_type = "升" if sign == 1 else "跌"

  clean_spx = spx.dropna()
  latest = clean_spx.iloc[-1].to_dict()
  latest["streak"] = int(streak)
  latest["streak_type"] = streak_type

  returns = (np.log(clean_spx["close"] / clean_spx["close"].shift(1)) * 100).dropna()
  gjr_garch = arch_model(
      returns, vol="Garch", p=1, o=1, q=1, mean="Constant", dist="normal"
  )
  res = gjr_garch.fit(disp="off", show_warning=False)
  forecast = res.forecast(horizon=10, reindex=False)
  daily_var_forecasts = forecast.variance.iloc[-1].values

  return clean_spx, latest, daily_var_forecasts


# ======================================================================
# 3. 網頁介面：頂部欄 + 側邊情景控制 + 分組輸入
# ======================================================================
top_l, top_r = st.columns([8, 2])
with top_l:
  st.title("📊 SPX 多空雙向期權邊界與風控 Dashboard")
with top_r:
  st.write("")
  if st.button("登出", use_container_width=True):
    st.session_state["authenticated"] = False
    st.rerun()

spx, latest, daily_var_forecasts = load_market_and_garch()
current_price = float(latest["close"])
garch_ann_vol = float(np.sqrt(daily_var_forecasts[0]) * np.sqrt(252))
garch_5d_1sd_pct = float(np.sqrt(np.sum(daily_var_forecasts[:5])))

with st.sidebar:
  st.header("🎛️ 情景與波幅模擬")
  use_exact = st.checkbox("直接輸入指定 SPX 價位", value=False)
  if use_exact:
    sim_price = st.number_input(
        "指定 SPX 價位", value=round(current_price, 2), step=5.0
    )
  else:
    spx_delta = st.slider(
        "SPX 變動點數", min_value=-250.0, max_value=250.0, value=0.0, step=5.0
    )
    sim_price = current_price + spx_delta

  vix_chg_pct = st.slider(
      "VIX 變動 %", min_value=-40.0, max_value=60.0, value=0.0, step=1.0
  )
  spread_width = st.number_input(
      "Spread 闊度點數", min_value=1.0, max_value=200.0, value=10.0, step=5.0
  )

  st.divider()
  st.subheader("🟢 下方資料 (Put 0.16Δ + Spread)")
  p_k_in = st.number_input("下方 5D Put 0.16Δ Strike", value=0.0, step=5.0)
  p_iv_in = st.number_input("下方 5D Put 0.16Δ IV %", value=0.0, step=0.5)
  p_px_in = st.number_input("下方 5D Put 0.16Δ 價錢 $", value=0.0, step=0.1)
  dn_sp_in = st.number_input("下方 Spread 價錢 $", value=0.0, step=0.1)

  st.divider()
  st.subheader("🔴 上方資料 (Call 0.16Δ + Spread)")
  c_k_in = st.number_input("上方 5D Call 0.16Δ Strike", value=0.0, step=5.0)
  c_iv_in = st.number_input("上方 5D Call 0.16Δ IV %", value=0.0, step=0.5)
  c_px_in = st.number_input("上方 5D Call 0.16Δ 價錢 $", value=0.0, step=0.1)
  up_sp_in = st.number_input("上方 Spread 價錢 $", value=0.0, step=0.1)

# 轉換任選輸入（0 代表未填）
p_k = float(p_k_in) if p_k_in > 0 else None
p_iv = float(p_iv_in) if p_iv_in > 0 else None
p_px = float(p_px_in) if p_px_in > 0 else None
dn_sp = float(dn_sp_in) if dn_sp_in > 0 else None

c_k = float(c_k_in) if c_k_in > 0 else None
c_iv = float(c_iv_in) if c_iv_in > 0 else None
c_px = float(c_px_in) if c_px_in > 0 else None
up_sp = float(up_sp_in) if up_sp_in > 0 else None

# 計算模擬後指標與策略建議
sim_vix = max(5.0, float(latest["vix"]) * (1.0 + vix_chg_pct / 100.0))
sim_vix_inv = sim_vix / float(latest["vix3m"])
sim_vix_spike = (
    (sim_vix - float(latest["vix_sma20"])) / float(latest["vix_sma20"]) * 100.0
)
sim_vrp = sim_vix / (float(latest["rv20"]) + 1e-5)

(
    sim_bias,
    sim_trend,
    sim_rsi,
    sim_ret3d,
    sim_sma200,
    sim_streak,
    sim_streak_type,
) = derive_simulated_metrics(spx, sim_price)

(
    call_win,
    put_win,
    call_touch,
    put_touch,
    base_rec,
    base_reasons,
    base_rejections,
) = calculate_dual_suitability_and_recommendation(
    bias_ema20=sim_bias,
    trend_ma_diff=sim_trend,
    rsi=sim_rsi,
    ret_3d=sim_ret3d,
    vrp_ratio=sim_vrp,
    vix_inversion=sim_vix_inv,
    vix_spike=sim_vix_spike,
    vix=sim_vix,
    spx_close=sim_price,
    sma200=sim_sma200,
    streak=sim_streak,
    streak_type=sim_streak_type,
    put_016_strike=p_k,
    put_016_iv=p_iv,
    put_016_price=p_px,
    call_016_strike=c_k,
    call_016_iv=c_iv,
    call_016_price=c_px,
)

final_rec, final_reasons, final_rejections, sp_rows = (
    auto_evaluate_spreads_and_recommend(
        spread_width=spread_width,
        down_spread_price=dn_sp,
        up_spread_price=up_sp,
        short_put_win=put_win,
        short_call_win=call_win,
        long_put_touch=put_touch,
        long_call_touch=call_touch,
        base_recommend=base_rec,
        base_reasons=base_reasons,
        base_rejections=base_rejections,
        vix=sim_vix,
        vix_inversion=sim_vix_inv,
        spx_close=sim_price,
        sma200=sim_sma200,
        rsi=sim_rsi,
        put_016_strike=p_k,
        call_016_strike=c_k,
    )
)

# ======================================================================
# 4. 主畫面呈現：KPI 指標列 + 決策建議 + 圖表 + 結構與 Spread 表
# ======================================================================
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(
    "SPX 價位",
    f"{sim_price:.2f}",
    delta=f"{sim_price - current_price:+.2f} (原 {current_price:.2f})",
)
m2.metric("VIX", f"{sim_vix:.2f}", delta=f"{vix_chg_pct:+.1f}%")
m3.metric("VIX / VIX3M 期限比", f"{sim_vix_inv:.3f}")
m4.metric("RSI (14)", f"{sim_rsi:.1f}")
m5.metric("200SMA", f"{sim_sma200:.2f}")

st.info(f"**最新建議策略：{final_rec}**")
for idx, r in enumerate(final_reasons, 1):
  st.markdown(f"- {r}")

with st.expander("查看其餘策略淘汰原因"):
  for rej in final_rejections:
    st.markdown(f"- {rej}")

# 視覺化：75% 防線 + DTE 邊界平移
vol_scale = max(0.5, 1.0 + vix_chg_pct / 100.0)
dtes = [1, 3, 5, 7, 10]
put_bounds, call_bounds, dte_labels = [], [], []
for d in dtes:
  cum_std = np.sqrt(np.sum(daily_var_forecasts[:d]))
  mv = cum_std * (1.0 if d == 1 else 1.25) * vol_scale
  put_bounds.append(round(sim_price * (1 - mv / 100.0) / 5.0) * 5)
  call_bounds.append(round(sim_price * (1 + mv / 100.0) / 5.0) * 5)
  dte_labels.append(f"{d}D")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.0))

bar_labels = ["Short Put 存活", "Short Call 存活"]
bar_vals = [put_win, call_win]
bar_colors = [
    "#2ecc71" if put_win >= 75 else "#e67e22",
    "#3498db" if call_win >= 75 else "#e74c3c",
]
ax1.barh(bar_labels, bar_vals, color=bar_colors, height=0.45)
ax1.axvline(75, color="navy", linestyle="--", linewidth=1.5, label="75% 防線")
ax1.set_xlim(0, 100)
ax1.set_title("75% 實戰勝率防線檢測")
ax1.legend(loc="lower right")
for i, v in enumerate(bar_vals):
  ax1.text(v + 1.5, i, f"{v:.1f}%", va="center", fontweight="bold")

ax2.plot(
    dte_labels, call_bounds, marker="^", color="#e74c3c", label="上方 Call 邊界"
)
ax2.axhline(sim_price, color="black", linestyle="-.", label="SPX 現價")
ax2.plot(
    dte_labels, put_bounds, marker="v", color="#2ecc71", label="下方 Put 邊界"
)
ax2.set_title("GJR-GARCH 行使價安全區間平移")
ax2.legend(loc="best")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# 雙邊 0.16Δ 結構摘要
if any(x is not None for x in [p_k, p_iv, p_px, c_k, c_iv, c_px]):
  st.subheader("🧭 5D 雙邊 0.16Δ 市場結構摘要")
  s_cols = st.columns(2)
  with s_cols[0]:
    p_dist = (
        f" (-{(sim_price - p_k) / sim_price * 100:.2f}%)"
        if p_k is not None
        else ""
    )
    st.markdown(
        f"**🟢 下方 Put 0.16Δ：** Strike `{p_k or '-'}`{p_dist} │ IV"
        f" `{p_iv if p_iv is not None else '-'}%` │ 價錢"
        f" `${p_px if p_px is not None else '-'}`"
    )
  with s_cols[1]:
    c_dist = (
        f" (+{(c_k - sim_price) / sim_price * 100:.2f}%)"
        if c_k is not None
        else ""
    )
    st.markdown(
        f"**🔴 上方 Call 0.16Δ：** Strike `{c_k or '-'}`{c_dist} │ IV"
        f" `{c_iv if c_iv is not None else '-'}%` │ 價錢"
        f" `${c_px if c_px is not None else '-'}`"
    )

  if p_iv is not None and c_iv is not None:
    avg_iv = (p_iv + c_iv) / 2.0
    iv_5d = avg_iv * math.sqrt(5.0 / 252.0)
    st.caption(
        f"Skew (Put-Call): {p_iv - c_iv:+.1f}% │ 平均 0.16Δ IV: {avg_iv:.1f}%"
        f" vs VIX {sim_vix:.2f}% vs GJR-GARCH 年化"
        f" {garch_ann_vol * vol_scale:.1f}% │ 5D 隱含 ±{iv_5d:.2f}% vs GARCH"
        f" 1σ ±{garch_5d_1sd_pct * vol_scale:.2f}%"
    )

# Spread 雙向精算表
if sp_rows:
  st.subheader("🎲 上下方 Spread 自動雙向精算（Credit vs Debit）")
  st.dataframe(pd.DataFrame(sp_rows), use_container_width=True, hide_index=True)

st.caption(
    "注：本 Dashboard 為量化分析與情景模擬工具，僅供參考，不構成投資建議。"
)