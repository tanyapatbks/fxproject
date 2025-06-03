Multi-Currency Bagging AI Trading System
วิทยานิพนธ์ปริญญาโท: ระบบเทรด AI แบบ Multi-Currency Bagging สำหรับตลาด Forex
ภาพรวมโครงการ
โครงการนี้พัฒนาระบบ AI Trading ที่ใช้หลักการ Multi-Currency Bagging เพื่อพิสูจน์ว่าการวิเคราะห์หลายคู่เงินพร้อมกันมีประสิทธิภาพเหนือกว่าการเทรดแบบคู่เงินเดียว โดยมี Adaptive Exit Strategy เป็นหัวใจของระบบ

หลักการสำคัญ
🎯 Multi-Currency Bagging Approach
วิเคราะห์ 3 คู่เงินหลัก: EURUSD, GBPUSD, USDJPY
ใช้ cross-currency correlations และ multi-timeframe analysis
รวมข้อมูลจาก timeframe 1H (tactical) และ 4H (strategic)
🔄 Adaptive Exit Strategy
ปรัชญาการเทรด "รีบเก็บกำไรเมื่อเห็นโอกาส แต่อดทนรอเมื่อยังไม่เห็นผล":

t+1: หากมีกำไร → ปิดทันทีเพื่อล็อคกำไร
t+2: หากยังไม่มีกำไรที่ t+1 → รอดูที่ t+2, ถ้ามีแล้วปิดทันที
t+3: หากยังไม่มี → รอจนถึง t+3 แล้วปิดไม่ว่าจะกำไรหรือขาดทุน
🔒 การป้องกัน Data Leakage
Training Set: 2018-2020 (3 ปี) สำหรับการเรียนรู้โมเดล
Validation Set: 2021 (1 ปี) สำหรับการปรับแต่งและเลือกโมเดล
Test Set: 2022 (1 ปี) ถูกปกป้องอย่างเข้มงวด ใช้เพียงครั้งเดียวในการประเมินผลสุดท้าย
โมเดล AI ที่ใช้
CNN_LSTM Hybrid: จับ spatial patterns และ temporal dependencies
Temporal Fusion Transformer (TFT): attention mechanism สำหรับ multi-currency analysis
XGBoost: feature interpretability และ classical ML approach
Ensemble Strategy: รวมจุดแข็งของทั้ง 3 โมเดล
เป้าหมายประสิทธิภาพ
Win Rate: เป้าหมายมากกว่า 65%
Profit Factor: มากกว่า 1.5 (Gross Profit / Gross Loss)
Sharpe Ratio: สำหรับ risk-adjusted returns
เปรียบเทียบกับ single-pair strategies และ traditional trading methods
โครงสร้างโปรเจกต์
forex_ai_trading/
├── data/                    # ข้อมูล OHLCV ทั้ง 3 คู่เงิน
│   ├── EURUSD_1H.csv       # ข้อมูล EURUSD timeframe 1 ชั่วโมง
│   ├── EURUSD_4H.csv       # ข้อมูล EURUSD timeframe 4 ชั่วโมง
│   ├── GBPUSD_1H.csv       # ข้อมูล GBPUSD timeframe 1 ชั่วโมง
│   ├── GBPUSD_4H.csv       # ข้อมูล GBPUSD timeframe 4 ชั่วโมง
│   ├── USDJPY_1H.csv       # ข้อมูล USDJPY timeframe 1 ชั่วโมง
│   └── USDJPY_4H.csv       # ข้อมูล USDJPY timeframe 4 ชั่วโมง
├── main.py                  # ควบคุมระบบทั้งหมด
├── data_processor.py        # จัดการข้อมูล + ป้องกัน data leakage
├── models.py               # โมเดล CNN_LSTM, TFT, XGBoost + ensemble
├── trading_system.py       # adaptive exit strategy + evaluation
├── requirements.txt        # Python dependencies
└── README.md              # เอกสารโปรเจกต์

logs/                        # บันทึกการ training (สร้างอัตโนมัติ)
├── training_log_[timestamp].log
└── model_performance_[timestamp].json
การติดตั้งและใช้งาน
ข้อกำหนดระบบ
Python 3.11.9
MacBook Pro M3 Pro (หรือระบบที่รองรับ)
RAM อย่างน้อย 16GB สำหรับการ training โมเดลขนาดใหญ่
การติดตั้ง
bash
# 1. Clone หรือ download โปรเจกต์
cd forex_ai_trading

# 2. สร้าง virtual environment
python3.11 -m venv forex_env
source forex_env/bin/activate  # สำหรับ macOS/Linux

# 3. ติดตั้ง dependencies
pip install -r requirements.txt

# 4. วางไฟล์ข้อมูล CSV ในโฟลเดอร์ data/

# 5. รันการตรวจสอบข้อมูล
python data_processor.py

# 6. เริ่มการ training (เมื่อพร้อม)
python main.py
สมมติฐานการเทรด
Spread: 2 pips สำหรับทุกคู่เงิน
เกณฑ์กำไร: มากกว่า 2 pips (สุทธิหลังหัก spread)
ไม่มีค่าคอมมิชชั่น: ตามสมมติฐานของการวิจัย
บัญชี Standard: เหมาะสมกับนักลงทุนทั่วไป
การประเมินผลและเปรียบเทียบ
Metrics หลัก
Financial Performance: Win Rate, Profit Factor, Sharpe Ratio, Sortino Ratio, Maximum Drawdown
Strategy Efficiency: Average Holding Time, Early Exit Success Rate
Risk Management: Risk-adjusted returns, Downside risk analysis
Benchmarks
Single-pair trading strategies
Traditional technical analysis (MA Crossover, RSI-based)
Buy & Hold strategy
Random trading (Monte Carlo baseline)
การพัฒนาแบบเป็นขั้นตอน
โครงการนี้พัฒนาตามขั้นตอนที่ชัดเจน:

✅ Data Infrastructure: ระบบโหลดและป้องกัน data leakage
⏳ Feature Engineering: Technical indicators + cross-currency features
⏳ Adaptive Labeling: Exit strategy logic + multi-horizon labels
⏳ Model Development: CNN_LSTM, TFT, XGBoost training
⏳ Ensemble Integration: การรวมโมเดลและ dynamic weighting
⏳ Trading System: Backtesting และ performance evaluation
⏳ Final Validation: การประเมินครั้งเดียวกับ test set
หลักการวิจัยที่สำคัญ
การป้องกัน Data Leakage
Test set (2022) ถูกปกป้องอย่างเข้มงวดตลอดการพัฒนา
ใช้เฉพาะข้อมูล 2018-2021 ในการพัฒนาโมเดล
การประเมินสุดท้ายทำเพียงครั้งเดียวเพื่อความน่าเชื่อถือ
การบันทึกและติดตาม
Logging ครอบคลุมทุกขั้นตอนการ training
บันทึกพารามิเตอร์และผลลัพธ์อย่างละเอียด
สร้างกราฟเปรียบเทียบประสิทธิภาพโมเดล
ผลที่คาดหวัง
การวิจัยนี้มุ่งหวังที่จะพิสูจน์ว่า:

Multi-currency bagging มีประสิทธิภาพเหนือกว่า single-pair trading
Adaptive exit strategy ช่วยเพิ่มประสิทธิภาพการเทรด
การรวมโมเดลหลายแบบให้ผลลัพธ์ที่แข็งแกร่งกว่าโมเดลเดี่ยว
ระบบสามารถทำงานได้ดีในสภาวะตลาดที่แตกต่างกัน
หมายเหตุ: โครงการนี้เป็นงานวิจัยวิชาการ ไม่ใช่คำแนะนำการลงทุน การเทรดจริงต้องคำนึงถึงความเสี่ยงและปรึกษาผู้เชี่ยวชาญ

