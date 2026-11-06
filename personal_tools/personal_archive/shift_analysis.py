import openpyxl.worksheet._reader as reader
orig_cast = reader._cast_number
def safe_cast(value):
    try: return orig_cast(value)
    except: return value
reader._cast_number = safe_cast

import openpyxl, json
from datetime import datetime, date
from collections import defaultdict, Counter

wb = openpyxl.load_workbook('D:/微信资料/xwechat_files/wxid_1saara878wtd21_a804/msg/file/2026-07/2026年健康生活.xlsx', data_only=True)
ws = wb['详细记录表']

data = []
for row in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True):
    d = row[0]
    if d is None:
        continue
    if isinstance(d, str):
        dt = datetime.strptime(d, '%Y-%m-%d')
    elif isinstance(d, (datetime, date)):
        dt = d if isinstance(d, datetime) else datetime.combine(d, datetime.min.time())

    def num(v):
        return v if isinstance(v, (int, float)) else None

    data.append({
        'date': dt,
        'date_str': dt.strftime('%Y-%m-%d'),
        'weekday': dt.weekday(),
        'weekday_name': row[1],
        'date_type': row[2],
        'shift': row[3],
        'weight': num(row[4]),
        'sleep': num(row[5]),
        'run': num(row[6]),
        'alcohol': num(row[7]),
        'calories': num(row[8]),
        'cal_diff': num(row[9]),
    })

print(f"总记录: {len(data)} 天")
print(f"日期范围: {data[0]['date_str']} ~ {data[-1]['date_str']}")

# === 班次分布 ===
print("\n=== 班次分布 ===")
shift_counter = Counter(d['shift'] for d in data)
for s, cnt in shift_counter.most_common():
    print(f"  {s}: {cnt}天 ({cnt/len(data)*100:.1f}%)")

# === 各班次的指标均值 ===
print("\n=== 各班次指标对比 ===")
shift_data = defaultdict(list)
for d in data:
    shift_data[d['shift']].append(d)

shift_names = {'早': '早班', '中': '中班', '夜': '夜班', '休': '休息', '白': '白班'}
print(f"{'班次':6s} | {'天数':4s} | {'均体重':8s} | {'均睡眠':8s} | {'均跑步':8s} | {'运动率':6s} | {'均热量':8s} | {'均热量差':10s} | {'均饮酒':8s} | {'饮酒率':6s}")
print("-" * 110)
for shift in sorted(shift_data.keys(), key=lambda x: (x is None, x or '')):
    rows = shift_data[shift]
    n = len(rows)
    weights = [r['weight'] for r in rows if r['weight'] is not None]
    sleeps = [r['sleep'] for r in rows if r['sleep'] is not None]
    runs = [r['run'] for r in rows if r['run'] is not None]
    run_days = [r for r in rows if r['run'] is not None and r['run'] > 0]
    cals = [r['calories'] for r in rows if r['calories'] is not None]
    diffs = [r['cal_diff'] for r in rows if r['cal_diff'] is not None]
    alcs = [r['alcohol'] for r in rows if r['alcohol'] is not None]
    alc_days = [r for r in rows if r['alcohol'] is not None and r['alcohol'] > 0]

    avg_w = f"{sum(weights)/len(weights):.1f}" if weights else "-"
    avg_s = f"{sum(sleeps)/len(sleeps):.1f}" if sleeps else "-"
    avg_r = f"{sum(runs)/len(runs):.0f}" if runs else "-"
    run_pct = f"{len(run_days)}/{n}" if runs else "-"
    avg_c = f"{sum(cals)/len(cals):.0f}" if cals else "-"
    avg_d = f"{sum(diffs)/len(diffs):+.0f}" if diffs else "-"
    avg_a = f"{sum(alcs)/len(alcs):.0f}" if alcs else "-"
    alc_pct = f"{len(alc_days)}/{n}" if alcs else "-"

    label = shift_names.get(shift, str(shift))
    print(f"{label:6s} | {n:4d} | {avg_w:8s} | {avg_s:8s} | {avg_r:8s} | {run_pct:6s} | {avg_c:8s} | {avg_d:10s} | {avg_a:8s} | {alc_pct:6s}")

# === 班次与星期交叉 ===
print("\n=== 班次 x 星期 交叉表 ===")
weekday_names = ['周一','周二','周三','周四','周五','周六','周日']
cross = defaultdict(lambda: defaultdict(int))
for d in data:
    cross[d['shift']][d['weekday']] += 1

header = "班次 | " + " | ".join(f"{wn:4s}" for wn in weekday_names) + " | 合计"
print(header)
print("-" * len(header))
for shift in sorted(cross.keys(), key=lambda x: (x is None, x or '')):
    vals = [cross[shift][wd] for wd in range(7)]
    total = sum(vals)
    label = shift_names.get(shift, str(shift))
    print(f"{label:4s} | " + " | ".join(f"{v:4d}" for v in vals) + f" | {total:4d}")

# === 不同班次次日的体重变化 ===
print("\n=== 不同班次对次日体重变化的影响 ===")
weight_changes = []
for i in range(1, len(data)):
    if data[i]['weight'] is not None and data[i-1]['weight'] is not None:
        wc = round(data[i]['weight'] - data[i-1]['weight'], 2)
        weight_changes.append({
            'date': data[i]['date_str'],
            'weight_change': wc,
            'prev_shift': data[i-1]['shift'],
            'today_shift': data[i]['shift'],
            'prev_sleep': data[i-1]['sleep'],
            'prev_run': data[i-1]['run'],
            'prev_alcohol': data[i-1]['alcohol'],
            'prev_cal_diff': data[i-1]['cal_diff'],
        })

for shift in sorted(set(w['prev_shift'] for w in weight_changes), key=lambda x: (x is None, x or '')):
    subset = [w for w in weight_changes if w['prev_shift'] == shift]
    if subset:
        avg = sum(w['weight_change'] for w in subset) / len(subset)
        avg_sleep = sum(w['prev_sleep'] for w in subset if w['prev_sleep']) / max(1, len([w for w in subset if w['prev_sleep']]))
        avg_run = sum(w['prev_run'] for w in subset if w['prev_run']) / max(1, len([w for w in subset if w['prev_run']]))
        avg_diff = sum(w['prev_cal_diff'] for w in subset if w['prev_cal_diff']) / max(1, len([w for w in subset if w['prev_cal_diff']]))
        label = shift_names.get(shift, str(shift))
        print(f"  {label}后次日: n={len(subset):3d}, 均变化={avg:+.3f}kg | 均睡眠={avg_sleep:.1f}h | 均跑步={avg_run:.0f}min | 均热量差={avg_diff:+.0f}")

# === 月度班次分布 ===
print("\n=== 月度班次分布 ===")
monthly_shift = defaultdict(lambda: defaultdict(int))
for d in data:
    monthly_shift[d['date'].month][d['shift']] += 1

month_names = ['一月','二月','三月','四月','五月','六月','七月','八月','九月','十月','十一月','十二月']
for m in range(1, 8):
    if m not in monthly_shift:
        continue
    parts = []
    for shift in sorted(monthly_shift[m].keys(), key=lambda x: (x is None, x or '')):
        label = shift_names.get(shift, shift)
        parts.append(f"{label}{monthly_shift[m][shift]}天")
    print(f"  {month_names[m-1]}: {' | '.join(parts)}")

# === 班次轮换规律 ===
print("\n=== 班次轮换序列(前30天 + 最近30天) ===")
print("前30天:")
for d in data[:30]:
    print(f"  {d['date_str']} {d['weekday_name']} 班次={d['shift']} 睡眠={d['sleep']} 跑步={d['run']} 体重={d['weight']}")
print("\n最近30天:")
for d in data[-30:]:
    print(f"  {d['date_str']} {d['weekday_name']} 班次={d['shift']} 睡眠={d['sleep']} 跑步={d['run']} 体重={d['weight']}")

# === 班次与睡眠关系深入 ===
print("\n=== 各班次睡眠详情 ===")
for shift in sorted(shift_data.keys(), key=lambda x: (x is None, x or '')):
    rows = shift_data[shift]
    sleeps = [r['sleep'] for r in rows if r['sleep'] is not None]
    if sleeps:
        avg = sum(sleeps) / len(sleeps)
        mn = min(sleeps)
        mx = max(sleeps)
        # 分布
        buckets = {'<4h': 0, '4-5h': 0, '5-6h': 0, '6-7h': 0, '7-8h': 0, '8h+': 0}
        for s in sleeps:
            if s < 4: buckets['<4h'] += 1
            elif s < 5: buckets['4-5h'] += 1
            elif s < 6: buckets['5-6h'] += 1
            elif s < 7: buckets['6-7h'] += 1
            elif s < 8: buckets['7-8h'] += 1
            else: buckets['8h+'] += 1
        label = shift_names.get(shift, str(shift))
        print(f"  {label}: 均值{avg:.1f}h | 最短{mn:.1f}h | 最长{mx:.1f}h")
        print(f"    分布: {buckets}")

# Export for charts
print("\n===CHART_DATA===")
chart = {
    'shift_dist': [[shift_names.get(s, str(s)), shift_counter[s]] for s in sorted(shift_counter.keys(), key=lambda x: (x is None, x or ''))],
    'shift_metrics': {},
    'shift_weekday': {shift_names.get(s, str(s)): [cross[s][wd] for wd in range(7)] for s in sorted(cross.keys(), key=lambda x: (x is None, x or ''))},
    'monthly_shift': {month_names[m-1]: {shift_names.get(k, str(k)): v for k, v in monthly_shift[m].items()} for m in sorted(monthly_shift.keys())},
    'shift_weight_change': {},
    'shift_sleep_detail': {},
}
for shift in sorted(shift_data.keys(), key=lambda x: (x is None, x or '')):
    rows = shift_data[shift]
    label = shift_names.get(shift, str(shift))
    weights = [r['weight'] for r in rows if r['weight'] is not None]
    sleeps = [r['sleep'] for r in rows if r['sleep'] is not None]
    runs = [r['run'] for r in rows if r['run'] is not None and r['run'] > 0]
    diffs = [r['cal_diff'] for r in rows if r['cal_diff'] is not None]
    cals = [r['calories'] for r in rows if r['calories'] is not None]
    alcs = [r['alcohol'] for r in rows if r['alcohol'] is not None and r['alcohol'] > 0]
    chart['shift_metrics'][label] = {
        'days': len(rows),
        'avg_weight': round(sum(weights)/len(weights), 1) if weights else None,
        'avg_sleep': round(sum(sleeps)/len(sleeps), 1) if sleeps else None,
        'avg_run': round(sum(runs)/len(runs)) if runs else 0,
        'run_days': len(runs),
        'run_pct': round(len(runs)/len(rows)*100, 1) if rows else 0,
        'avg_cal': round(sum(cals)/len(cals)) if cals else None,
        'avg_diff': round(sum(diffs)/len(diffs)) if diffs else None,
        'avg_alc': round(sum(alcs)/len(alcs)) if alcs else 0,
        'alc_days': len(alcs),
        'alc_pct': round(len(alcs)/len(rows)*100, 1) if rows else 0,
    }

for shift in sorted(set(w['prev_shift'] for w in weight_changes), key=lambda x: (x is None, x or '')):
    subset = [w for w in weight_changes if w['prev_shift'] == shift]
    if subset:
        avg = sum(w['weight_change'] for w in subset) / len(subset)
        label = shift_names.get(shift, str(shift))
        chart['shift_weight_change'][label] = {
            'n': len(subset),
            'avg_change': round(avg, 3),
        }

for shift in sorted(shift_data.keys(), key=lambda x: (x is None, x or '')):
    rows = shift_data[shift]
    label = shift_names.get(shift, str(shift))
    sleeps = [r['sleep'] for r in rows if r['sleep'] is not None]
    if sleeps:
        chart['shift_sleep_detail'][label] = {
            'avg': round(sum(sleeps)/len(sleeps), 1),
            'min': round(min(sleeps), 1),
            'max': round(max(sleeps), 1),
            'values': sorted([round(s, 1) for s in sleeps]),
        }

print(json.dumps(chart, ensure_ascii=False))
