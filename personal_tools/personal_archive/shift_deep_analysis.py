import openpyxl.worksheet._reader as reader
orig_cast = reader._cast_number
def safe_cast(value):
    try: return orig_cast(value)
    except: return value
reader._cast_number = safe_cast

import openpyxl, json
from datetime import datetime, date, timedelta

wb = openpyxl.load_workbook('D:/微信资料/xwechat_files/wxid_1saara878wtd21_a804/msg/file/2026-07/2026年健康生活.xlsx', data_only=True)
ws = wb['详细记录表']

data = []
for row in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True):
    d = row[0]
    if d is None: continue
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
        'shift': row[3],  # D列班次
        'weight': num(row[4]),
        'sleep': num(row[5]),
        'run': num(row[6]),
        'alcohol': num(row[7]),
        'calories': num(row[8]),
        'cal_diff': num(row[9]),
    })

print(f"总记录: {len(data)}天, {data[0]['date_str']} ~ {data[-1]['date_str']}")

# ============================================================
# 1. 班次切换间隔分析
# ============================================================
print("\n=== 班次切换间隔分析 ===")
transitions = []
for i in range(1, len(data)):
    prev_shift = data[i-1]['shift']
    curr_shift = data[i]['shift']
    if prev_shift and curr_shift and prev_shift != curr_shift:
        gap_hours = (data[i]['date'] - data[i-1]['date']).total_seconds() / 3600
        transitions.append({
            'date': data[i]['date_str'],
            'from': prev_shift,
            'to': curr_shift,
            'gap_hours': gap_hours,
            'weight_change': (data[i]['weight'] - data[i-1]['weight']) if (data[i]['weight'] and data[i-1]['weight']) else None,
            'run': data[i]['run'],
            'alcohol': data[i]['alcohol'],
            'sleep': data[i]['sleep'],
            'cal_diff': data[i]['cal_diff'],
        })

# 按切换类型分组
from collections import defaultdict
trans_patterns = defaultdict(list)
for t in transitions:
    key = f"{t['from']}→{t['to']}"
    trans_patterns[key].append(t)

shift_names = {'早': '早班', '中': '中班', '夜': '夜班', '休': '休息', None: '未记录', '': '未记录'}

print("\n切换类型统计:")
for pattern, items in sorted(trans_patterns.items(), key=lambda x: -len(x[1])):
    gaps = [i['gap_hours'] for i in items]
    runs = [i['run'] for i in items if i['run'] is not None]
    run_days = [i for i in items if i['run'] is not None and i['run'] > 0]
    alcs = [i['alcohol'] for i in items if i['alcohol'] is not None]
    alc_days = [i for i in items if i['alcohol'] is not None and i['alcohol'] > 0]
    wcs = [i['weight_change'] for i in items if i['weight_change'] is not None]
    sleeps = [i['sleep'] for i in items if i['sleep'] is not None]
    diffs = [i['cal_diff'] for i in items if i['cal_diff'] is not None]

    avg_gap = sum(gaps)/len(gaps) if gaps else 0
    avg_wc = sum(wcs)/len(wcs) if wcs else 0
    avg_sleep = sum(sleeps)/len(sleeps) if sleeps else 0
    avg_diff = sum(diffs)/len(diffs) if diffs else 0
    run_pct = f"{len(run_days)}/{len(items)}"
    alc_pct = f"{len(alc_days)}/{len(items)}"
    avg_alc = sum(alcs)/len(alc_days) if alc_days else 0

    print(f"  {pattern:8s}: {len(items):3d}次 | 间隔{avg_gap:.0f}h | 次日体重{avg_wc:+.3f}kg | 睡眠{avg_sleep:.1f}h | 运动{run_pct} | 饮酒{alc_pct}({avg_alc:.0f}mL) | 热量差{avg_diff:+.0f}")

# ============================================================
# 2. 五一假期分析 (4/28 - 5/10)
# ============================================================
print("\n=== 五一假期分析 (4/28~5/10) ===")
may_day = [d for d in data if '2026-04-28' <= d['date_str'] <= '2026-05-10']
if may_day:
    first_w = may_day[0]['weight']
    last_w = may_day[-1]['weight']
    weights = [(d['date_str'], d['weight']) for d in may_day if d['weight'] is not None]
    print(f"  日期范围: {may_day[0]['date_str']} ~ {may_day[-1]['date_str']} ({len(may_day)}天)")
    if first_w and last_w:
        print(f"  假期前体重: {first_w}kg, 假期末体重: {last_w}kg, 变化: {last_w-first_w:+.1f}kg")

    # 找最高点和最低点
    valid_w = [(d['date_str'], d['weight']) for d in may_day if d['weight'] is not None]
    if valid_w:
        max_w = max(valid_w, key=lambda x: x[1])
        min_w = min(valid_w, key=lambda x: x[1])
        print(f"  最高: {max_w[1]}kg ({max_w[0]}), 最低: {min_w[1]}kg ({min_w[0]})")

    # 每日明细
    print("\n  每日明细:")
    for d in may_day:
        w = f"{d['weight']:.1f}" if d['weight'] else "-"
        s = f"{d['sleep']:.1f}" if d['sleep'] else "-"
        r = f"{d['run']:.0f}" if d['run'] else "-"
        a = f"{d['alcohol']:.0f}" if d['alcohol'] else "-"
        c = f"{d['calories']:.0f}" if d['calories'] else "-"
        diff = f"{d['cal_diff']:+.0f}" if d['cal_diff'] else "-"
        sh = d['shift'] or "-"
        print(f"    {d['date_str']} [{sh}] W={w} S={s} R={r} A={a} C={c} D={diff}")

    # 恢复期分析 - 假期后多久回到假期前体重
    if first_w:
        print(f"\n  恢复分析 (目标回到{first_w}kg):")
        recovered = False
        for d in data:
            if d['date_str'] > '2026-05-10' and d['weight'] is not None and d['weight'] <= first_w:
                days_to_recover = (d['date'] - may_day[0]['date']).days
                print(f"    {d['date_str']} 体重={d['weight']}kg, 距假期开始{days_to_recover}天恢复")
                recovered = True
                break
        if not recovered:
            print(f"    假期后未回到{first_w}kg以下")

# ============================================================
# 3. 春节平台期分析 (1/25~2/15)
# ============================================================
print("\n=== 春节期间分析 (1/20~2/20) ===")
cny = [d for d in data if '2026-01-20' <= d['date_str'] <= '2026-02-20']
if cny:
    weights_cny = [(d['date_str'], d['weight']) for d in cny if d['weight'] is not None]
    if weights_cny:
        print(f"  日期范围: {cny[0]['date_str']} ~ {cny[-1]['date_str']} ({len(cny)}天)")
        print(f"  起始: {weights_cny[0][1]}kg, 结束: {weights_cny[-1][1]}kg, 变化: {weights_cny[-1][1]-weights_cny[0][1]:+.1f}kg")

        # 找平台期 - 连续N天体重变化<0.3kg
        print("\n  体重序列:")
        for ds, w in weights_cny:
            print(f"    {ds}: {w}kg")

        # 检测平台期
        plateau_start = None
        plateau_days = 0
        max_plateau = 0
        max_plateau_start = None
        max_plateau_end = None
        for i in range(1, len(weights_cny)):
            change = abs(weights_cny[i][1] - weights_cny[i-1][1])
            if change < 0.3:
                if plateau_start is None:
                    plateau_start = weights_cny[i-1][0]
                plateau_days += 1
            else:
                if plateau_days > max_plateau:
                    max_plateau = plateau_days
                    max_plateau_start = plateau_start
                    max_plateau_end = weights_cny[i-1][0]
                plateau_start = None
                plateau_days = 0
        if plateau_days > max_plateau:
            max_plateau = plateau_days
            max_plateau_start = plateau_start
            max_plateau_end = weights_cny[-1][0]

        if max_plateau > 3:
            print(f"\n  最长平台期: {max_plateau}天 ({max_plateau_start} ~ {max_plateau_end})")

# ============================================================
# 4. 全局平台期检测 (连续10天体重波动<0.5kg)
# ============================================================
print("\n=== 全局平台期检测 (连续7天净变化<0.3kg) ===")
valid_weights = [(d['date'], d['date_str'], d['weight']) for d in data if d['weight'] is not None]
plateaus = []
i = 0
while i < len(valid_weights):
    j = i + 6  # 7天窗口
    if j >= len(valid_weights):
        break
    window = valid_weights[i:j+1]
    w_change = window[-1][2] - window[0][2]
    if abs(w_change) < 0.3:
        # 向后扩展
        while j + 1 < len(valid_weights):
            next_change = valid_weights[j+1][2] - window[0][2]
            if abs(next_change) < 0.5:
                j += 1
                window = valid_weights[i:j+1]
            else:
                break
        total_change = window[-1][2] - window[0][2]
        plateaus.append({
            'start': window[0][1],
            'end': window[-1][1],
            'days': j - i + 1,
            'start_w': window[0][2],
            'end_w': window[-1][2],
            'change': round(total_change, 2),
        })
        i = j + 1
    else:
        i += 1

for p in plateaus:
    print(f"  {p['start']} ~ {p['end']} ({p['days']}天) | {p['start_w']}→{p['end_w']}kg ({p['change']:+.2f}kg)")

# ============================================================
# 5. 班次切换日的饮酒和运动模式
# ============================================================
print("\n=== 班次切换日详细分析 ===")
# 用户说的：夜→中只隔8小时，下了中班想喝酒，早→夜隔32小时也喝
key_transitions = ['夜→中', '中→早', '早→夜', '夜→休', '中→休', '早→休', '休→早', '休→中', '休→夜']

for kt in key_transitions:
    items = trans_patterns.get(kt, [])
    if not items:
        continue
    run_days = [i for i in items if i['run'] is not None and i['run'] > 0]
    alc_days = [i for i in items if i['alcohol'] is not None and i['alcohol'] > 0]
    wcs = [i['weight_change'] for i in items if i['weight_change'] is not None]
    sleeps = [i['sleep'] for i in items if i['sleep'] is not None]
    diffs = [i['cal_diff'] for i in items if i['cal_diff'] is not None]
    alcs = [i['alcohol'] for i in alc_days]

    avg_gap = sum(i['gap_hours'] for i in items) / len(items)
    avg_wc = sum(wcs)/len(wcs) if wcs else 0
    avg_sleep = sum(sleeps)/len(sleeps) if sleeps else 0
    avg_diff = sum(diffs)/len(diffs) if diffs else 0
    avg_alc = sum(alcs)/len(alcs) if alcs else 0

    print(f"\n  {kt} ({len(items)}次, 平均间隔{avg_gap:.0f}h):")
    print(f"    次日体重变化: {avg_wc:+.3f}kg")
    print(f"    睡眠: {avg_sleep:.1f}h")
    print(f"    运动率: {len(run_days)}/{len(items)} ({100*len(run_days)/len(items):.0f}%)")
    print(f"    饮酒率: {len(alc_days)}/{len(items)} ({100*len(alc_days)/len(items):.0f}%), 饮酒日均值{avg_alc:.0f}mL")
    print(f"    热量差: {avg_diff:+.0f}卡")

    # 显示前3个实例
    for item in items[:3]:
        w = f"{item['weight_change']:+.2f}" if item['weight_change'] is not None else "-"
        r = f"{item['run']:.0f}" if item['run'] else "0"
        a = f"{item['alcohol']:.0f}" if item['alcohol'] else "0"
        s = f"{item['sleep']:.1f}" if item['sleep'] else "-"
        print(f"      {item['date']} 间隔{item['gap_hours']:.0f}h 体重{w}kg 睡{s}h 跑{r}min 酒{a}mL")

# ============================================================
# 6. 饮酒与班次切换的关系
# ============================================================
print("\n=== 饮酒场景分析 ===")
# 哪些班次日饮酒最多？
drink_by_shift = defaultdict(lambda: {'days': 0, 'drink_days': 0, 'amounts': []})
for d in data:
    if d['shift']:
        drink_by_shift[d['shift']]['days'] += 1
        if d['alcohol'] and d['alcohol'] > 0:
            drink_by_shift[d['shift']]['drink_days'] += 1
            drink_by_shift[d['shift']]['amounts'].append(d['alcohol'])

print("\n各班次饮酒率:")
for shift in ['早', '中', '夜', '休']:
    info = drink_by_shift[shift]
    if info['days'] > 0:
        pct = 100 * info['drink_days'] / info['days']
        avg = sum(info['amounts'])/len(info['amounts']) if info['amounts'] else 0
        print(f"  {shift_names[shift]}: {info['drink_days']}/{info['days']} ({pct:.0f}%), 饮酒日均值{avg:.0f}mL")

# 切换日 vs 非切换日饮酒率
switch_dates = set(t['date'] for t in transitions)
switch_drink = sum(1 for d in data if d['date_str'] in switch_dates and d['alcohol'] and d['alcohol'] > 0)
switch_total = sum(1 for d in data if d['date_str'] in switch_dates)
non_switch_drink = sum(1 for d in data if d['date_str'] not in switch_dates and d['alcohol'] and d['alcohol'] > 0)
non_switch_total = sum(1 for d in data if d['date_str'] not in switch_dates)

print(f"\n切换日饮酒率: {switch_drink}/{switch_total} ({100*switch_drink/max(1,switch_total):.0f}%)")
print(f"非切换日饮酒率: {non_switch_drink}/{non_switch_total} ({100*non_switch_drink/max(1,non_switch_total):.0f}%)")

# ============================================================
# 7. 五一假期恢复曲线
# ============================================================
print("\n=== 五一假期体重恢复曲线 ===")
# 找五一前最低体重
pre_may = [d for d in data if '2026-04-20' <= d['date_str'] <= '2026-04-27' and d['weight'] is not None]
if pre_may:
    pre_w = pre_may[-1]['weight']
    print(f"  假期前体重(4/27): {pre_w}kg")

    # 找假期中最高点
    may_period = [d for d in data if '2026-04-28' <= d['date_str'] <= '2026-05-15' and d['weight'] is not None]
    if may_period:
        peak = max(may_period, key=lambda x: x['weight'])
        print(f"  假期峰值: {peak['weight']}kg ({peak['date_str']}), 涨了{peak['weight']-pre_w:+.1f}kg")

        # 恢复到pre_w的天数
        for d in data:
            if d['date_str'] > peak['date_str'] and d['weight'] is not None and d['weight'] <= pre_w:
                days = (d['date'] - peak['date']).days
                print(f"  恢复到{pre_w}kg: {d['date_str']} (用时{days}天)")
                break

        # 假期后30天趋势
        print("\n  假期后30天体重:")
        post_may = [d for d in data if d['date_str'] > peak['date_str'] and d['weight'] is not None][:30]
        for d in post_may[::3]:  # 每3天显示一次
            print(f"    {d['date_str']}: {d['weight']}kg")

# ============================================================
# 8. 导出图表数据
# ============================================================
print("\n===CHART_DATA===")

# 五一假期前后体重曲线
holiday_data = []
pre_hol = [d for d in data if '2026-04-20' <= d['date_str'] <= '2026-05-25' and d['weight'] is not None]
for d in pre_hol:
    holiday_data.append([d['date_str'], d['weight']])

# 春节期间体重曲线
cny_data = []
cny_period = [d for d in data if '2026-01-15' <= d['date_str'] <= '2026-02-28' and d['weight'] is not None]
for d in cny_period:
    cny_data.append([d['date_str'], d['weight']])

# 班次切换类型汇总
transition_summary = {}
for kt in key_transitions:
    items = trans_patterns.get(kt, [])
    if not items: continue
    run_days = [i for i in items if i['run'] is not None and i['run'] > 0]
    alc_days = [i for i in items if i['alcohol'] is not None and i['alcohol'] > 0]
    wcs = [i['weight_change'] for i in items if i['weight_change'] is not None]
    sleeps = [i['sleep'] for i in items if i['sleep'] is not None]
    diffs = [i['cal_diff'] for i in items if i['cal_diff'] is not None]
    alcs = [i['alcohol'] for i in alc_days]

    transition_summary[kt] = {
        'count': len(items),
        'avg_gap': round(sum(i['gap_hours'] for i in items) / len(items)),
        'avg_weight_change': round(sum(wcs)/len(wcs), 3) if wcs else 0,
        'avg_sleep': round(sum(sleeps)/len(sleeps), 1) if sleeps else 0,
        'run_rate': round(100 * len(run_days) / len(items)),
        'drink_rate': round(100 * len(alc_days) / len(items)),
        'avg_alcohol': round(sum(alcs)/len(alcs)) if alcs else 0,
        'avg_cal_diff': round(sum(diffs)/len(diffs)) if diffs else 0,
    }

# 各班次完整指标（修复早班不显示问题）
shift_metrics = {}
for shift in ['早', '中', '夜', '休']:
    rows = [d for d in data if d['shift'] == shift]
    if not rows: continue
    weights = [r['weight'] for r in rows if r['weight'] is not None]
    sleeps = [r['sleep'] for r in rows if r['sleep'] is not None]
    runs = [r['run'] for r in rows if r['run'] is not None]
    run_days = [r for r in rows if r['run'] is not None and r['run'] > 0]
    alcs = [r['alcohol'] for r in rows if r['alcohol'] is not None]
    alc_days = [r for r in rows if r['alcohol'] is not None and r['alcohol'] > 0]
    diffs = [r['cal_diff'] for r in rows if r['cal_diff'] is not None]
    cals = [r['calories'] for r in rows if r['calories'] is not None]

    # 次日体重变化
    next_changes = []
    for r in rows:
        # 找下一天
        idx = data.index(r)
        if idx + 1 < len(data) and data[idx+1]['weight'] and r['weight']:
            next_changes.append(data[idx+1]['weight'] - r['weight'])

    shift_metrics[shift_names[shift]] = {
        'days': len(rows),
        'avg_weight': round(sum(weights)/len(weights), 1) if weights else 0,
        'avg_sleep': round(sum(sleeps)/len(sleeps), 1) if sleeps else 0,
        'run_rate': round(100 * len(run_days) / len(rows)),
        'avg_run': round(sum(r['run'] for r in run_days) / len(run_days)) if run_days else 0,
        'drink_rate': round(100 * len(alc_days) / len(rows)),
        'avg_alcohol': round(sum(r['alcohol'] for r in alc_days) / len(alc_days)) if alc_days else 0,
        'avg_cal': round(sum(cals)/len(cals)) if cals else 0,
        'avg_cal_diff': round(sum(diffs)/len(diffs)) if diffs else 0,
        'next_weight_change': round(sum(next_changes)/len(next_changes), 3) if next_changes else 0,
    }

chart = {
    'holiday_weight': holiday_data,
    'cny_weight': cny_data,
    'transitions': transition_summary,
    'shift_metrics': shift_metrics,
    'plateaus': plateaus,
}
print(json.dumps(chart, ensure_ascii=False))
