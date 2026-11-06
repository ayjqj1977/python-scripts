# -*- coding: utf-8 -*-
import openpyxl.worksheet._reader as reader
orig_cast = reader._cast_number
def safe_cast(value):
    try: return orig_cast(value)
    except: return value
reader._cast_number = safe_cast

import openpyxl, json
from datetime import datetime, date

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
        'weight': num(row[4]),
        'sleep': num(row[5]),
        'run': num(row[6]),
        'alcohol': num(row[7]),
        'calories': num(row[8]),
        'cal_diff': num(row[9]),
    })

print(f'总记录天数: {len(data)}')
print(f'日期范围: {data[0]["date_str"]} ~ {data[-1]["date_str"]}')

# Daily weight changes
weight_changes = []
for i in range(1, len(data)):
    if data[i]['weight'] is not None and data[i-1]['weight'] is not None:
        d = data[i]
        prev = data[i-1]
        wc = round(d['weight'] - prev['weight'], 2)
        weight_changes.append({
            'date': d['date_str'],
            'weight_change': wc,
            'prev_sleep': prev['sleep'],
            'prev_run': prev['run'],
            'prev_alcohol': prev['alcohol'],
            'prev_cal': prev['calories'],
            'prev_cal_diff': prev['cal_diff'],
            'today_sleep': d['sleep'],
            'today_run': d['run'],
            'today_alcohol': d['alcohol'],
            'today_cal': d['calories'],
            'today_cal_diff': d['cal_diff'],
        })

print(f'\n有效体重变化记录: {len(weight_changes)} 天')

def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 5: return None, 0
    xs2 = [p[0] for p in pairs]
    ys2 = [p[1] for p in pairs]
    n = len(pairs)
    mx, my = sum(xs2)/n, sum(ys2)/n
    cov = sum((x-mx)*(y-my) for x,y in pairs)
    sx = (sum((x-mx)**2 for x in xs2))**0.5
    sy = (sum((y-my)**2 for y in ys2))**0.5
    if sx == 0 or sy == 0: return None, 0
    return cov/(sx*sy), n

print('\n=== 各因素与「次日体重变化」的Pearson相关系数 ===')
factors = [
    ('前一天睡眠', [w['prev_sleep'] for w in weight_changes]),
    ('前一天跑步', [w['prev_run'] for w in weight_changes]),
    ('前一天白酒', [w['prev_alcohol'] for w in weight_changes]),
    ('前一天热量', [w['prev_cal'] for w in weight_changes]),
    ('前一天热量差', [w['prev_cal_diff'] for w in weight_changes]),
    ('当天睡眠', [w['today_sleep'] for w in weight_changes]),
    ('当天跑步', [w['today_run'] for w in weight_changes]),
    ('当天白酒', [w['today_alcohol'] for w in weight_changes]),
    ('当天热量', [w['today_cal'] for w in weight_changes]),
    ('当天热量差', [w['today_cal_diff'] for w in weight_changes]),
]
target = [w['weight_change'] for w in weight_changes]
for name, vals in factors:
    r, n = pearson(vals, target)
    if r is not None:
        sig = '***' if abs(r) > 0.3 else ('**' if abs(r) > 0.2 else ('*' if abs(r) > 0.1 else ''))
        print(f'  {name:12s}: r={r:+.4f} (n={n}) {sig}')
    else:
        print(f'  {name:12s}: 数据不足')

print('\n=== 按星期分析 ===')
weekday_names = ['周一','周二','周三','周四','周五','周六','周日']
for wd in range(7):
    wd_data = [d for d in data if d['weekday'] == wd]
    if not wd_data: continue
    weights = [d['weight'] for d in wd_data if d['weight'] is not None]
    sleeps = [d['sleep'] for d in wd_data if d['sleep'] is not None]
    runs = [d['run'] for d in wd_data if d['run'] is not None and d['run'] > 0]
    alcs = [d['alcohol'] for d in wd_data if d['alcohol'] is not None and d['alcohol'] > 0]
    cals = [d['calories'] for d in wd_data if d['calories'] is not None]
    diffs = [d['cal_diff'] for d in wd_data if d['cal_diff'] is not None]
    
    avg_w = f'{sum(weights)/len(weights):.1f}' if weights else '-'
    avg_s = f'{sum(sleeps)/len(sleeps):.1f}' if sleeps else '-'
    avg_run = f'{sum(runs)/len(runs):.0f}' if runs else '-'
    run_pct = f'{len(runs)}/{len(wd_data)}'
    alc_pct = f'{len(alcs)}/{len(wd_data)}'
    avg_cal = f'{sum(cals)/len(cals):.0f}' if cals else '-'
    avg_diff = f'{sum(diffs)/len(diffs):+.0f}' if diffs else '-'
    
    print(f'  {weekday_names[wd]}: {len(wd_data):2d}天 | 均重{avg_w}kg | 睡{avg_s}h | 跑{run_pct}天 均{avg_run}min | 饮酒{alc_pct}天 | 均{avg_cal}卡 差{avg_diff}')

print('\n=== 饮酒对次日体重的影响 ===')
drink_yes = [w for w in weight_changes if (w['prev_alcohol'] or 0) > 0]
drink_no = [w for w in weight_changes if (w['prev_alcohol'] or 0) == 0]

if drink_yes:
    avg_wc_drink = sum(w['weight_change'] for w in drink_yes) / len(drink_yes)
    print(f'  饮酒次日: n={len(drink_yes)}天, 平均体重变化={avg_wc_drink:+.3f}kg')
if drink_no:
    avg_wc_nodrink = sum(w['weight_change'] for w in drink_no) / len(drink_no)
    print(f'  不饮酒次日: n={len(drink_no)}天, 平均体重变化={avg_wc_nodrink:+.3f}kg')
if drink_yes and drink_no:
    print(f'  差异: {avg_wc_drink - avg_wc_nodrink:+.3f}kg/天')

print('\n  --- 饮酒量分级 ---')
for label, lo, hi in [('不饮', 0, 0), ('少量(1-150)', 1, 150), ('中量(151-300)', 151, 300), ('大量(300+)', 301, 99999)]:
    if lo == 0:
        subset = [w for w in weight_changes if (w['prev_alcohol'] or 0) == 0]
    else:
        subset = [w for w in weight_changes if lo <= (w['prev_alcohol'] or 0) <= hi]
    if subset:
        avg = sum(w['weight_change'] for w in subset) / len(subset)
        print(f'  {label:15s}: n={len(subset):3d}, 次日均变化={avg:+.3f}kg')

print('\n=== 跑步量对次日体重的影响 ===')
for label, lo, hi in [('不跑', 0, 0), ('轻度(1-30min)', 1, 30), ('中度(31-60min)', 31, 60), ('重度(>60min)', 61, 99999)]:
    if lo == 0:
        subset = [w for w in weight_changes if (w['prev_run'] or 0) == 0]
    else:
        subset = [w for w in weight_changes if lo <= (w['prev_run'] or 0) <= hi]
    if subset:
        avg = sum(w['weight_change'] for w in subset) / len(subset)
        print(f'  {label:18s}: n={len(subset):3d}, 次日均变化={avg:+.3f}kg')

print('\n=== 睡眠时长对次日体重的影响 ===')
for label, lo, hi in [('<5h', 0, 5), ('5-6h', 5, 6), ('6-7h', 6, 7), ('7-8h', 7, 8), ('8h+', 8, 99)]:
    subset = [w for w in weight_changes if w['prev_sleep'] is not None and lo <= w['prev_sleep'] < hi]
    if subset:
        avg = sum(w['weight_change'] for w in subset) / len(subset)
        print(f'  {label:8s}: n={len(subset):3d}, 次日均变化={avg:+.3f}kg')

print('\n=== 热量差对次日体重的影响 ===')
for label, lo, hi in [('极低(<-800)', -99999, -800), ('低(-800~-500)', -800, -500), ('中(-500~-200)', -500, -200), ('小(-200~0)', -200, 0), ('盈余(>0)', 0, 99999)]:
    subset = [w for w in weight_changes if w['prev_cal_diff'] is not None and lo <= w['prev_cal_diff'] < hi]
    if subset:
        avg = sum(w['weight_change'] for w in subset) / len(subset)
        print(f'  {label:18s}: n={len(subset):3d}, 次日均变化={avg:+.3f}kg')

print('\n=== 周度减重速度 ===')
weekly = []
for i in range(0, len(data), 7):
    week = data[i:i+7]
    weights = [d['weight'] for d in week if d['weight'] is not None]
    if len(weights) >= 2:
        w_change = weights[-1] - weights[0]
        week_start = week[0]['date_str']
        week_end = week[-1]['date_str']
        runs = [d['run'] for d in week if d['run'] is not None and d['run'] > 0]
        alcs = [d['alcohol'] for d in week if d['alcohol'] is not None and d['alcohol'] > 0]
        sleeps = [d['sleep'] for d in week if d['sleep'] is not None]
        diffs = [d['cal_diff'] for d in week if d['cal_diff'] is not None]
        total_run = sum(runs)
        total_alc = sum(alcs)
        avg_sleep = sum(sleeps)/len(sleeps) if sleeps else 0
        total_diff = sum(diffs) if diffs else 0
        weekly.append({
            'start': week_start, 'end': week_end,
            'change': round(w_change, 2),
            'run': round(total_run), 'alc': round(total_alc),
            'sleep': round(avg_sleep, 1), 'cal_diff': round(total_diff)
        })
        print(f'  {week_start}~{week_end}: {w_change:+.2f}kg | 跑步{total_run:.0f}min 饮酒{total_alc:.0f}mL 睡眠{avg_sleep:.1f}h 热量差{total_diff:+.0f}')

print('\n=== 平台期分析(连续7天体重变化<0.3kg) ===')
in_plateau = False
plateau_start = None
for i in range(len(data)):
    if i >= 7 and data[i]['weight'] is not None:
        week_ago = None
        for j in range(i-1, max(i-8, -1), -1):
            if data[j]['weight'] is not None:
                week_ago = data[j]['weight']
                break
        if week_ago is not None:
            recent_change = abs(data[i]['weight'] - week_ago)
            if recent_change < 0.3:
                if not in_plateau:
                    plateau_start = data[i]['date_str']
                    in_plateau = True
            else:
                if in_plateau:
                    print(f'  平台期: {plateau_start} ~ {data[i-1]["date_str"]}')
                    in_plateau = False
if in_plateau:
    print(f'  平台期: {plateau_start} ~ {data[-1]["date_str"]} (持续中)')

print('\n=== 体重突破日(单日下降>0.8kg) ===')
breakthroughs = [w for w in weight_changes if w['weight_change'] <= -0.8]
print(f'  共{len(breakthroughs)}天')
if breakthroughs:
    b_sleep = [w['prev_sleep'] for w in breakthroughs if w['prev_sleep']]
    b_run = [w['prev_run'] for w in breakthroughs if w['prev_run']]
    b_diff = [w['prev_cal_diff'] for w in breakthroughs if w['prev_cal_diff']]
    alc_days = len([w for w in breakthroughs if (w['prev_alcohol'] or 0) > 0])
    print(f'  前一天平均睡眠: {sum(b_sleep)/len(b_sleep):.1f}h' if b_sleep else '  前一天睡眠: N/A')
    print(f'  前一天平均跑步: {sum(b_run)/len(b_run):.0f}min' if b_run else '  前一天跑步: N/A')
    print(f'  前一天平均热量差: {sum(b_diff)/len(b_diff):+.0f}卡' if b_diff else '  前一天热量差: N/A')
    print(f'  前一天饮酒天数: {alc_days}/{len(breakthroughs)}')

print('\n=== 体重反弹日(单日上升>0.5kg) ===')
rebounds = [w for w in weight_changes if w['weight_change'] >= 0.5]
print(f'  共{len(rebounds)}天')
if rebounds:
    r_sleep = [w['prev_sleep'] for w in rebounds if w['prev_sleep']]
    r_run = [w['prev_run'] for w in rebounds if w['prev_run']]
    r_diff = [w['prev_cal_diff'] for w in rebounds if w['prev_cal_diff']]
    r_alc = [w['prev_alcohol'] for w in rebounds if w['prev_alcohol']]
    alc_days = len([w for w in rebounds if (w['prev_alcohol'] or 0) > 0])
    print(f'  前一天平均睡眠: {sum(r_sleep)/len(r_sleep):.1f}h' if r_sleep else '  前一天睡眠: N/A')
    print(f'  前一天平均跑步: {sum(r_run)/len(r_run):.0f}min' if r_run else '  前一天跑步: N/A')
    print(f'  前一天平均热量差: {sum(r_diff)/len(r_diff):+.0f}卡' if r_diff else '  前一天热量差: N/A')
    print(f'  前一天饮酒天数: {alc_days}/{len(rebounds)}')
    print(f'  前一天饮酒日均量: {sum(r_alc)/len(r_alc):.0f}mL' if r_alc else '')

# === 组合效应分析 ===
print('\n=== 组合效应分析 ===')
# 跑了+不喝酒 vs 没跑+喝酒 vs 跑了+喝酒
combo_a = [w for w in weight_changes if (w['prev_run'] or 0) > 0 and (w['prev_alcohol'] or 0) == 0]
combo_b = [w for w in weight_changes if (w['prev_run'] or 0) == 0 and (w['prev_alcohol'] or 0) == 0]
combo_c = [w for w in weight_changes if (w['prev_run'] or 0) > 0 and (w['prev_alcohol'] or 0) > 0]
combo_d = [w for w in weight_changes if (w['prev_run'] or 0) == 0 and (w['prev_alcohol'] or 0) > 0]
for label, subset in [('跑了+不喝酒', combo_a), ('没跑+不喝酒', combo_b), ('跑了+喝酒', combo_c), ('没跑+喝酒', combo_d)]:
    if subset:
        avg = sum(w['weight_change'] for w in subset) / len(subset)
        print(f'  {label:12s}: n={len(subset):3d}, 次日均变化={avg:+.3f}kg')

# === 滑动窗口分析: 7天平均热量差 vs 7天体重变化 ===
print('\n=== 7天滑动窗口: 热量差总量 vs 体重变化 ===')
for i in range(7, len(data)):
    week = data[i-7:i]
    weights = [d['weight'] for d in week if d['weight'] is not None]
    diffs = [d['cal_diff'] for d in week if d['cal_diff'] is not None]
    if len(weights) >= 2 and len(diffs) >= 5:
        w_change = weights[-1] - weights[0]
        total_diff = sum(diffs)
        # only print notable ones
        if abs(w_change) > 1.5 or total_diff < -4000 or total_diff > 500:
            pass  # collect for chart instead

# Export chart data
print('\n===CHART_DATA===')
chart_data = {
    'weight': [[d['date_str'], d['weight']] for d in data if d['weight'] is not None],
    'sleep': [[d['date_str'], d['sleep']] for d in data if d['sleep'] is not None],
    'run': [[d['date_str'], d['run']] for d in data if d['run'] is not None and d['run'] > 0],
    'alcohol': [[d['date_str'], d['alcohol']] for d in data if d['alcohol'] is not None and d['alcohol'] > 0],
    'cal_diff': [[d['date_str'], d['cal_diff']] for d in data if d['cal_diff'] is not None],
    'weight_changes': [[w['date'], w['weight_change'], w['prev_sleep'] or 0, w['prev_run'] or 0, w['prev_alcohol'] or 0, w['prev_cal_diff'] or 0] for w in weight_changes],
    'weekly': weekly,
}
print(json.dumps(chart_data, ensure_ascii=False))
