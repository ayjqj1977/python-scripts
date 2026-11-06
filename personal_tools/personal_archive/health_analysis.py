import openpyxl.worksheet._reader as reader
orig_cast = reader._cast_number
def safe_cast(value):
    try:
        return orig_cast(value)
    except (ValueError, TypeError):
        return value
reader._cast_number = safe_cast

import openpyxl
from datetime import datetime, date
from collections import defaultdict

wb = openpyxl.load_workbook('D:/微信资料/xwechat_files/wxid_1saara878wtd21_a804/msg/file/2026-07/2026年健康生活.xlsx', data_only=True)
ws = wb['详细记录表']

rows = []
for row in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True):
    date_val = row[0]
    weight = row[4]
    sleep = row[5]
    run = row[6]
    alcohol = row[7]
    calories = row[8]
    cal_diff = row[9]
    study = row[13]
    study_content = row[14]
    study_result = row[15]
    if date_val is None:
        continue
    rows.append({
        'date': date_val, 'weight': weight, 'sleep': sleep, 'run': run,
        'alcohol': alcohol, 'calories': calories, 'cal_diff': cal_diff,
        'study': study, 'study_content': study_content, 'study_result': study_result
    })

# Today's data
today_str = '2026-07-04'
today_data = None
for r in rows:
    d = r['date']
    if isinstance(d, str) and d == today_str:
        today_data = r
    elif isinstance(d, (datetime, date)) and d.strftime('%Y-%m-%d') == today_str:
        today_data = r

# Monthly stats
monthly = defaultdict(lambda: {'days':0, 'weights':[], 'sleeps':[], 'runs':[], 'alcohols':[], 'cals':[], 'cal_diffs':[], 'studies':[]})
for r in rows:
    d = r['date']
    if isinstance(d, str):
        dt = datetime.strptime(d, '%Y-%m-%d')
    elif isinstance(d, (datetime, date)):
        dt = d if isinstance(d, datetime) else datetime.combine(d, datetime.min.time())
    month = dt.month
    monthly[month]['days'] += 1
    for key, target in [('weight','weights'),('sleep','sleeps'),('run','runs'),('alcohol','alcohols'),('calories','cals'),('cal_diff','cal_diffs'),('study','studies')]:
        if isinstance(r[key], (int, float)):
            monthly[month][target].append(r[key])

print('=== 数据范围 ===')
print(f'第一条: {rows[0]["date"]}, 最后一条: {rows[-1]["date"]}, 总天数: {len(rows)}')

print('\n=== 今日(7月4日) ===')
if today_data:
    for k in ['weight','sleep','run','alcohol','calories','cal_diff','study']:
        print(f'  {k}: {today_data[k]}')
else:
    print('  今日暂无数据')

print('\n=== 月度统计 ===')
month_names = ['一月','二月','三月','四月','五月','六月','七月','八月','九月','十月','十一月','十二月']
for m in range(1, 13):
    d = monthly[m]
    if d['days'] == 0:
        continue
    avg_w = sum(d['weights'])/len(d['weights']) if d['weights'] else None
    avg_s = sum(d['sleeps'])/len(d['sleeps']) if d['sleeps'] else None
    total_run = sum(d['runs']) if d['runs'] else 0
    total_alc = sum(d['alcohols']) if d['alcohols'] else 0
    total_study = sum(d['studies']) if d['studies'] else 0
    avg_cal = sum(d['cals'])/len(d['cals']) if d['cals'] else None
    total_cal_diff = sum(d['cal_diffs']) if d['cal_diffs'] else 0

    w_str = f'{avg_w:.1f}' if avg_w else '-'
    s_str = f'{avg_s:.1f}' if avg_s else '-'
    cal_str = f'{avg_cal:.0f}' if avg_cal else '-'
    diff_str = f'{total_cal_diff:+.0f}' if d['cal_diffs'] else '-'

    print(f'{month_names[m-1]}: {d["days"]}天 | 体重{w_str}kg | 睡眠{s_str}h | 跑步{total_run:.0f}分 | 白酒{total_alc:.0f}mL | 热量{cal_str}卡 | 差{diff_str}卡 | 学习{total_study:.1f}h')

# Weight trend
print('\n=== 体重变化 ===')
vw = [(r['date'], r['weight']) for r in rows if isinstance(r['weight'], (int, float))]
if vw:
    first_w = vw[0][1]
    last_w = vw[-1][1]
    min_w = min(w for _, w in vw)
    max_w = max(w for _, w in vw)
    min_date = [d for d, w in vw if w == min_w][0]
    print(f'起始: {first_w}kg -> 最新: {last_w}kg -> 变化: {last_w-first_w:+.1f}kg')
    print(f'最低: {min_w}kg ({min_date}) | 最高: {max_w}kg')

# Study
print('\n=== 学习统计 ===')
vs = [(r['date'], r['study']) for r in rows if isinstance(r['study'], (int, float)) and r['study'] > 0]
ts = sum(s for _, s in vs)
print(f'学习天数: {len(vs)} | 累计: {ts:.1f}h | 日均: {ts/len(vs):.1f}h')
mx = max(vs, key=lambda x: x[1])
print(f'单日最长: {mx[1]}h ({mx[0]})')

# Study content
print('\n=== 学习内容汇总 ===')
for r in rows:
    if r['study_content']:
        print(f'  {r["date"]}: {r["study_content"].strip()} -> {r["study_result"] or ""}')

# Running
print('\n=== 运动统计 ===')
vr = [(r['date'], r['run']) for r in rows if isinstance(r['run'], (int, float)) and r['run'] > 0]
tr = sum(r for _, r in vr)
print(f'运动天数: {len(vr)} | 累计: {tr:.0f}分钟({tr/60:.1f}h) | 日均: {tr/len(vr):.0f}分钟')
mx = max(vr, key=lambda x: x[1])
print(f'单日最长: {mx[1]}分钟 ({mx[0]})')

# Sleep
print('\n=== 睡眠统计 ===')
vsl = [(r['date'], r['sleep']) for r in rows if isinstance(r['sleep'], (int, float))]
asl = sum(s for _, s in vsl) / len(vsl) if vsl else 0
mn = min(vsl, key=lambda x: x[1])
mx = max(vsl, key=lambda x: x[1])
print(f'记录天数: {len(vsl)} | 平均: {asl:.1f}h | 最短: {mn[1]}h({mn[0]}) | 最长: {mx[1]}h({mx[0]})')

# Alcohol
print('\n=== 饮酒统计 ===')
va = [(r['date'], r['alcohol']) for r in rows if isinstance(r['alcohol'], (int, float)) and r['alcohol'] > 0]
ta = sum(a for _, a in va)
print(f'饮酒天数: {len(va)} | 累计: {ta:.0f}mL | 饮酒日均值: {ta/len(va):.0f}mL' if va else '无饮酒记录')

# Last 7 days
print('\n=== 最近7天 ===')
for r in rows[-7:]:
    w = r['weight'] if isinstance(r['weight'], (int, float)) else '-'
    s = r['sleep'] if isinstance(r['sleep'], (int, float)) else '-'
    run = r['run'] if isinstance(r['run'], (int, float)) else '-'
    alc = r['alcohol'] if isinstance(r['alcohol'], (int, float)) else '-'
    cal = r['calories'] if isinstance(r['calories'], (int, float)) else '-'
    diff = r['cal_diff'] if isinstance(r['cal_diff'], (int, float)) else '-'
    stu = r['study'] if isinstance(r['study'], (int, float)) else '-'
    print(f'{r["date"]}: 体重{w} 睡眠{s} 跑步{run} 白酒{alc} 热量{cal} 差{diff} 学习{stu}')
