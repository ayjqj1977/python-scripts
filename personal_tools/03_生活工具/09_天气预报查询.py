import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import re

# 全局设置：解决中文乱码+图表样式
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

# 定义城市信息（城市名: 城市代码）
city_info = {
    "海口": "101310101",
    "安阳": "101180501"
}
# 定义城市样式映射（区分不同城市的线条和标记）
city_style = {
    "海口": {"high_color": "#FF6B9D", "low_color": "#4ECDC4", "high_marker": "o", "low_marker": "s"},
    "安阳": {"high_color": "#FF9F43", "low_color": "#0984e3", "high_marker": "^", "low_marker": "p"}
}

def crawl_weather(city_name, city_code):
    """爬取单个城市的7天天气数据"""
    url = f"http://www.weather.com.cn/weather/{city_code}.shtml"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        "Referer": "http://www.weather.com.cn/"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")
        weather_items = soup.select("div[class='c7d'] ul li")[:7]
        
        dates = []
        high_temps = []
        low_temps = []
        weathers = []
        
        for item in weather_items:
            # 提取日期
            date = item.select_one("h1").get_text().strip() if item.select_one("h1") else "未知"
            dates.append(date)
            # 提取温度
            temp_text = item.select_one("p[class='tem']").get_text().strip() if item.select_one("p[class='tem']") else "0℃/0℃"
            temp_nums = re.findall(r"\d+", temp_text)
            if len(temp_nums) >= 2:
                high_temp = int(temp_nums[0])
                low_temp = int(temp_nums[1])
            elif len(temp_nums) == 1:
                high_temp = int(temp_nums[0])
                low_temp = high_temp
            else:
                high_temp = 0
                low_temp = 0
            high_temps.append(high_temp)
            low_temps.append(low_temp)
            # 提取天气状况
            weather = item.select_one("p[class='wea']").get_text().strip() if item.select_one("p[class='wea']") else "未知"
            weathers.append(weather)
        
        return {
            "dates": dates,
            "high_temps": high_temps,
            "low_temps": low_temps,
            "weathers": weathers
        }
    except Exception as e:
        print(f"爬取{city_name}天气失败：{str(e)[:50]}")
        return None

# 爬取两座城市的天气数据
weather_data = {}
for city_name, city_code in city_info.items():
    print(f"开始爬取{city_name}天气数据...")
    data = crawl_weather(city_name, city_code)
    if data:
        weather_data[city_name] = data
        print(f"{city_name}天气数据爬取完成！")
    else:
        print(f"{city_name}天气数据爬取失败，跳过该城市")

# 绘制双城市天气对比图
if weather_data:
    fig, ax = plt.subplots()
    # 遍历城市绘制折线
    for city_name, data in weather_data.items():
        style = city_style[city_name]
        # 绘制最高温折线
        ax.plot(
            data["dates"], data["high_temps"],
            marker=style["high_marker"], color=style["high_color"], linewidth=2.5,
            markersize=9, label=f"{city_name}最高温"
        )
        # 绘制最低温折线
        ax.plot(
            data["dates"], data["low_temps"],
            marker=style["low_marker"], color=style["low_color"], linewidth=2.5,
            markersize=9, label=f"{city_name}最低温"
        )
        # 标注温度数值
        for i in range(len(data["dates"])):
            # 最高温数值
            ax.text(
                data["dates"][i], data["high_temps"][i] + 0.8,
                f"{data['high_temps'][i]}℃", ha="center", fontsize=10, color=style["high_color"]
            )
            # 最低温数值
            ax.text(
                data["dates"][i], data["low_temps"][i] - 1.2,
                f"{data['low_temps'][i]}℃", ha="center", fontsize=10, color=style["low_color"]
            )

    # 图表样式美化
    ax.set_title("海口&安阳一周温度趋势对比", fontsize=20, pad=25)
    ax.set_xlabel("日期", fontsize=16)
    ax.set_ylabel("温度（℃）", fontsize=16)
    ax.tick_params(axis="x", rotation=45, labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(loc="upper right", fontsize=14, ncol=2)
    # 自动调整布局
    plt.tight_layout()
    # 保存并显示图表
    plt.savefig("海口安阳天气对比图.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 打印详细天气信息
    print("\n===== 海口&安阳一周天气明细 =====")
    for city_name, data in weather_data.items():
        print(f"\n【{city_name}】")
        for i in range(len(data["dates"])):
            print(f"{data['dates'][i]}：{data['weathers'][i]}，最高{data['high_temps'][i]}℃，最低{data['low_temps'][i]}℃")
else:
    print("未爬取到任何城市的天气数据，无法生成对比图")