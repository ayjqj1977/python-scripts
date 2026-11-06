import requests
from bs4 import BeautifulSoup
import smtplib
from email.mime.text import MIMEText
import time

# ========== 1. 配置参数（需替换为自己的信息） ==========
# 京东商品完整URL（必须带https://）
GOODS_URL = "https://item.jd.com/100012345678.html"  # 替换为真实商品链接
# 价格阈值：低于该价格发送提醒
PRICE_THRESHOLD = 200
# 邮箱配置（QQ邮箱为例）
EMAIL_USER = "你的QQ邮箱@qq.com"  # 发送方邮箱
EMAIL_PWD = "你的QQ邮箱授权码"      # 邮箱SMTP授权码（不是登录密码）
RECIPIENT = "接收提醒的邮箱@xxx.com"  # 接收方邮箱

# ========== 2. 爬取京东商品价格 ==========
def get_jd_price(url):
    """爬取京东商品的当前价格"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        "Referer": "https://www.jd.com/"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")
        # 提取商品价格（适配京东商品页价格标签）
        price_tag = soup.select_one("span.price J_price") or soup.select_one("p.price")
        if price_tag:
            price_text = price_tag.get_text().strip().replace("¥", "").replace("元", "")
            return float(price_text) if price_text.replace(".", "").isdigit() else None
        else:
            # 备用：从京东的价格接口提取（更稳定）
            price_api = f"https://p.3.cn/prices/mgets?skuIds=J_{url.split('/')[-1].split('.')[0]}"
            api_response = requests.get(price_api, headers=headers)
            price_data = api_response.json()
            return float(price_data[0]["p"]) if price_data else None
    except Exception as e:
        print(f"价格爬取失败：{str(e)[:50]}")
        return None

# ========== 3. 发送邮件提醒 ==========
def send_price_email(current_price, threshold):
    """发送价格提醒邮件"""
    if not EMAIL_USER or not EMAIL_PWD:
        print("请配置邮箱账号和授权码")
        return
    # 邮件内容
    msg = MIMEText(f"【商品价格提醒】\n当前价格：{current_price}元\n低于阈值：{threshold}元，可下单！", "plain", "utf-8")
    msg["From"] = EMAIL_USER
    msg["To"] = RECIPIENT
    msg["Subject"] = "京东商品价格达标提醒"
    
    try:
        # 连接QQ邮箱SMTP服务器
        with smtplib.SMTP_SSL("smtp.qq.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PWD)
            smtp.send_message(msg)
        print("价格提醒邮件发送成功！")
    except Exception as e:
        print(f"邮件发送失败：{str(e)[:50]}")

# ========== 4. 定时监控主逻辑 ==========
if __name__ == "__main__":
    print(f"开始监控京东商品：{GOODS_URL}")
    print(f"价格阈值：{PRICE_THRESHOLD}元，每小时检测一次...\n")
    
    while True:
        current_price = get_jd_price(GOODS_URL)
        if current_price is not None:
            print(f"当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}，商品价格：{current_price}元")
            # 价格低于阈值则发送邮件并停止监控
            if current_price < PRICE_THRESHOLD:
                send_price_email(current_price, PRICE_THRESHOLD)
                print("价格达标，监控结束！")
                break
        else:
            print(f"当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}，价格爬取失败")
        
        # 每小时检测一次（3600秒）
        time.sleep(3600)