from concurrent import futures
import time
import threading
import csv
import requests

# 全局变量定义
result = []          # 临时存储爬取的用户数据
lock = threading.Lock()  # 线程锁（保证多线程写入安全）
total = 1            # 用户数据自增ID
cookie = {
    'domain': '/',
    'expires': 'false',
    'httpOnly': 'false',
    'name': 'buvid3',
    'path': 'Fri, 29 Jan 2021 08:50:10 GMT',
    'value': '7A29BBDE-VA94D-4F66-QC63-D9CB8568D84331045infoc,bilibili.com'
}

# 模拟iPhone浏览器的User-Agent
uas = 'Mozilla/5.0 (iPhone; CPU iPhone OS 10_3 \
       like Mac OS X) AppleWebKit/602.1.50 (KHTML, like Gecko) CriOS/56.0.2924.75 \
       Mobile/14E5239e Safari/602.1'

# 初始化CSV文件并写入表头
with open('user_info11.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'mid', 'name', 'sex', 'following', 'fans', 'level'])


def run(url):
    """
    爬取单个用户空间信息的核心函数（线程执行单元）
    
    参数:
        url (str): 用户空间URL（格式：https://m.bilibili.com/space/{mid}）
    """
    global total, result, uas, cookie
    mid = url.replace('https://m.bilibili.com/space/', '')  # 从URL中提取用户mid
    head = {
        'User-Agent': uas,
        'X-Requested-With': 'XMLHttpRequest',
        'Origin': 'http://space.bilibili.com',
        'Host': 'm.bilibili.com',
        'AlexaToolbar-ALX_NS_PH': 'AlexaToolbar/alx-4.0',
        'Accept-Language': 'zh-CN,zh;q=0.8,en;q=0.6,ja;q=0.4',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Referer': url
    }
    time.sleep(0.5)     # 延迟0.5秒防止IP被封
    try:
        # 获取用户空间页面数据
        r = requests.get(url, headers=head, cookies=cookie, timeout=10).text
        if r.find("name\":") == -1:  # 无有效数据则跳过
            return
        
        # 使用正则表达式提取用户昵称和性别
        import re
        name_match = re.search(r'"name":"(.*?)"', r)
        name = name_match.group(1) if name_match else "未知"
        sex_match = re.search(r'"sex":"(.*?)"', r)
        sex = sex_match.group(1) if sex_match else "未知"

        # 根据页面内容判断用户等级（通过lv0~lv6的CSS类名）
        if r.find('lv0') != -1:
            level = 0
        elif r.find('lv1') != -1:
            level = 1
        elif r.find('lv2') != -1:
            level = 2
        elif r.find('lv3') != -1:
            level = 3
        elif r.find('lv4') != -1:
            level = 4
        elif r.find('lv5') != -1:
            level = 5
        elif r.find('lv6') != -1:
            level = 6
        else:
            level = -1  # 未知等级

        # 获取用户关注数和粉丝数（调用B站关系接口）
        head = {
            'User-Agent': uas,
            'X-Requested-With': 'XMLHttpRequest',
            'Origin': 'http://space.bilibili.com',
            'Host': 'api.bilibili.com',
            'AlexaToolbar-ALX_NS_PH': 'AlexaToolbar/alx-4.0',
            'Accept-Language': 'zh-CN,zh;q=0.8,en;q=0.6,ja;q=0.4',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Referer': url
        }
        res = requests.get('https://api.bilibili.com/x/relation/stat?jsonp=jsonp&vmid='+str(mid),
                           headers=head, cookies=cookie, timeout=10).text
        res_js = eval(res)  # 解析JSON数据
        following = res_js['data']['following']  # 关注数
        follower = res_js['data']['follower']    # 粉丝数

        # 构造用户数据元组
        users = (total, mid, name, sex, following, follower, level)
        print(f"爬取到用户信息：id={users[0]}, mid={users[1]}, 昵称={name}, 性别={sex}, 关注数={users[4]}, 粉丝数={users[5]}, 等级={users[6]}")
    except Exception as e:
        print('error')
        print(e)
        print(i)
        print(total)
        return
    with lock:
        result.append(users)  # 加锁保证多线程写入安全
        print(i)
        print(total)
        total += 1


def save():
    """
    将临时存储的用户数据写入CSV文件（线程安全）
    """
    global result, total
    with open('user_info11.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(result)  # 批量写入数据
    result = []  # 清空临时存储


if __name__ == "__main__":
    # 主程序入口
    total_num = 15000  # 目标爬取用户数（仅爬取15000条数据）
    num = 32*20        # 每批次处理的用户数（32*20=640）
    # 调整循环次数确保覆盖目标数量
    for i in range(12, int(total_num/num) + 12):
        begin = num * i  # 计算当前批次起始mid
        urls = ["https://m.bilibili.com/space/{}".format(j) for j in range(begin, begin + num)]  # 生成用户空间URL列表

        # 使用线程池并发处理（最大10个线程）
        with futures.ThreadPoolExecutor(10) as executor:
            executor.map(run, urls)

        save()  # 保存当前批次数据到CSV文件
