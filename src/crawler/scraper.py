import requests
import time
import pandas as pd
import re


def trans_data(timeStamp):
    """
    将Unix时间戳转换为可读的日期时间字符串
    
    参数:
        timeStamp (int): Unix时间戳（秒级）
    
    返回:
        str: 格式化后的日期时间字符串（格式：YYYY-MM-DD HH:MM:SS）
    """
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime


def search_bilibili(keyword, max_page, out_file):
    """
    爬取B站指定关键词的视频信息并保存到CSV文件
    
    参数:
        keyword (str): 搜索关键词
        max_page (int): 最大爬取页数
        out_file (str): 输出CSV文件路径
    """
    for page in range(1, max_page + 1):
        print('正在爬取第', page, '页')
        # 构建请求URL和查询参数
        url = 'https://api.bilibili.com/x/web-interface/search/type'
        headers = {
            # 浏览器标识头，模拟真实用户请求
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            # 登录Cookie（需更新为有效Cookie才能获取完整数据）
            'Cookie': "buvid3=83B979CF-3BC6-32E3-22D3-964861D3AC8321559infoc; b_nut=1708959821; i-wanna-go-back=-1; b_ut=7; _uuid=1351059F9-10271-5883-DC32-931DA5B3223126186infoc; enable_web_push=DISABLE; header_theme_version=CLOSE; DedeUserID=9552818; DedeUserID__ckMd5=b4c870f95007b415; hit-dyn-v2=1; CURRENT_FNVAL=4048; rpdid=|(J|)k)J)JuR0J'u~|m|JY|Jk; CURRENT_QUALITY=80; FEED_LIVE_VERSION=V8; buvid_fp_plain=undefined; buvid4=DC388901-AC12-FF04-8BC5-36EA28462B6D25168-024022615-DtgSrDL24kcpC%2Fs4auvC7Q%3D%3D; SESSDATA=1322a8ee%2C1725080477%2Ce3447%2A31CjAN73qBSUwbhhnLO6pi9sFp8yh6brppsAd_S7fuT-5XVYTt8X99N1lmbQmaqpQB46MSVkZQcmc1Z2pRUy02d0hNc2JTdm44NDNKQ1Q5MV9UUTZ1dHphSzU1aWt4d294TlUxZy1DNndCTzh1YUgyM19rR3BuOWplSkhacWFDeE1FeUhuOHM0dnNRIIEC; bili_jct=67e068afc842b0e79079b491accf9e83; home_feed_column=4; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MDk4NzI4NjgsImlhdCI6MTcwOTYxMzYwOCwicGx0IjotMX0.k5aKAapMZ3tu62GuMybeR386xwZLnjg4CPmsHagPSTY; bili_ticket_expires=1709872808; fingerprint=8d846a880e718f07d071627a9ac7f1f9; PVID=3; b_lsid=99771D39_18E0E4C5423; sid=6kt9e1co; buvid_fp=8d846a880e718f07d071627a9ac7f1f9; bp_video_offset_9552818=905375382297903141; browser_resolution=794-634",
        }
        params = {
            'search_type': 'video',  # 搜索类型为视频
            'keyword': keyword,      # 搜索关键词
            'page': page,            # 当前页码
        }

        try:
            # 发送GET请求获取数据
            r = requests.get(url, headers=headers, params=params)
            # 检查HTTP状态码（200表示请求成功）
            if r.status_code == 200:
                j_data = r.json()
                data_list = j_data['data']['result']  # 获取视频列表数据
                print('数据长度', len(data_list))

                # 解析每条视频数据
                collected_data = []
                for data in data_list:
                    # 清理标题中的HTML标签（如<em>等高亮标签）
                    title = re.compile(r'<[^>]+>', re.S).sub('', data['title'])
                    collected_data.append({
                        '标题': title,
                        '作者': data['author'],
                        'bvid': data['bvid'],         # 视频唯一标识
                        '上传时间': trans_data(data['pubdate']),  # 转换时间戳
                        '视频时长': data['duration'],   # 时长（秒）
                        '弹幕数': data.get('video_review', 0),  # 处理可能不存在的键
                        '点赞数': data.get('like', 0),   # 同上
                        '播放量': data['play'],         # 播放次数
                        '收藏量': data['favorites'],    # 收藏次数
                        '分区类型': data['typename'],   # 视频所属分区
                        '标签': data['tag'],            # 视频标签（逗号分隔）
                        '描述': data['description'],    # 视频简介
                    })

                # 写入CSV文件（追加模式，首次写入时添加表头）
                df = pd.DataFrame(collected_data)
                with open(out_file, 'a', encoding='utf-8-sig', newline='') as f:
                    df.to_csv(f, index=False, header=f.tell()==0)  # 仅首行写入表头

                print(f'第{page}页爬取完成。')
            else:
                print(f'请求失败，状态码：{r.status_code}')
        except Exception as e:
            print('发生错误:', e)

        time.sleep(1)  # 延迟1秒防止被封IP


if __name__ == '__main__':
    # 主程序入口
    search_keyword = '智能安防设备'  # 搜索关键词
    max_page = 30                     # 最大爬取30页
    result_file = 'item.csv'          # 输出文件路径
    search_bilibili(search_keyword, max_page, result_file)