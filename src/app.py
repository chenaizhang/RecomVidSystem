import streamlit as st
import pandas as pd
from pathlib import Path

from main import run_usercf, run_wals, run_popularity, run_clustering

ITEM_CSV = Path('data/item.csv')

@st.cache_data
def load_items():
    df = pd.read_csv(ITEM_CSV)
    return df[[
        'itemId', 'length', 'comment', 'like', 'watch', 'share', 'name'
    ]]


def user_page():
    st.header('用户界面')
    items = load_items()
    if 'index' not in st.session_state:
        st.session_state['index'] = 0
    cols = st.columns(2)
    if cols[0].button('上一条'):
        st.session_state['index'] = (st.session_state['index'] - 1) % len(items)
    if cols[1].button('下一条'):
        st.session_state['index'] = (st.session_state['index'] + 1) % len(items)
    item = items.iloc[st.session_state['index']]
    st.subheader(item['name'])
    st.write(f"视频ID: {item['itemId']}")
    st.write(f"播放量: {item['watch']}")
    st.write(f"时长: {item['length']}")
    st.write(f"评论数: {item['comment']}")
    st.write(f"点赞数: {item['like']}")
    st.write(f"分享量: {item['share']}")


def admin_page():
    st.header('后台管理')
    if st.button('运行协同过滤推荐'):
        df = run_usercf()
        st.success('F3.csv 生成完毕')
        st.dataframe(df.head())
    if st.button('运行 WALS 推荐'):
        df = run_wals()
        st.success('F4.csv 生成完毕')
        st.dataframe(df.head())
    if st.button('计算视频热度'):
        df = run_popularity()
        st.success('F5.csv 生成完毕')
        st.dataframe(df.head())
    if st.button('执行聚类分析'):
        users, items = run_clustering()
        st.success('F6.csv 和 F7.csv 已生成')
        st.subheader('用户聚类')
        st.dataframe(users.head())
        st.subheader('视频聚类')
        st.dataframe(items.head())


def main():
    page = st.sidebar.selectbox('选择页面', ['用户界面', '后台管理'])
    if page == '用户界面':
        user_page()
    else:
        admin_page()


if __name__ == '__main__':
    main()
