from flask import Flask, render_template, request
import pandas as pd
from pathlib import Path

from main import run_usercf, run_wals, run_popularity, run_clustering

app = Flask(__name__)

ITEM_CSV = Path('data/item.csv')
items = pd.read_csv(ITEM_CSV)[['itemId', 'length', 'comment', 'like', 'watch', 'share', 'name']]

@app.route('/')
@app.route('/user')
def user():
    idx = int(request.args.get('index', 0))
    item = items.iloc[idx % len(items)]
    return render_template(
        'user.html',
        item=item,
        index=idx,
        total=len(items),
        title='用户界面'
    )


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    result = None
    result_name = ''
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'usercf':
            df = run_usercf()
            result = df.head().to_html(index=False)
            result_name = 'F3.csv'
        elif action == 'wals':
            df = run_wals()
            result = df.head().to_html(index=False)
            result_name = 'F4.csv'
        elif action == 'popularity':
            df = run_popularity()
            result = df.head().to_html(index=False)
            result_name = 'F5.csv'
        elif action == 'clustering':
            users, items_df = run_clustering()
            result = '<h4>用户聚类</h4>' + users.head().to_html(index=False)
            result += '<h4>视频聚类</h4>' + items_df.head().to_html(index=False)
            result_name = 'F6.csv & F7.csv'
    return render_template('admin.html', result=result, result_name=result_name, title='后台管理')


if __name__ == '__main__':
    app.run(debug=True)
