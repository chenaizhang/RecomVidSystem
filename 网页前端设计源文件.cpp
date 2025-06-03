from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import uuid
import random
from sqlalchemy.sql import func

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///aikandb.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 确保上传文件夹存在
if not os.path.exists(app.config['UPLOAD_FOLDER']) :
    os.makedirs(app.config['UPLOAD_FOLDER'])

    # 初始化数据库
    db = SQLAlchemy(app)

    # 初始化登录管理器
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'

    # 用户模型
    class User(UserMixin, db.Model) :
    id = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String(150), unique = True, nullable = False)
    password_hash = db.Column(db.String(150), nullable = False)
    email = db.Column(db.String(150), unique = True, nullable = False)
    avatar = db.Column(db.String(150), default = 'default_avatar.png')
    bio = db.Column(db.String(300))
    created_at = db.Column(db.DateTime, default = datetime.utcnow)
    videos = db.relationship('Video', backref = 'user', lazy = True)
    likes = db.relationship('Like', backref = 'user', lazy = True)
    comments = db.relationship('Comment', backref = 'user', lazy = True)

    def set_password(self, password) :
    self.password_hash = generate_password_hash(password)

    def check_password(self, password) :
    return check_password_hash(self.password_hash, password)

    # 视频模型
    class Video(db.Model) :
    id = db.Column(db.Integer, primary_key = True)
    title = db.Column(db.String(150), nullable = False)
    description = db.Column(db.Text)
    filename = db.Column(db.String(150), nullable = False)
    thumbnail = db.Column(db.String(150), nullable = False)
    category = db.Column(db.String(50), nullable = False)
    views = db.Column(db.Integer, default = 0)
    likes = db.Column(db.Integer, default = 0)
    comments_count = db.Column(db.Integer, default = 0)
    created_at = db.Column(db.DateTime, default = datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable = False)
    comments = db.relationship('Comment', backref = 'video', lazy = True)
    likes = db.relationship('Like', backref = 'video', lazy = True)

    # 评论模型
    class Comment(db.Model) :
    id = db.Column(db.Integer, primary_key = True)
    content = db.Column(db.Text, nullable = False)
    created_at = db.Column(db.DateTime, default = datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable = False)
    video_id = db.Column(db.Integer, db.ForeignKey('video.id'), nullable = False)

    # 点赞模型
    class Like(db.Model) :
    id = db.Column(db.Integer, primary_key = True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable = False)
    video_id = db.Column(db.Integer, db.ForeignKey('video.id'), nullable = False)
    created_at = db.Column(db.DateTime, default = datetime.utcnow)

    # 用户加载回调
    @login_manager.user_loader
    def load_user(user_id) :
    return User.query.get(int(user_id))

    # 首页路由
    @app.route('/')
    def index() :
    # 获取推荐视频 - 随机选择一些热门视频
    featured_videos = Video.query.order_by(func.random()).limit(10).all()
    # 获取最新视频
    latest_videos = Video.query.order_by(Video.created_at.desc()).limit(10).all()
    return render_template('index.html', featured_videos = featured_videos, latest_videos = latest_videos)

    # 视频播放页
    @app.route('/video/<int:video_id>')
    def video_detail(video_id) :
    video = Video.query.get_or_404(video_id)
    video.views += 1
    db.session.commit()

    # 获取相关视频
    related_videos = Video.query.filter(Video.category == video.category, Video.id != video.id)\
    .order_by(func.random()).limit(6).all()

    # 获取评论
    comments = Comment.query.filter_by(video_id = video_id).order_by(Comment.created_at.desc()).all()

    # 检查用户是否已点赞
    liked = False
    if current_user.is_authenticated:
liked = Like.query.filter_by(user_id = current_user.id, video_id = video_id).first() is not None

return render_template('video_detail.html', video = video, related_videos = related_videos, comments = comments, liked = liked)

# 分类浏览
@app.route('/category/<category>')
def category(category) :
    videos = Video.query.filter_by(category = category).order_by(Video.created_at.desc()).paginate(per_page = 12)
    return render_template('category.html', videos = videos, category = category)

    # 用户注册
    @app.route('/register', methods = ['GET', 'POST'])
    def register() :
    if request.method == 'POST' :
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(username = username).first() :
            return jsonify({ "error": "用户名已存在" }), 400

            if User.query.filter_by(email = email).first() :
                return jsonify({ "error": "邮箱已注册" }), 400

                user = User(username = username, email = email)
                user.set_password(password)
                db.session.add(user)
                db.session.commit()

                login_user(user)
                return redirect(url_for('index'))

                return render_template('register.html')

                # 用户登录
                @app.route('/login', methods = ['GET', 'POST'])
                def login() :
                if request.method == 'POST' :
                    username = request.form['username']
                    password = request.form['password']

                    user = User.query.filter_by(username = username).first()

                    if user is None or not user.check_password(password) :
                        return jsonify({ "error": "用户名或密码错误" }), 400

                        login_user(user)
                        return redirect(url_for('index'))

                        return render_template('login.html')

                        # 用户登出
                        @app.route('/logout')
                        @login_required
                        def logout() :
                        logout_user()
                        return redirect(url_for('index'))

                        # 上传视频页面
                        @app.route('/upload', methods = ['GET', 'POST'])
                        @login_required
                        def upload() :
                        if request.method == 'POST' :
                            title = request.form['title']
                            description = request.form['description']
                            category = request.form['category']
                            video_file = request.files['video']
                            thumbnail_file = request.files['thumbnail']

                            if not video_file or not thumbnail_file:
return jsonify({ "error": "请上传视频和缩略图" }), 400

# 生成唯一文件名
video_filename = secure_filename(f"{uuid.uuid4()}_{video_file.filename}")
thumbnail_filename = secure_filename(f"{uuid.uuid4()}_{thumbnail_file.filename}")

# 保存文件
video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
thumbnail_path = os.path.join(app.config['UPLOAD_FOLDER'], thumbnail_filename)

video_file.save(video_path)
thumbnail_file.save(thumbnail_path)

# 创建视频记录
video = Video(
    title = title,
    description = description,
    filename = video_filename,
    thumbnail = thumbnail_filename,
    category = category,
    user_id = current_user.id
)

db.session.add(video)
db.session.commit()

return redirect(url_for('video_detail', video_id = video.id))

return render_template('upload.html')

# 处理点赞
@app.route('/like/<int:video_id>', methods = ['POST'])
@login_required
def like(video_id) :
    video = Video.query.get_or_404(video_id)

    # 检查用户是否已点赞
    like_entry = Like.query.filter_by(user_id = current_user.id, video_id = video_id).first()

    if like_entry:
# 取消点赞
db.session.delete(like_entry)
video.likes -= 1
action = 'unlike'
    else:
# 添加点赞
like_entry = Like(user_id = current_user.id, video_id = video_id)
db.session.add(like_entry)
video.likes += 1
action = 'like'

db.session.commit()

return jsonify({ "status": "success", "action" : action, "likes" : video.likes })

# 发表评论
@app.route('/comment/<int:video_id>', methods = ['POST'])
@login_required
def comment(video_id) :
    video = Video.query.get_or_404(video_id)
    content = request.form['content']

    if not content :
        return jsonify({ "error": "评论内容不能为空" }), 400

        comment = Comment(
            content = content,
            user_id = current_user.id,
            video_id = video_id
        )

        db.session.add(comment)
        video.comments_count += 1
        db.session.commit()

        return redirect(url_for('video_detail', video_id = video_id))

        # 搜索功能
        @app.route('/search')
        def search() :
        query = request.args.get('query', '')
        videos = Video.query.filter(Video.title.ilike(f'%{query}%')).order_by(Video.created_at.desc()).all()
        return render_template('search.html', videos = videos, query = query)

        # 提供视频文件
        @app.route('/uploads/<filename>')
        def serve_file(filename) :
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # 用户个人主页
        @app.route('/user/<int:user_id>')
        def user_profile(user_id) :
        user = User.query.get_or_404(user_id)
        videos = Video.query.filter_by(user_id = user_id).order_by(Video.created_at.desc()).all()

        # 检查当前用户是否关注了该用户
        is_following = False
        if current_user.is_authenticated and current_user.id != user_id:
# 这里可以实现关注功能
pass

return render_template('user_profile.html', profile_user = user, videos = videos, is_following = is_following)

if __name__ == '__main__':
# 创建数据库表
with app.app_context() :
    db.create_all()
    app.run(debug = True)