import sqlite3
from flask import Flask, render_template, request, url_for, flash, redirect
from demo.final_classifier import DeployedClassifier, LABEL_DICT

app = Flask(__name__)
app.config['SECRET_KEY'] = 'maishu is fat again, 555'

COLOR_PALLETE = {
    'happy': '#FCE220',
    'sad': '#00B0F0',
    'surprise': '#FC990A',
    'fear': '#b868f9',
    'neutral': '#b1b0b0',
    'angry': '#FF4C21'
}
sentiment_classifier = DeployedClassifier()


def get_db_connection(db_directory = 'demo/database.db'):
    '''Connect to the database

    Returns: a sqlite connection to the db
    '''
    conn = sqlite3.connect(db_directory)
    conn.row_factory = sqlite3.Row
    return conn

def get_post(post_id):
    '''Obtrain a post from db
    '''
    conn = get_db_connection()
    post = conn.execute('SELECT * FROM posts WHERE id = ?',
                        (post_id,)).fetchone()
    conn.close()
    return post

@app.route('/') 
def index():
    conn = get_db_connection()
    posts = conn.execute('SELECT * FROM posts').fetchall()
    conn.close()
    return render_template('index.html', posts=posts)

@app.route('/posts/<int:post_id>')
def post(post_id):
    post = get_post(post_id)
    return render_template('post.html', post=post)

@app.route('/posts/new', methods=('GET', 'POST'))
def new():
    '''pages to create a post
    '''
    emotions = None
    colors_in_order = None
    emotions_scipt = ''
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        if request.form['action'] == 'Submit':
            if not title:
                flash('Title can not be empty!')
            elif not content:
                flash('Content can not be empty')
            else:
                y_pred_all, y_pred = sentiment_classifier.pred(content)
                conn = get_db_connection()
                conn.execute("INSERT INTO posts (title, content, fear, neutral, sad, surprise, angry, happy) VALUES (?, ?, ?, ?, ?, ?, ?, ?);", 
                            (title, content) + tuple((proba for pred, proba in y_pred_all)))
                conn.commit()
                conn.close()
                return redirect(url_for('index'))
        elif request.form['action'] == 'Guess Emotion':
            if not content:
                flash('Content can not be empty')
            else:
                y_pred_all, y_pred = sentiment_classifier.pred(content)
                y_pred_all.sort(key = lambda x: x[1], reverse=True)
                emotions = {'Task' : 'Percentage'}
                emotions.update({pred: proba for pred, proba in y_pred_all})
                emotions_scipt = '｜'.join([f'{pred}({round(100*proba, 1)}%)' for pred, proba in y_pred_all][:3])
                print(emotions_scipt)
                colors_in_order = [COLOR_PALLETE[pred] for pred, _ in y_pred_all]
                # emotions = [f'{pred} ({round(100*proba, 3)}%)' for pred, proba in y_pred_all]
                
    return render_template('new.html', emotions=emotions, colors_in_order=colors_in_order, emotions_scipt=emotions_scipt)

@app.route('/posts/<int:id>/edit', methods=('GET', 'POST'))
def edit(id):
    '''pages to edit a post
    '''
    post = get_post(id)

    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']

        if not title:
            flash('Title is required!')
        else:
            conn = get_db_connection()
            conn.execute('UPDATE posts SET title = ?, content = ?'
                         ' WHERE id = ?',
                         (title, content, id))
            conn.commit()
            conn.close()
            return redirect(url_for('index'))
    return render_template('edit.html', post=post)

@app.route('/posts/<int:id>/delete', methods=('POST',))
def delete(id):
    post = get_post(id)
    conn = get_db_connection()
    conn.execute('DELETE FROM posts WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    flash('"{}" Successfully Delete!'.format(post['title']))
    return redirect(url_for('index'))

@app.route('/about')
def about():
    '''pages to summary all the diary
    '''
    conn = get_db_connection()
    query = ','.join([f'AVG({emotion}) AS {emotion}' for emotion in LABEL_DICT.keys()])
    emotions_query = conn.execute(f'SELECT {query} FROM posts').fetchone()
    y_all = [(emotion, emotions_query[emotion]) for emotion in LABEL_DICT.keys()]
    y_all.sort(key = lambda x: x[1], reverse=True)
    emotions = {emotion: emotion_level for emotion, emotion_level in y_all}
    emotions_scipt = '｜'.join([f'{emotion}({round(100*float(emotion_level), 1)}%)' for emotion, emotion_level in emotions.items()][:3])
    print(emotions_scipt)
    colors_in_order = [COLOR_PALLETE[emotion] for emotion, _ in y_all]
    output_emotions = {'Task' : 'Percentage'}
    output_emotions.update(emotions)
    return render_template('about.html', emotions=output_emotions, colors_in_order=colors_in_order)