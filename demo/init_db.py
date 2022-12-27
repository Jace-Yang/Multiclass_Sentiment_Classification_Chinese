import sqlite3

# Connet to db
connection = sqlite3.connect('demo/database.db')

# Initialize table
with open('demo/db.sql') as f:
    connection.executescript(f.read())

# Insert 2 diary
cur = connection.cursor()
initial_rows = [
    ('Orientation', 'I am so happy today!', 0, 0, 0, 0, 0, 1),
    ('Day before project deadline', 'This is just so many work!!', 0.3, 0, 0.7, 0, 0, 0)
]
cur.executemany("INSERT INTO posts (title, content, fear, neutral, sad, surprise, angry, happy) VALUES (?, ?, ?, ?, ?, ?, ?, ?);", initial_rows)

# Commit the data manipulation
connection.commit()

# Close the cursor
connection.close()
