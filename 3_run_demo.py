from demo import app
import os
if __name__ == '__main__':
    # Initialize a database if it is not initialized
    if not os.path.exists('demo/database.db'):
        print('Initializing Database...')
        exec(open('demo/init_db.py').read())
    # Run the front end inside the 'demo' module
    app.run(debug=True, host = '127.0.0.1', port = 8111)