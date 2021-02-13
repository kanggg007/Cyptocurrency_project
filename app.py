from flask import Flask, render_template, redirect
# from flask_pymongo import PyMongo
from sqlalchemy import create_engine
import json
import pandas as pd
from config2 import user_name
from config2 import aws_password
from config2 import db_password
# import scraping

app = Flask(__name__)
# url='cryptodb.crgu064gyupd.us-east-2.rds.amazonaws.com'
# aws_string=f"postgresql://{user_name}:{aws_password}@{url}:5432/postgres"
# engine = create_engine(aws_string)

db_string = f"postgres://postgres:{db_password}@localhost/cryptocurrency_db"
engine = create_engine(db_string)
# Use flask_pymongo to set up mongo connection
# app.config["MONGO_URI"] = "mongodb://localhost:27017/mars_app"
# mongo = PyMongo(app)


@app.route("/")
def index():
#    mars = mongo.db.mars.find_one()
    df = pd.read_sql_query('SELECT asset_id, time, close FROM all_data', con=engine)
    
    return render_template('index.html', df=json.loads(df.to_json(orient='records')))

# @app.route("/scrape")
# def scrape():
#    mars = mongo.db.mars
#    mars_data = scraping.scrape_all()
#    mars.update({}, mars_data, upsert=True)
#    return redirect("/")



if __name__ == "__main__":
   app.run(debug=True)