from flask import Flask, request, render_template
import numpy as np

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

# Home-Page Route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('race_ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = request.form.get('reading_score'),
            writing_score = request.form.get('writing_score')
        )
        
        # Data is converted into DataFrame.
        df = data.get_data_as_df()
        
        print(df)
        
        predict_pipeline = PredictPipeline()
        
        math_score = predict_pipeline.predict(df)
        
        return render_template('result.html', results = round(math_score[0], 2))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
