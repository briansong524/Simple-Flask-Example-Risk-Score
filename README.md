# Simple-Flask-Example-Risk-Score

A simple example on a full data pipeline from importing csv to running a live Flask service. The Flask service accepts GET requests and returns a JSON output. 

## Instructions

1. extract data.zip
2. run train.py
3. run app.py
4. test if the Flask service is working by typing http://127.0.0.1:8000/api?gender=M&own_car=Y&own_home=Y&n_children=0&income=100000 in an internet browser or otherwise

## Notes

The model sucks, I know. It is barely predictive, I'm using like half the features in the dataset, and the response variable is probably not well defined. I just designed it to be just barely predictable so the output it returns looks normal, while keeping the GET (input) parameters easy to modify. Otherwise, the parameter list would be super long and might stray away from the main intent of this.
