from app import application
from flask import render_template, flash, redirect, Response , url_for , send_file
from app.forms import LoginForm, RegistrationForm, CreateQuestionForm 
from flask_login import current_user, login_user, logout_user, login_required
from app.models import User, Question, TrainingData 
from urllib.parse import urlparse, unquote
from app import db
from flask import request 
from app.serverlibrary import  predict_linreg, trainModel , split_data, filepath
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io


# Home route to display the prediction form and results
@application.route('/')
@application.route('/index/', methods=["GET", "POST"])
@login_required
def index():
	prefix = application.wsgi_app.prefix[:-1]
	prediction , Temperature, Nitrogen, Potassium= None,0,0,0  # Default state for the prediction
	beta, means, stds = trainModel()

	if request.method == "POST":
		try:
			# Retrieve input from the form
			Temperature = float(request.form.get('Temperature', 0))
			Nitrogen = float(request.form.get('Nitrogen', 0))
			Potassium = float(request.form.get('Potassium', 0))
			
			# Prepare input and make a prediction
			user_input = np.array([[Temperature, Nitrogen, Potassium]])
			print(user_input)
			beta, means, stds = trainModel()
			prediction = predict_linreg(user_input, beta, means, stds)
			prediction = round(prediction[0][0], 2)  # Extract scalar and round to 2 decimals
		except ValueError as e:
			print(e)
			flash("Invalid input. Please enter valid numbers for rainfall and fertilizer.")
			return redirect(url_for('index'))
			
	# Render the index.html template with the prediction result
	return render_template('index.html', title='Home', prefix=prefix, prediction=prediction, Temperature = Temperature, Nitrogen=Nitrogen, Potassium=Potassium)



@application.route('/metrics/')
@login_required
def metrics():
	prefix = application.wsgi_app.prefix[:-1]
	return render_template('metrics.html', title='metrics', prefix=prefix)

# Questions route to display and create new questions
@application.route('/questions/', methods=['GET','POST'])
@login_required
def questions():
	prefix = application.wsgi_app.prefix[:-1]
	questions = current_user.questions.order_by(Question.created_at.desc()).all() # get all questions 
	form = CreateQuestionForm() # create comments form

	if form.validate_on_submit():
		questions = Question(expression=form.expression.data)
		questions.author = current_user.id
		db.session.add(questions)
		db.session.commit()
		flash('Your comment has been posted.')
		questions = current_user.questions.all()  # Refresh comments after adding
		
	return render_template('questions.html', title='Questions', 
							user=current_user,
							questions=questions,
							form=form, prefix=prefix)

# Forum route to display all questions and their authors
@application.route('/Forum/') # change to get questions table
def Forum():
	prefix = application.wsgi_app.prefix[:-1]
	questions = Question.query.order_by(Question.created_at.desc()).all()
	users = User.query.all()
	userlist = [(u.username) for u in users]
	return render_template('Forum.html', title="Forum", questions=questions, userlist = userlist, prefix=prefix)


# Login route
@application.route('/login/', methods=['GET', 'POST'])
def login():
	prefix = application.wsgi_app.prefix[:-1]
	if current_user.is_authenticated:
		return redirect(url_for('index'))

	form = LoginForm()

	if form.validate_on_submit():
		user = User.query.filter_by(username=form.username.data).first()
		if user is None or not user.check_password(form.password.data):
			flash('Invalid username or password')
			return redirect(url_for('login'))
		login_user(user, remember=form.remember_me.data)
		if (request.args.get('next')) is None:
			next_page = None
		else:
			next_page = unquote(request.args.get('next'))

		if not next_page or urlparse(next_page).netloc != '':
			next_page = url_for('index')
		return redirect(next_page)
	return render_template('login.html', title='Sign In', form=form, prefix=prefix)

# Logout route
@application.route('/logout/')
def logout():
	#prefix = application.wsgi_app.prefix[:-1]
	logout_user()
	return redirect(url_for('index'))

# Registration route
@application.route('/register/', methods=['GET', 'POST'])
def register():
	prefix = application.wsgi_app.prefix[:-1]
	if current_user.is_authenticated:
		return redirect(url_for('index'))
	form = RegistrationForm()
	if form.validate_on_submit():
		user = User(username=form.username.data)
		user.set_password(form.password.data)
		db.session.add(user)
		db.session.commit()
		flash('Congratulations, you are now a registered user.')
		return redirect(url_for('login'))
	return render_template('register.html', title='Register', form=form, prefix=prefix)

# Function to import training data from a specified CSV file
def import_csv(file_path):
    # Read the CSV file
	data = pd.read_csv(file_path)
	try:
		for _, row in data.iterrows():

			new_entry = TrainingData(
				rainfall=row['Rain Fall (mm)'],  
				fertilizer=row['Fertilizer'],
				temperature=row['Temperatue'],
				nitrogen=row['Nitrogen (N)'],
				phosphorus=row['Phosphorus (P)'],
				potassium=row['Potassium (K)'],
				yield_=row['Yield (Q/acre)']
			)
			db.session.add(new_entry)
		db.session.commit()  # Save all changes to the database
		print("Data added successfully!")
	except Exception as e:
		db.session.rollback()  # Roll back if there’s an error
		print(f"An error occurred: {e}")

# Serve a regression plot based on the current training data
@application.route('/regression-plot')
def regression_plot():
	query_result = db.session.query(TrainingData).all()

	# Extract features and target
	df_features = pd.DataFrame([{
		'Temperature': record.temperature,
		'Nitrogen (N)': record.nitrogen,
		'Potassium (K)': record.potassium
	} for record in query_result])

	df_target = pd.DataFrame([{'Yield': record.yield_} for record in query_result])

	#Data cleaning features 
	df_features.dropna(inplace=True)
	df_target.dropna(inplace=True)
	df_features['Temperature'] = pd.to_numeric(df_features['Temperature'], downcast='float')

	df_features_train, df_features_test, df_target_train, df_target_test = split_data(df_features, df_target, 100, 0.3)
	beta, means, stds = trainModel()
	pred: np.ndarray = predict_linreg(df_features_test.to_numpy(), beta, means, stds)

	# Generate plots
	features = ['Temperature', 'Nitrogen (N)', 'Potassium (K)']
	plt.figure(figsize=(10, 5))  # Adjust figure size
	X_test = df_features_test.to_numpy()
	y_test = df_target_test.to_numpy()
	for i, feature in enumerate(features):
		plt.subplot(1, 3, i+1)
		plt.scatter(X_test[:, i], y_test.flatten(), alpha=0.7, c='blue', label="Actual")
		plt.scatter(X_test[:, i], pred, c='orange', label="Predicted")
		slope, intercept = np.polyfit(X_test[:, i], y_test.flatten(), 1)
		plt.plot(X_test[:, i], slope * X_test[:, i] + intercept, color='black', label="Best Fit Line")
		plt.xlabel(feature)
		plt.ylabel("Yield (Q/acre)")
		plt.title(f"{feature} vs. Yield")
		plt.legend()


	plt.tight_layout()
	buf = io.BytesIO()
	plt.savefig(buf, format='png')
	buf.seek(0)
	plt.close()
	return Response(buf.getvalue(), mimetype='image/png')

# Check if the training data table is empty
def is_database_None(): # check if the training data database is none
	return db.session.query(TrainingData).first() is None

@application.route('/training/') # change to get questions table
def training():
	prefix = application.wsgi_app.prefix[:-1]
	length = db.session.query(TrainingData).count()
	print(length)
	return render_template('TrainingData.html', title="training", prefix=prefix, length = length)


@application.route('/upload', methods=['POST'])
def upload():
	prefix = application.wsgi_app.prefix[:-1]

	if 'file' not in request.files:

		flash("No file part")
		return redirect(url_for('training'))

	file = request.files['file']
	if file.filename == '':
		flash("No selected file")
		return redirect(url_for('training'))
	
	if file and file.filename.endswith('.csv'):
		# Read the uploaded CSV file
		data = pd.read_csv(file)
		for _, row in data.iterrows():
				record = TrainingData(
					fertilizer= 0 ,
					temperature=row['Temperature'],
					nitrogen=row['Nitrogen (N)'],
					phosphorus= 0 ,
					potassium=row['Potassium (K)'],
					yield_=row['Yield (Q/acre)']
				)

				db.session.add(record)
		flash('Your File has been Added.')
		length = db.session.query(TrainingData).count()
		db.session.commit()
		return render_template('TrainingData.html', title="training", prefix = prefix, length =  length)
	else:
		return "Unsupported file format", 400

# Reset training data in the database to the original state
@application.route('/clear_training_data', methods=['POST'])
def clear_training_data():
	prefix = application.wsgi_app.prefix[:-1]
	#C:/Users/leeji/Desktop/d2w_mini_projects/mp_calc/app/static/crop yield data sheet.csv
	try:
		# Delete all rows from the table
		num_rows_deleted = db.session.query(TrainingData).delete()
		
		if is_database_None():
			import_csv(filepath)
		length = db.session.query(TrainingData).count()
		db.session.commit()
		flash('The database has been successfully reset')
		print(f"Successfully deleted {num_rows_deleted} rows from the training data table!")
		return render_template('TrainingData.html', title="training", prefix = prefix, length = length)
	
	except Exception as e:
		flash('Some error has occured, please reload the application')
		
		db.session.rollback()
		length = db.session.query(TrainingData).count()
		print(f"An error occurred: {str(e)}", 500)
		return render_template('TrainingData.html', title="training", prefix = prefix, length = length)


# Serve a downloadable example CSV file for user reference
@application.route('/download_example', methods=['GET'])
def download_example():
    file_path = 'static/example.csv'  # Path to the example file
    return send_file(file_path, as_attachment=True)