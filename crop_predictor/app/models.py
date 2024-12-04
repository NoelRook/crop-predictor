from app import db
from app import login
from datetime import datetime 
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

@login.user_loader
def load_user(id):
	"""
    Load a user by their ID.
    This is required by Flask-Login to manage user sessions.
    """
	return User.query.get(int(id))

class User(UserMixin, db.Model):
	"""
    Represents a user in the application.
    Inherits from `UserMixin` to provide default implementations for Flask-Login.
    """
	__tablename__ = 'user'
	id = db.Column(db.Integer, primary_key=True)
	username = db.Column(db.String(64), index=True, unique=True)
	password_hash = db.Column(db.String(128))
	questions = db.relationship('Question', backref='from_user', 
								lazy='dynamic')
	def set_password(self, password):
		self.password_hash = generate_password_hash(password)

	def check_password(self, password):
		return check_password_hash(self.password_hash, password)

	def __repr__(self):
		return f'<User {self.username:}>'


class Question(db.Model): # chage this to comments table
	"""
    Represents a question or comment associated with a user.
    Each question is linked to a user through a foreign key.
    """
	__tablename__ = 'question'
	id = db.Column(db.Integer, primary_key=True)
	expression = db.Column(db.String(255))
	created_at = db.Column(db.DateTime, default=datetime.now()) 
	author = db.Column(db.Integer, db.ForeignKey('user.id'))

	def __repr__(self):
		return f'<Question {self.expression:}>'	
	
	
class TrainingData(db.Model): 
	"""
    Represents the training data stored in the database.
    Used to build and train machine learning models.
    """
	__tablename__= "trainingdata"
	id = db.Column(db.Integer, primary_key=True)
	rainfall = db.Column(db.Integer)
	fertilizer = db.Column(db.Integer)
	temperature = db.Column(db.Integer)
	nitrogen = db.Column(db.Integer)
	phosphorus = db.Column(db.Integer)
	potassium = db.Column(db.Integer)
	yield_ = db.Column(db.Integer)
	