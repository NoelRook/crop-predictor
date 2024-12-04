
# Crop predictor
- Created this web application using flask and python for the Term 3 module Data Driven World

### Key features

- Enable users to predict crop yields based on environmental factors like temperature, potassium, and nitrogen.
- Allow users to interact through a forum system.
- Store user and crop-related data securely in a database.
- Utilize machine learning techniques (linear regression) to analyze and predict yields.

### Features

- User Authentication:
        Secure login and registration with password hashing.
        User sessions handled via Flask-Login.

- Forum:
        Users can post questions and interact with others.

- Machine Learning:
        Train models using a provided dataset.
        Predict crop yields based on features like temperature, rainfall, and fertilizers.

- Dynamic Web Pages:
        HTML templates rendered with Flask.

- Database Integration:
        Store user data, questions, and training data in a structured SQL database.
## Table of content
- Installation
- Application Structure
- Database model
- License

## Installation
1. Download the file from vocareum

2. Open the location of the folder in the terminal

```
cd crop_predictor
```

### create virtual environment 
3. In a terminal with the folder "crop_predictor", run the following commands


Install pipenv package
```
python -m pip install --user pipenv
```
From the root folder, install the packages specified in the Pipfile.
```
python -m pipenv install
```

Start the virtual environment
```
python -m pipenv shell
```

To ensure that the database is initiated, run the following commands
```
flask db init
flask db migrate
flask db upgrade
```

Now, run flask by typing
```
flask run
```


You should see that some output will be thrown out, which one of them would be:
```
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

To stop the web app, type `CTRL+C`.


If you are running this project in Vocareum, do the following:
1. Before running `flask run`, edit the file `crop_predictor/app/__init__.py` and change the argument from `voc=False` to `voc=True` in the following line.
    ```python
    application.wsgi_app = PrefixMiddleware(application.wsgi_app, voc=False)
    ```
1. Go to `crop_predictor` directory or your root directory and type `flask run` in the terminal.
1. you can do a CTRL-click (Windows) or CMD-click on the `http://127.0.0.1:5000` link in the terminal and it will open a new tab in your browser. 

## Application Structure

### Root Directory
- `crop_predictor`
    - `app`
        - `static` : Stores static assets like images, datasets, and scripts.
        - `templates` : HTML files for rendering web pages.
        - `base.py` : Base template for navbar to navigate the website
        - `init.py` : Initializing the app and its extensions
        - `forms.py` : Form handling for user inputs.
        - `middleware.py` : Custom middleware for handling request prefixes.
        - `models.py` :Database models for users, questions, and training data.
        - `routes.py` : Application routes and logic for user interactions.
        - `serverlibrary.py` : Functions that enable utilities throughout the website 
    - `migrations`
        - `env.py`
        - `README.md`
    - `app.db`
    - `application.py`
    - `config.py` 
    - `Pipfile`: contains all the imports that need to be downloaded to enable the functionalities

## Database model

### User Model
- Fields
    - `id` : Unique identifier of the User
    - `username` : Username of the user.
    - `password_hash` : Hashed password.


### Question  Model
- Fields
    - `id` : Unique identifier for the question.
    - `username` : Username of the user.
    - `created_at` : Timestamp of creation.
    - `author` : Foreign key linking to the `User` model.

### TrainingData  Model
- Fields
    - `id` : Unique identifier.
    - Variables (temperature, nitrogen, potassium, etc).
    - `yield_` : Target variable (crop yield).
## License

[MIT](https://choosealicense.com/licenses/mit/)

