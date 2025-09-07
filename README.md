# Sleep Disorder Prediction Flask App

This project is a Flask web application that predicts sleep disorders based on user input. It utilizes a machine learning model to provide predictions and is designed to be user-friendly with a simple web interface.

## Project Structure

```
my-flask-app
├── src
│   ├── app.py                # Main entry point of the Flask application
│   ├── templates
│   │   └── index.html        # HTML template for the web interface
│   ├── static
│   │   └── style.css         # CSS styles for the web application
│   └── models
│       └── __init__.py       # Model-related functionalities
├── requirements.txt          # Project dependencies
├── config.py                 # Configuration settings for the Flask app
└── README.md                 # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd my-flask-app
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```
   python src/app.py
   ```

5. **Access the application:**
   Open your web browser and go to `http://127.0.0.1:5000`.

## Usage

- Enter the required information in the form fields provided on the web interface.
- Click the "Predict" button to receive a prediction regarding sleep disorders based on the input data.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.