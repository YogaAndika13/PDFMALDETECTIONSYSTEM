# PDF Malware Detection System

This is a web-based system built with Flask to detect malware in PDF files using a pre-trained machine learning model.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
    On Windows:
    ```bash
    venv\Scripts\activate
    ```
    On macOS/Linux:
    ```bash
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Verify model files:**
    -   Ensure your trained model files are in the `app/models/` directory:
        - `model_A_MLP_static.h5` and `pipeline_A_static.pkl`
        - `model_C_MLP_static_dynamic.h5` and `pipeline_C_static_dynamic.pkl`
        - `model_D_Stacking_static.pkl` and `pipeline_D_static.pkl`
        - `model_E_Stacking_static_dynamic.pkl` and `pipeline_E_static_dynamic.pkl`
    -   The system will automatically create the `uploads/` folder if it doesn't exist.

## Running the Application

Once everything is set up, you can run the application:

**For Development (with debug mode):**
```bash
set DEBUG=true
python run.py
```

**For Production:**
```bash
python run.py
```

The application will be served by Waitress (production) or Flask dev server (debug mode) and accessible at `http://localhost:5000`.

### Environment Variables

You can customize the application using environment variables:

- `DEBUG=true` - Enable debug mode (development only)
- `PORT=5000` - Set the port number
- `HOST=127.0.0.1` - Set the host address  
- `SECRET_KEY=your-secret-key` - Set a custom secret key for security

## Project Structure

-   `run.py`: The entry point to start the application.
-   `config.py`: Flask configuration settings.
-   `requirements.txt`: A list of all the python packages required.
-   `app/`: The main application folder.
    -   `__init__.py`: Initializes the Flask application.
    -   `routes.py`: Defines the application's routes and view logic.
    -   `utils.py`: Contains helper functions for feature extraction, model loading, and predictions.
    -   `models/`: **(Placeholder)** Place your trained model and pipeline files here.
    -   `templates/`: Contains the HTML templates for the web pages.
    -   `static/`: Contains static files like CSS and JavaScript.
-   `uploads/`: **(Placeholder)** A temporary folder for storing uploaded files.

