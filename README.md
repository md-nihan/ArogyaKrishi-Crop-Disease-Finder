# ArogyaKrishi: Crop Disease Finder

AI-powered crop disease detection developed by Nihan. This application uses a custom-trained AI model to identify plant diseases from leaf images and provides detailed treatment recommendations.

## Features

- Upload leaf photos for disease analysis
- AI-powered predictions with confidence scores
- Detailed disease documentation with treatment options
- Prediction history tracking
- Weather-based disease risk alerts
- Responsive, modern UI

## Technology Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Node.js with Express
- **Database**: MongoDB (Atlas)
- **AI Server**: Python with Flask
- **AI Models**: TensorFlow (image classification)

## Setup Instructions

### Prerequisites

- Node.js and npm
- Python 3.9.13
- MongoDB Atlas account

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/md-nihan/ArogyaKrishi-Crop-Disease-Finder.git
   ```

2. Install Node.js dependencies:
   ```bash
   cd plant-disease-scanner
   npm install
   ```

3. Install Python dependencies:
   ```bash
   cd ../ai-server
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the AI server:
   ```bash
   cd ai-server
   python ai_server.py
   ```

2. Start the web server:
   ```bash
   cd plant-disease-scanner
   npm start
   ```

3. Open your browser and navigate to `http://localhost:3000`

## Project Structure

```
├── ai-server/
│   ├── ai_server.py          # Flask API for AI model
│   ├── leaf_disease_model.keras  # Trained AI model
│   ├── class_names.pkl       # Disease class names
│   └── requirements.txt      # Python dependencies
│
├── plant-disease-scanner/
│   ├── server.js             # Main Node.js server
│   ├── package.json          # Node.js dependencies
│   ├── populateDiseaseInfo.js # Script to populate disease documentation
│   ├── models/               # Mongoose models
│   │   ├── DiseaseInfo.js
│   │   ├── Prediction.js
│   │   └── Result.js
│   └── public/               # Frontend files
│       ├── index.html
│       ├── history.html
│       ├── script.js
│       ├── history.js
│       └── style.css
```

## Author

Nihan - Custom AI model developer and full-stack developer

## License

This project is proprietary and developed by Nihan.