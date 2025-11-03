# 🚀 QUICK START GUIDE - Boat Classification Project

## ⚡ Fast Track (If You Know What You're Doing)

```powershell
# 1. Train the model (run all cells in notebook)
# 2. Start backend
.\start.ps1

# 3. Open frontend/index.html in browser
```

## 📚 Complete Step-by-Step Guide (For Beginners)

### STEP 1: Train the Machine Learning Model

1. Open VS Code
2. Find and open `boat-classification.ipynb`
3. Click "Select Kernel" at the top right → Choose Python environment
4. Click "Run All" button (or run each cell from top to bottom)
5. Wait for training to complete (30-60 minutes)
   - You'll see progress bars and training metrics
   - The model will be saved as `boat_classifier_mobilenet.h5`

**What's happening?**
- The notebook splits your boat images into training/validation/test sets
- It trains two AI models (CNN and MobileNetV2)
- It compares their performance
- It saves the best model for use in the web app

### STEP 2: Start the Backend Server

1. Open a new terminal in VS Code (Terminal → New Terminal)
2. Run this command:
   ```powershell
   .\start.ps1
   ```
3. You should see:
   ```
   [OK] Model found
   Starting backend server...
   Server will run on http://localhost:5000
   ```
4. **Keep this terminal open!** The backend needs to keep running.

**What's happening?**
- The script loads your trained AI model
- It starts a web server on your computer (port 5000)
- This server will process image uploads and return predictions

### STEP 3: Open the Frontend

1. Open File Explorer
2. Navigate to your project folder
3. Go into the `frontend` folder
4. Double-click `index.html`
5. It will open in your web browser

**What's happening?**
- The HTML file creates a user interface
- When you upload an image, it sends it to the backend (Step 2)
- The backend uses AI to classify the boat
- The result is shown on the webpage

### STEP 4: Test the Classifier!

1. Click "Choose Boat Image" button
2. Select any boat image from your dataset or internet
3. Wait a few seconds
4. See the results:
   - Predicted boat type
   - Confidence score (how sure the AI is)
   - All probabilities for each boat type

## 🗂️ Project Structure Explained

```
boat-classification/
├── backend/                    # The "brain" of the app
│   ├── app.py                 # Python server code
│   ├── requirements.txt       # Required Python packages
│   └── boat_classifier_mobilenet.h5  # Trained AI model
│
├── frontend/                   # The "face" of the app
│   └── index.html             # Web page you interact with
│
├── data/                       # Your training data (created by notebook)
│   ├── train/                 # Images for teaching the AI
│   ├── validation/            # Images for tuning the AI
│   └── test/                  # Images for final testing
│
├── boat-classification.ipynb   # Training notebook
├── start.ps1                   # Quick start script
└── README.md                   # Project documentation
```

## 🎯 Supported Boat Types

The AI can recognize these 9 types of boats:

1. **Buoy** - Floating markers in water
2. **Cruise Ship** - Large passenger ships
3. **Ferry Boat** - Boats that transport people/vehicles
4. **Freight Boat** - Cargo/container ships
5. **Gondola** - Traditional Venetian rowing boats
6. **Inflatable Boat** - Rubber dinghies/rafts
7. **Kayak** - Small paddle boats
8. **Paper Boat** - Origami-style paper boats
9. **Sailboat** - Wind-powered sailing vessels

## ❓ Troubleshooting

### Problem: "Model not found" error
**Solution:** 
- Make sure you ran ALL cells in the notebook
- Check if `boat_classifier_mobilenet.h5` file exists
- If it's in the main folder, move it to the `backend` folder

### Problem: Backend won't start
**Solution:**
```powershell
cd backend
pip install -r requirements.txt
python app.py
```

### Problem: Frontend shows "Network Error"
**Solution:**
- Make sure the backend is running (Step 2)
- Check if you see "Server will run on http://localhost:5000"
- Try refreshing the frontend page

### Problem: Prediction is wrong
**Solution:**
- The AI isn't perfect! It learned from your training data
- Try clearer images with better lighting
- Make sure the boat is the main subject in the image
- Check the "All Predictions" section to see other possibilities

### Problem: ImportError when running backend
**Solution:**
```powershell
pip install flask flask-cors tensorflow numpy pillow
```

## 📊 Understanding the Results

### Confidence Score
- **80-100%**: Very confident, likely correct
- **60-80%**: Confident, probably correct
- **40-60%**: Uncertain, could be multiple types
- **Below 40%**: Not confident, image may be unclear

### All Predictions
Shows confidence for ALL 9 boat types. Example:
```
Sailboat: 85.2%  ← AI thinks it's most likely a sailboat
Kayak: 8.3%      ← Small chance it's a kayak
Ferry: 3.1%      ← Very small chance it's a ferry
...
```

## 🎓 Learning Resources

Want to understand how it works?

1. **Read the notebook comments** - Every step is explained
2. **Look at app.py** - Backend code has detailed comments
3. **Check index.html** - Frontend code is well-documented

## 🛠️ Advanced: Improving the Model

Want better accuracy? Try these:

1. **More training data** - Add more images per boat type
2. **Better data quality** - Use clear, high-resolution images
3. **More epochs** - Train for longer (change `epochs=50` to `epochs=100`)
4. **Fine-tuning** - Unfreeze some MobileNetV2 layers
5. **Data augmentation** - Increase rotation_range, zoom_range

## 💡 Common Questions

**Q: How accurate is the model?**
A: Check the notebook's confusion matrix and classification report after training. Typically 70-90% depending on your dataset quality.

**Q: Can I add new boat types?**
A: Yes! Add a new folder in the dataset with images, retrain the model, and update the CLASS_NAMES in app.py.

**Q: Can I deploy this online?**
A: Yes! You can deploy to Heroku, AWS, Google Cloud, or similar services. You'll need to modify the frontend to point to your online backend URL.

**Q: Why MobileNetV2?**
A: It's lightweight, fast, and pre-trained on millions of images. It's perfect for image classification tasks like this.

**Q: How long does training take?**
A: 30-60 minutes on a modern CPU, 10-20 minutes on a GPU.

## 📞 Need Help?

1. Check the error messages carefully
2. Read this guide again
3. Look at the code comments
4. Search the error message online
5. Check if backend is running when testing frontend

## 🎉 Success Checklist

- ✅ Notebook ran successfully
- ✅ Model file exists in backend folder
- ✅ Backend server is running
- ✅ Frontend opens in browser
- ✅ Can upload and classify images
- ✅ Results show prediction and confidence

**Congratulations! Your AI boat classifier is working! 🚢**
