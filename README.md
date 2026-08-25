# 🎓 Student Performance Prediction

An end-to-end machine learning project built using **NumPy, Pandas, and PyTorch** to analyze student performance and predict final scores.

The project starts with raw student and academic performance data, performs data cleaning and numerical processing, and then trains a PyTorch neural network to predict a student's final score.

## 🚀 Project Pipeline

```text
Student CSV Data
       ↓
     Pandas
       ↓
Data Cleaning & Analysis
       ↓
     NumPy
       ↓
Numerical Processing
       ↓
 PyTorch Tensors
       ↓
Neural Network
       ↓
Forward Propagation
       ↓
Loss Calculation
       ↓
Autograd & Backpropagation
       ↓
Model Training
       ↓
Final Score Prediction
```

## 🛠️ Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **PyTorch**
* **Git & GitHub**

## 📚 Concepts Covered

### NumPy

* 1D and multidimensional arrays
* Array indexing
* Array slicing
* Arithmetic operations
* Vector operations
* Scalar operations
* Broadcasting
* `np.where()`
* Array aggregation
* NumPy → PyTorch conversion

### Pandas

* Series
* DataFrames
* CSV file importing
* Column and row selection
* Filtering
* Aggregations
* `groupby()`
* `merge()`
* Missing data handling
* Duplicate detection and removal
* Data cleaning
* Feature creation

### PyTorch

* Tensor creation
* Tensor attributes
* Tensor indexing and slicing
* Tensor operations
* CPU/GPU device handling
* `.to(device)`
* Neural network creation
* Linear layers
* ReLU activation
* Forward propagation
* Mean Squared Error loss
* Autograd
* Backward propagation
* Adam optimizer
* Training loops
* Model evaluation
* Prediction with `torch.no_grad()`

## 📂 Project Structure

```text
student-performance-prediction/
│
├── data/
│   ├── students.csv
│   └── performance.csv
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 📊 Dataset

The project uses two CSV files.

### students.csv

Contains basic student information:

* Student ID
* Name
* Department
* Age

### performance.csv

Contains academic information:

* Student ID
* Mathematics
* Physics
* Python
* Attendance

The two datasets are merged using `student_id`.

## 🧹 Data Cleaning

The project intentionally introduces common data-quality problems to demonstrate data cleaning.

The following issues are handled:

* Missing values
* Duplicate records
* Invalid marks outside the 0–100 range

Missing subject scores are filled using the corresponding column mean, duplicate records are removed, and invalid marks are constrained to the valid range.

## 🧮 Feature Engineering

The project calculates:

```text
Final Score =
25% Mathematics
+ 25% Physics
+ 30% Python
+ 20% Attendance
```

Students are also categorized based on their average academic performance:

```text
Average >= 80 → Excellent
Average >= 60 → Good
Average < 60  → Needs Improvement
```

## 🤖 Machine Learning Model

A simple feed-forward neural network is used for prediction.

```text
Input Features
     │
     ├── Math
     ├── Physics
     ├── Python
     └── Attendance
          │
          ▼
     Linear Layer
       4 → 8
          │
          ▼
        ReLU
          │
          ▼
     Linear Layer
       8 → 1
          │
          ▼
   Predicted Final Score
```

The model is trained using:

* **Loss:** Mean Squared Error (MSE)
* **Optimizer:** Adam
* **Learning Rate:** `0.01`
* **Epochs:** `1000`

## 📈 Example Result

After training, the model produced predictions close to the calculated target scores.

Example:

```text
Afsar | Actual: 86.35 | Predicted: 86.37
Rahul | Actual: 72.75 | Predicted: 73.12
Anu   | Actual: 87.10 | Predicted: 87.00
John  | Actual: 60.40 | Predicted: 60.70
```

The final training run achieved:

```text
Mean Absolute Error: 0.2019
```

A new student's score can also be predicted by providing:

```text
Math
Physics
Python
Attendance
```

Example:

```text
Math:       90
Physics:    85
Python:     95
Attendance: 92

Predicted Final Score: 90.35
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/student-performance-prediction.git
cd student-performance-prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

### Git Bash

```bash
source venv/Scripts/activate
```

### Windows CMD

```cmd
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

```bash
python main.py
```

The program will:

1. Load the student datasets.
2. Merge the datasets.
3. Analyze the data using Pandas.
4. Process numerical data using NumPy.
5. Clean missing and invalid data.
6. Convert the data into PyTorch tensors.
7. Train the neural network.
8. Display actual vs predicted scores.
9. Calculate the prediction error.
10. Predict the final score for a new student.

## 🎯 Learning Objective

This project was created to practically combine the concepts learned while studying:

**NumPy → Pandas → PyTorch**

Rather than learning each library independently, the project demonstrates how they work together in a basic machine-learning workflow.

## 🔮 Future Improvements

Possible future versions of the project could include:

* Train/test dataset splitting
* Better model evaluation
* Feature normalization
* Visualization using Matplotlib
* Saving and loading the trained model
* Streamlit web interface
* Larger real-world dataset
* Deployment as a web application

## 👨‍💻 Author

**Afsar**

Computer Science & Engineering

Built as a hands-on learning project for NumPy, Pandas, and PyTorch.
