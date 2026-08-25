import pandas as pd
import numpy as np
import torch
import torch.nn as nn


students = pd.read_csv("data/students.csv")
performance = pd.read_csv("data/performance.csv")

df = pd.merge(students, performance, on="student_id")

print("\nCombined Dataset:")
print(df)

print("\nStudent Names:")
print(df["name"])

print("\nSelected Columns:")
print(df[["name", "department", "python"]])

print("\nStudents with Python score > 80:")
print(df[df["python"] > 80])

print("\nStudents with attendance > 90:")
print(df[df["attendance"] > 90])

print("\nAverage Math:", df["math"].mean())
print("Average Physics:", df["physics"].mean())
print("Average Python:", df["python"].mean())

print("\nHighest Python Score:", df["python"].max())
print("Lowest Python Score:", df["python"].min())

print("\nDepartment-wise average:")
print(df.groupby("department")[["math", "physics", "python"]].mean())

marks = df[["math", "physics", "python"]].to_numpy()

print("\nMarks Array:")
print(marks)

print("Shape:", marks.shape)

print("\nFirst student marks:")
print(marks[0])

print("\nPython marks:")
print(marks[:, 2])

print("\nFirst five students:")
print(marks[:5])

total_marks = marks.sum(axis=1)

print("\nTotal marks:")
print(total_marks)

bonus_marks = marks + 5

print("\nMarks after bonus:")
print(bonus_marks)
average_marks = marks.mean(axis=1)
print(average_marks)

performance_category = np.where(
    marks.mean(axis=1) >= 80,
    "Excellent",
    "Needs Improvement"
)

print("\nPerformance category:")
print(performance_category)

performance_category = np.where(
    average_marks >= 80,
    "Excellent",
    np.where(
        average_marks >= 60,
        "Good",
        "Needs Improvement"
    )
)

df["performance_category"] = performance_category

print(df[["name", "department", "python", "performance_category"]])

df["final_score"] = (
    df["math"] * 0.25 +
    df["physics"] * 0.25 +
    df["python"] * 0.30 +
    df["attendance"] * 0.20
)

print(df[[
    "name",
    "math",
    "physics",
    "python",
    "attendance",
    "final_score"
]])

# Create some messy data for cleaning practice

df.loc[2, "python"] = np.nan
df.loc[5, "physics"] = np.nan

df = pd.concat([df, df.iloc[[3]]], ignore_index=True)

df.loc[6, "math"] = 105

print("\nMessy Dataset:")
print(df)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nInvalid Math scores:")
print(df[(df["math"] < 0) | (df["math"] > 100)])

df["math"] = df["math"].fillna(df["math"].mean())
df["physics"] = df["physics"].fillna(df["physics"].mean())
df["python"] = df["python"].fillna(df["python"].mean())

df = df.drop_duplicates()

invalid_math = (df["math"] < 0) | (df["math"] > 100)

print("\nInvalid math records:")
print(df[invalid_math])

df["math"] = df["math"].clip(0, 100)
df["physics"] = df["physics"].clip(0, 100)
df["python"] = df["python"].clip(0, 100)
df["attendance"] = df["attendance"].clip(0, 100)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicates after cleaning:")
print(df.duplicated().sum())

print("\nCleaned dataset:")
print(df)

marks = df[["math", "physics", "python"]].to_numpy()

average_marks = marks.mean(axis=1)

performance_category = np.where(
    average_marks >= 80,
    "Excellent",
    np.where(
        average_marks >= 60,
        "Good",
        "Needs Improvement"
    )
)

df["performance_category"] = performance_category

df["final_score"] = (
    df["math"] * 0.25 +
    df["physics"] * 0.25 +
    df["python"] * 0.30 +
    df["attendance"] * 0.20
)

print("\nFinal cleaned dataset:")
print(df[[
    "student_id",
    "name",
    "department",
    "math",
    "physics",
    "python",
    "attendance",
    "final_score",
    "performance_category"
]])

X = df[["math", "physics", "python", "attendance"]].to_numpy()
y = df["final_score"].to_numpy()

print("\nFeatures X:")
print(X)

print("\nTarget y:")
print(y)

print("\nX shape:", X.shape)
print("y shape:", y.shape)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

print("\nX Tensor:")
print(X_tensor)

print("\ny Tensor:")
print(y_tensor)

print("\nTensor attributes:")
print("X shape:", X_tensor.shape)
print("X dtype:", X_tensor.dtype)
print("X device:", X_tensor.device)

print("y shape:", y_tensor.shape)
print("y dtype:", y_tensor.dtype)
print("y device:", y_tensor.device)

print("\nTensor operations:")

print("First student:")
print(X_tensor[0])

print("\nAll Python scores:")
print(X_tensor[:, 2])

print("\nFirst five students:")
print(X_tensor[:5])

print("\nTensor operations:")

print("First student:")
print(X_tensor[0])

print("\nAll Python scores:")
print(X_tensor[:, 2])

print("\nFirst five students:")
print(X_tensor[:5])

average_tensor = X_tensor[:, :3].mean(dim=1)

print("\nAverage academic score:")
print(average_tensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nUsing device:", device)

X_tensor = X_tensor.to(device)
y_tensor = y_tensor.to(device)

print("X device:", X_tensor.device)
print("y device:", y_tensor.device)

class StudentPerformanceModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(4, 8)
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)

        return x
model = StudentPerformanceModel()

model = model.to(device)

print("\nModel:")
print(model)

predictions = model(X_tensor)

print("\nPredictions:")
print(predictions)

loss_function = nn.MSELoss()

loss = loss_function(
    predictions.squeeze(),
    y_tensor
)

print("\nLoss:")
print(loss)

# loss.backward()

# print("\nGradients:")

# for name, parameter in model.named_parameters():

#     print(name)
#     print(parameter.grad)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 1000

for epoch in range(epochs):

    # Forward propagation
    predictions = model(X_tensor)

    # Calculate loss
    loss = loss_function(
        predictions.squeeze(),
        y_tensor
    )

    # Clear old gradients
    optimizer.zero_grad()

    # Backward propagation
    loss.backward()

    # Update weights
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {loss.item():.4f}"
        )

model.eval()

with torch.no_grad():
    predictions = model(X_tensor)

predictions = predictions.squeeze()

print("\nModel Predictions:")
for i in range(len(df)):
    print(
        f"{df.iloc[i]['name']:>8} | "
        f"Actual: {y_tensor[i].item():6.2f} | "
        f"Predicted: {predictions[i].item():6.2f}"
    )
mae = torch.mean(torch.abs(predictions - y_tensor))

print(f"\nMean Absolute Error: {mae.item():.4f}")

def predict_student(math, physics, python, attendance):

    student_data = torch.tensor(
        [[math, physics, python, attendance]],
        dtype=torch.float32
    ).to(device)

    model.eval()

    with torch.no_grad():
        prediction = model(student_data)

    return prediction.item()

new_score = predict_student(
    math=90,
    physics=85,
    python=95,
    attendance=92
)

print("\nNew Student Prediction:")
print(f"Predicted Final Score: {new_score:.2f}")