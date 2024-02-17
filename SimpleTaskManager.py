import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random

# Initialize an empty task DataFrame
task_data = pd.DataFrame(columns=['description', 'priority'])

# Load pre-existing tasks from a CSV file (if any)
try:
    task_data = pd.read_csv('tasks.csv')
except FileNotFoundError:
    pass

# Function to save tasks to a CSV file
def save_tasks():
    task_data.to_csv('tasks.csv', index=False)

# Train the task priority classifier
vectorizer = CountVectorizer()
classifier = MultinomialNB()
priority_model = make_pipeline(vectorizer, classifier)
priority_model.fit(task_data['description'], task_data['priority'])

# Function to add a task to the list
def add_new_task(description, priority):
    global task_data  # Declare task_data as a global variable
    new_task = pd.DataFrame({'description': [description], 'priority': [priority]})
    task_data = pd.concat([task_data, new_task], ignore_index=True)
    save_tasks()

# Function to remove a task by description
def remove_existing_task(description):
    task_data = task_data[task_data['description'] != description]
    save_tasks()

# Function to list all tasks
def display_tasks():
    if task_data.empty:
        print("No tasks available.")
    else:
        print(task_data)

# Function to recommend a task based on machine learning
def suggest_task():
    if not task_data.empty:
        # Get high-priority tasks
        high_priority_tasks = task_data[task_data['priority'] == 'High']
        
        if not high_priority_tasks.empty:
            # Choose a random high-priority task
            random_task = random.choice(high_priority_tasks['description'])
            print(f"Recommended task: {random_task} - Priority: High")
        else:
            print("No high-priority tasks available for recommendation.")
    else:
        print("No tasks available for recommendations.")

# Main menu
while True:
    print("\nTask Management Application")
    print("1. Add New Task")
    print("2. Remove Existing Task")
    print("3. Display Tasks")
    print("4. Recommend Task")
    print("5. Exit")

    choice = input("Select an option: ")

    if choice == "1":
        task_description = input("Enter task description: ")
        task_priority = input("Enter task priority (Low/Medium/High): ").capitalize()
        add_new_task(task_description, task_priority)
        print("Task added successfully.")

    elif choice == "2":
        task_description = input("Enter task description to remove: ")
        remove_existing_task(task_description)
        print("Task removed successfully.")

    elif choice == "3":
        display_tasks()

    elif choice == "4":
        suggest_task()

    elif choice == "5":
        print("Goodbye!")
        break

    else:
        print("Invalid option. Please select a valid option.")
