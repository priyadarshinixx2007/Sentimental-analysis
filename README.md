# Sentimental-analysis
# Spam Classification using Qwen

A simple **AI-based Spam Classification** project built using **Python, Hugging Face Transformers, and the Qwen2.5-0.5B-Instruct model**.

The project uses a pre-trained language model to classify messages into two categories:

* **Spam**
* **Not Spam**

The classification is performed using prompt-based text generation, where the model is instructed to analyze a message and return the appropriate label.

---

## Project Overview

Spam messages are unwanted or potentially harmful messages that may contain advertisements, scams, fake offers, or suspicious links.

This project demonstrates how a Large Language Model (LLM) can be used as a simple spam detection system without training a separate machine-learning classifier.

The model receives a prompt such as:

```text
You are a spam detection system.

Classify this message:

"Congratulations! You've won a $1000 gift card, click here to claim now!"

Possible labels:
1. Spam
2. Not Spam

Answer with ONLY the label.
```

The model then generates a classification such as:

```text
Spam
```

---

## Objectives

* Understand how Hugging Face Transformers can be used with pre-trained models.
* Use an LLM for text classification through prompting.
* Detect spam and non-spam messages.
* Learn how prompt instructions influence model output.
* Understand text-generation pipelines.
* Build a simple AI-based classification system using Python.

---

## Technologies Used

| Technology                | Purpose                                |
| ------------------------- | -------------------------------------- |
| Python                    | Programming language                   |
| Hugging Face Transformers | Accessing the pre-trained model        |
| Qwen2.5-0.5B-Instruct     | Language model used for classification |
| Prompt Engineering        | Providing classification instructions  |
| VS Code                   | Development environment                |

---

## Model Used

This project uses the following pre-trained model:

**Qwen/Qwen2.5-0.5B-Instruct**

The model is loaded using the Hugging Face `pipeline()` function with the `text-generation` task.

The model is instructed through a prompt to classify the input message as either **Spam** or **Not Spam**.

---

## How the Project Works

The complete workflow is:

```text
User Message
     |
     v
Classification Prompt
     |
     v
Qwen2.5-0.5B-Instruct
     |
     v
Text Generation
     |
     v
Extract Generated Response
     |
     v
Spam / Not Spam
```

### Step 1: Import Pipeline

The Hugging Face Transformers pipeline is imported:

```python
from transformers import pipeline
```

### Step 2: Load the Model

The Qwen model is loaded using:

```python
classifier = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct"
)
```

The `text-generation` pipeline allows the model to generate a response based on the given prompt.

### Step 3: Create the Classification Function

The function:

```python
def spam_classification(text):
```

takes a message as input.

### Step 4: Create the Prompt

The message is inserted into a prompt that tells the model what task to perform.

The model is given two possible labels:

```text
1. Spam
2. Not Spam
```

It is also instructed to answer using only the label.

This helps reduce unnecessary generated text.

### Step 5: Generate the Prediction

The model processes the prompt:

```python
output = classifier(
    prompt,
    max_new_tokens=5,
    do_sample=False
)
```

**Parameters:**

* `max_new_tokens=5`: Limits the number of new tokens generated.
* `do_sample=False`: Disables random sampling to make generation more deterministic.

### Step 6: Extract the Generated Text

The generated response is obtained using:

```python
response = output[0]["generated_text"]
```

The original prompt is removed:

```python
generated = response[len(prompt):].strip()
```

This leaves the model's generated classification.

### Step 7: Display the Result

The program prints the message and its predicted label:

```text
--------------------------------
Text  : <message>
Label : <classification>
--------------------------------
```

---

## Test Cases

The project contains three test messages.

### Test Case 1: Spam

**Input:**

```text
Congratulations! You've won a $1000 gift card, click here to claim now!
```

Expected classification:

```text
Spam
```

### Test Case 2: Not Spam

**Input:**

```text
Hey, are we still meeting for lunch tomorrow?
```

Expected classification:

```text
Not Spam
```

### Test Case 3: Spam

**Input:**

```text
Limited time offer: buy one get one free on all products!
```

Expected classification:

```text
Spam
```

---

## Project Structure

```text
spam-classification/
|
|-- spam_classifier.py
|-- requirements.txt
|-- README.md
```

### File Description

| File                 | Description                  |
| -------------------- | ---------------------------- |
| `spam_classifier.py` | Main Python program          |
| `requirements.txt`   | Required Python dependencies |
| `README.md`          | Project documentation        |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/spam-classification.git
```

### 2. Navigate to the Project

```bash
cd spam-classification
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

Create a `requirements.txt` file containing:

```text
transformers
torch
```

---

## Run the Project

Run the Python file:

```bash
python spam_classifier.py
```

The model will be downloaded from Hugging Face the first time it is executed.

---

## Sample Output

The program produces output in the following format:

```text
--------------------------------
Text  : Congratulations! You've won a $1000 gift card, click here to claim now!
Label : Spam
--------------------------------

--------------------------------
Text  : Hey, are we still meeting for lunch tomorrow?
Label : Not Spam
--------------------------------

--------------------------------
Text  : Limited time offer: buy one get one free on all products!
Label : Spam
--------------------------------
```

The exact generated output may vary depending on the model's response.

---

## Key Concepts

### 1. Hugging Face Transformers

Transformers is a library for working with pre-trained transformer-based models.

In this project, it is used to load and run the Qwen model.

### 2. Large Language Models

A Large Language Model (LLM) is an AI model trained to understand and generate human language.

Here, the LLM performs classification through instructions rather than traditional supervised training.

### 3. Prompt Engineering

The project uses a carefully written prompt to specify:

* The task the model should perform.
* The message to classify.
* The permitted labels.
* The expected answer format.

### 4. Zero-Shot Classification

The model is not given labeled examples in the prompt.

Instead, it receives a task description and classifies the message directly. This is an example of zero-shot prompting.

### 5. Text Generation

Although the task is classification, the project uses a text-generation pipeline. The model generates the classification label as text.

---

## Features

* AI-powered spam detection.
* Classification of text messages.
* Pre-trained language model integration.
* Hugging Face Transformers integration.
* Prompt-based classification.
* Support for two classification labels.
* Simple Python implementation.
* No separate model training required.

---

## Limitations

This project uses an instruction-following LLM for classification, so it may not always produce the expected label.

Possible limitations include:

* The model may generate additional text despite the instruction.
* Sarcastic or ambiguous messages may be difficult to classify.
* Marketing messages are not always spam.
* Classification accuracy depends on the model's interpretation of the prompt.
* The model was not specifically fine-tuned for spam detection.
* The generated response should be validated before being treated as a final label.

For a production-level spam detection system, a dedicated classification model or a fine-tuned model could be used.

---

## Future Improvements

The project can be extended with:

* Accuracy evaluation using a labeled dataset.
* CSV file input.
* Streamlit web interface.
* Email spam detection.
* SMS spam detection.
* Classification statistics.
* Fine-tuning on a spam dataset.
* Confidence or probability analysis.
* Batch message classification.
* Database storage for classified messages.

---

## Learning Outcomes

By completing this project, I learned:

* How to use the Hugging Face Transformers library.
* How to load a pre-trained Qwen model.
* How to use a text-generation pipeline.
* How to design prompts for classification.
* How zero-shot prompting works.
* How to extract generated model responses.
* How LLMs can be used for practical text classification tasks.

---

## Project Information

**Domain:** Artificial Intelligence / Natural Language Processing

**Project Level:** Beginner

**Technique:** Zero-Shot Prompt-Based Classification

**Model:** Qwen2.5-0.5B-Instruct

**Task:** Spam / Not Spam Classification

---

## Author

**Priyadarshini D**

B.Sc. Computer Science with AI

---

## Conclusion

This project demonstrates a simple approach to spam message classification using a Large Language Model.

Instead of training a machine-learning model from scratch, the project uses Qwen2.5-0.5B-Instruct with a carefully designed prompt to classify messages as Spam or Not Spam.

It provides a practical introduction to Hugging Face, LLMs, prompt engineering, zero-shot classification, and NLP applications.
