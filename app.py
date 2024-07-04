from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import spacy
from collections import Counter
import random
from PyPDF2 import PdfReader
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)
Bootstrap(app)

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Load the T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Function to generate MCQs using the T5 model
def generate_mcqs_t5(text, num_questions=5):
    if not text:
        return []

    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    num_questions = min(num_questions, len(sentences))
    selected_sentences = random.sample(sentences, num_questions)
    mcqs = []

    for sentence in selected_sentences:
        sent_doc = nlp(sentence)
        nouns = [token.text.lower() for token in sent_doc if token.pos_ == "NOUN"]
        if len(nouns) < 2:
            continue

        noun_counts = Counter(nouns)
        if noun_counts:
            subject = noun_counts.most_common(1)[0][0]
            question_stem = sentence.replace(subject, "______")

            answer_choices = [subject]
            distractors = list(set(noun for noun in nouns if noun != subject))

            # Ensure we have at least 3 distractors
            if len(distractors) < 3:
                additional_distractors = generate_additional_distractors(doc, distractors, num_required=3 - len(distractors))
                distractors.extend(additional_distractors)

            # Shuffle distractors
            random.shuffle(distractors)

            # Add distractors to answer choices
            answer_choices.extend(distractors[:3])

            # Shuffle answer choices
            random.shuffle(answer_choices)

            mcqs.append({
                'question_stem': question_stem,
                'answer_choices': answer_choices,
                'correct_answer': subject
            })

    return mcqs

def generate_additional_distractors(doc, existing_distractors, num_required):
    additional_distractors = []
    adj_tokens = [token.text.lower() for token in doc if token.pos_ == "ADJ" and token.text.lower() not in existing_distractors]
    noun_tokens = [token.text.lower() for token in doc if token.pos_ == "NOUN" and token.text.lower() not in existing_distractors]

    additional_distractors.extend(random.sample(adj_tokens, min(num_required, len(adj_tokens))))
    additional_distractors.extend(random.sample(noun_tokens, min(num_required, len(noun_tokens))))

    return additional_distractors

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = ""

        # Check if files were uploaded
        if 'files[]' in request.files:
            files = request.files.getlist('files[]')
            for file in files:
                if file.filename.endswith('.pdf'):
                    # Process PDF file
                    text += process_pdf(file)
                elif file.filename.endswith('.txt'):
                    # Process text file
                    text += file.read().decode('utf-8')
        else:
            # Process manual input
            text = request.form['text']

        # Get the selected number of questions from the input field
        num_questions = int(request.form.get('num_questions', 5))

        mcqs = generate_mcqs_t5(text, num_questions=num_questions)
        mcqs_with_index = [(i + 1, mcq) for i, mcq in enumerate(mcqs)]
        return render_template('mcqs.html', mcqs=mcqs_with_index)

    return render_template('index.html')

def process_pdf(file):
    text = ""

    try:
        pdf_reader = PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_num].extract_text()
            if page_text:
                text += page_text
    except Exception as e:
        print(f"Error processing PDF: {e}")

    return text

if __name__ == '__main__':
    app.run(debug=True)
