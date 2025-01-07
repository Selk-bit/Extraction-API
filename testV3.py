import google.generativeai as genai
from fastapi.responses import JSONResponse, FileResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from translation import Translator
from ftfy import fix_text
import pdfplumber
import textract
import json
import os
import re
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from collections import Counter
from datetime import datetime, timezone
import threading
from statistics import mean
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import fitz
from collections import Counter
import mimetypes
import random
import copy
import asyncio
import subprocess
from pathlib import Path
import tempfile
import shutil
from fastapi import UploadFile
import io


upload_lock = threading.Lock()


# Load the environment variables
load_dotenv()

pre_defined_headers = [
    # English
    "education", "educational background", "academic background", "work experience", "professional experience", "employment history",
    "job history", "experience", "experiences", "skills", "skill set", "competences", "competencies", "abilities",
    "technical skills", "professional skills", "soft skills", "languages", "language proficiency", "projects", "personal projects",
    "certifications", "certificates", "credentials", "awards", "achievements", "honors", "recognitions", "accomplishments",
    "references", "referees", "interests", "hobbies", "activities", "extracurricular activities", "personal interests",
    "summary", "profile", "objective", "professional summary", "personal statement", "career objective",
    "volunteer experience", "volunteering", "volunteer work", "training", "workshops", "courses", "publications",
    "presentations", "affiliations", "memberships", "professional memberships",

    # French
    "éducation", "formation", "formation académique", "expérience professionnelle", "expériences professionnelles",
    "historique professionnel", "expériences", "compétences", "compétences techniques", "compétences professionnelles",
    "compétences interpersonnelles", "capacités", "aptitudes", "langues", "maîtrise des langues", "projets",
    "projets personnels", "certifications", "certificats", "accréditations", "récompenses", "distinctions",
    "honneurs", "réalisations", "références", "référents", "centres d'intérêt", "intérêts", "loisirs",
    "activités", "activités extra-scolaires", "profil", "objectif", "résumé", "déclaration personnelle",
    "objectif de carrière", "bénévolat", "volontariat", "travail bénévole", "formations", "ateliers", "cours",
    "publications", "présentations", "affiliations", "adhésions", "membres professionnels",

    # Spanish
    "educación", "formación", "historial académico", "experiencia laboral", "experiencia profesional", "experiencias laborales",
    "experiencias", "habilidades", "conocimientos", "competencias", "competencias técnicas", "competencias profesionales",
    "habilidades interpersonales", "idiomas", "dominio de idiomas", "proyectos", "proyectos personales", "certificaciones",
    "certificados", "credenciales", "premios", "logros", "honores", "reconocimientos", "referencias", "intereses",
    "actividades", "actividades extracurriculares", "perfil", "objetivo", "resumen profesional", "declaración personal",
    "objetivo profesional", "voluntariado", "experiencia de voluntariado", "entrenamiento", "cursos", "talleres",
    "publicaciones", "presentaciones", "afiliaciones", "membresías", "membresías profesionales",

    # German
    "bildung", "ausbildung", "akademischer hintergrund", "berufserfahrung", "berufliche erfahrung", "beruflicher werdegang",
    "arbeitserfahrung", "erfahrungen", "fähigkeiten", "kompetenzen", "kompetenzen technische", "berufliche kompetenzen",
    "soft skills", "sprachen", "sprachkenntnisse", "projekte", "persönliche projekte", "zertifizierungen", "zertifikate",
    "akkreditierungen", "auszeichnungen", "preise", "anerkennungen", "ehren", "leistungen", "referenzen", "interessen",
    "hobbys", "aktivitäten", "extrakurrikulare aktivitäten", "profil", "zielsetzung", "berufsziel", "persönliches profil",
    "freiwilligenarbeit", "ehrenamtliche arbeit", "praktika", "kurse", "fortbildungen", "publikationen", "präsentationen",
    "mitgliedschaften", "berufliche mitgliedschaften"
]

# Convert all headers to lowercase for case-insensitive matching
pre_defined_headers = [header.lower() for header in pre_defined_headers]

class salim:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.chain = self.get_conversational_chain()
        self.model = self.get_conversational_model()
        self.translator = Translator()
        self.report_file = "report.json"
        self.lock = threading.Lock()
        if not os.path.exists(self.report_file):
            with open(self.report_file, "w") as f:
                json.dump([], f)


    def count_tokens(self, text):
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        count_response = model.count_tokens(text)
        return count_response.total_tokens
        # return len(self.encoder.encode(text))


    def calculate_cost(self, tokens, is_input):
        if tokens <= 128000:
            if is_input:
                rate = 0.075 / 1_000_000
            else:
                rate = 0.30 / 1_000_000
        else:
            if is_input:
                rate = 0.15 / 1_000_000
            else:
                rate = 0.60 / 1_000_000
        return tokens * rate


    def calculate_average_word_gap(self, words):
        """
        Calculate the average horizontal space between consecutive words on the same line.
        """
        line_gaps = []
        current_line_y = None
        line_words = []

        # Group words by their vertical position (y0 and y1)
        for word in words:
            if current_line_y is None or abs(word['top'] - current_line_y) < 5:  # Adjust threshold for same line
                line_words.append(word)
                current_line_y = word['top']
            else:
                # Calculate gaps between words on the same line
                for i in range(1, len(line_words)):
                    gap = line_words[i]['x0'] - line_words[i-1]['x1']
                    if gap > 0:
                        line_gaps.append(gap)
                # Reset for next line
                line_words = [word]
                current_line_y = word['top']

        # Handle the last line
        for i in range(1, len(line_words)):
            gap = line_words[i]['x0'] - line_words[i-1]['x1']
            if gap > 0:
                line_gaps.append(gap)

        # Calculate average gap, fallback to a default if no gaps found
        return mean(line_gaps) if line_gaps else 0

    def detect_maximum_left_gap_persistence(self, words, page_width, average_gap, gap_multiplier=3, persistence_threshold=0.5):
        """
        Detects if there is a large gap in the left side that is significantly larger than the average word gap
        and if this gap persists across multiple lines.
        """
        left_column_words = [word for word in words if word['x0'] < page_width / 2]
        max_left_gaps = []
        current_line_y = None
        line_words = []

        # Group words by lines and calculate max gaps on the left side
        for word in left_column_words:
            if current_line_y is None or abs(word['top'] - current_line_y) < 5:
                line_words.append(word)
                current_line_y = word['top']
            else:
                # Calculate the largest gap on the left side
                for i in range(1, len(line_words)):
                    gap = line_words[i]['x0'] - line_words[i-1]['x1']
                    if gap > 0:
                        max_left_gaps.append(gap)

                # Reset for the next line
                line_words = [word]
                current_line_y = word['top']

        # Handle the last line
        for i in range(1, len(line_words)):
            gap = line_words[i]['x0'] - line_words[i-1]['x1']
            if gap > 0:
                max_left_gaps.append(gap)

        # Check if the maximum gap on the left side persists across a significant number of lines
        significant_gaps = [gap for gap in max_left_gaps if gap > average_gap * gap_multiplier]
        if len(significant_gaps) / len(max_left_gaps) > persistence_threshold:
            return True
        return False

    # def get_pdf_text(self, pdf_path):
    #     text = ""
    #     with pdfplumber.open(pdf_path) as pdf:
    #         for page in pdf.pages:
    #             words = page.extract_words()

    #             # Calculate the average word gap on the page
    #             average_gap = self.calculate_average_word_gap(words)

    #             # Detect if there is a significant and persistent left-side gap
    #             if self.detect_maximum_left_gap_persistence(words, page.width, average_gap):
    #                 # If a significant left-side gap is detected, split the page into left and right parts
    #                 left_column_boundary = max(word['x1'] for word in words if word['x0'] < page.width / 2)

    #                 left_bbox = (0, 0, left_column_boundary, page.height)
    #                 right_bbox = (left_column_boundary, 0, page.width, page.height)

    #                 # Extract text from the left part first
    #                 left_text = page.within_bbox(left_bbox).extract_text()
    #                 if left_text:
    #                     text += left_text + "\n"

    #                 # Extract text from the right part next
    #                 right_text = page.within_bbox(right_bbox).extract_text()
    #                 if right_text:
    #                     text += right_text + "\n"
    #             else:
    #                 # If no significant gap is detected, extract the whole page normally
    #                 single_column_text = page.extract_text()
    #                 if single_column_text:
    #                     text += single_column_text + "\n"

    #             # Extract tables if any, and append them after processing text
    #             tables = page.extract_tables()
    #             for table in tables:
    #                 for row in table:
    #                     filtered_row = [element for element in row if element is not None]
    #                     text += " | ".join(filtered_row) + "\n"
                        
    #     return text

    def extract_text_spans(self, pdf_path):
        doc = fitz.open(pdf_path)
        all_spans = []
        cumulative_height = 0  # Track cumulative height as we go through pages

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_width = page.rect.width
            page_height = page.rect.height
            blocks = page.get_text("dict")['blocks']

            for block in blocks:
                for line in block.get('lines', []):
                    spans = line.get('spans', [])
                    if not spans:
                        continue
                    for span in spans:
                        text = span['text'].strip().lower()  # Convert to lowercase for matching
                        font_size = span['size']
                        font_color = span['color']
                        bbox = span['bbox']

                        # Adjust y0 and y1 to include cumulative height from previous pages
                        adjusted_y0 = bbox[1] + cumulative_height
                        adjusted_y1 = bbox[3] + cumulative_height

                        # Add spans to the list with the adjusted y positions
                        all_spans.append({
                            'text': text,
                            'original_text': span['text'],  # Preserve the original case-sensitive text
                            'font_size': font_size,
                            'font_color': font_color,
                            'bbox': bbox,
                            'page_num': page_num,
                            'x0': bbox[0],
                            'y0': adjusted_y0,  # Use adjusted y0
                            'x1': bbox[2],
                            'y1': adjusted_y1,  # Use adjusted y1
                            'page_width': page_width,
                            'page_height': page_height,
                        })

            # Add the current page's height to the cumulative height for the next page
            cumulative_height += page_height

        return all_spans


    def remove_special_characters(self, text):
        # This pattern retains letters, accented characters, numbers, and whitespace
        cleaned_text = re.sub(r'[^\w\sÀ-ÿ]', '', text)
        return cleaned_text


    # Function to detect the largest header from the predefined list and find its properties
    def detect_largest_header_and_properties(self, spans):
        largest_header_span = None

        # Step 1: Identify spans that match predefined headers
        for span in spans:
            if self.remove_special_characters(span['text']).strip().lower() in pre_defined_headers:
                if largest_header_span is None or span['font_size'] > largest_header_span['font_size']:
                    largest_header_span = span

        if not largest_header_span:
            return None  # No headers found in predefined list

        return largest_header_span


    # Function to detect other headers based on header font properties
    def detect_other_headers(self, spans, header_font_properties, tolerance=0.05):
        header_font_size, header_font_color = header_font_properties
        headers = []

        # Detect all headers with the same font properties
        for span in spans:
            if (abs(span['font_size'] - header_font_size) <= tolerance):
                headers.append(span)

        return headers


    # Function to adjust the header x position based on the smallest content x0 found below it
    def adjust_header_x_position(self, spans, headers, tolerance=5):
        adjusted_headers = []
        for header in headers:
            smallest_x0 = header['x0']  # Initialize with header's x0

            # Look for the content under the header
            for span in spans:
                if span['y0'] > header['y1'] and span['x0'] <= header['x0'] and span['x1'] >= header['x0']:
                    # Find the smallest x0 in the content under the header
                    if span['x0'] <= smallest_x0:
                        smallest_x0 = span['x0']

            # Update the header's x0 to the smallest found value
            header['x0'] = smallest_x0
            adjusted_headers.append(header)

        return adjusted_headers


    # Function to group headers into columns based on adjusted x-positions
    def group_headers_by_columns(self, headers, gap_threshold=50):
        """
        This function groups headers into columns by analyzing the x-positions and identifying natural breaks.
        The gap_threshold defines the minimum gap in x-positions that separates two columns.
        """
        headers.sort(key=lambda h: h['x0'])  # Sort headers by x0 (leftmost position)

        first_column_headers = []
        second_column_headers = []

        # Identify the largest gap in x-positions to separate the two columns
        x_positions = [header['x0'] for header in headers]
        max_gap = 0
        break_index = 0

        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i - 1]
            if gap > max_gap and gap > gap_threshold:
                max_gap = gap
                break_index = i

        # Assign headers based on the break
        first_column_headers = headers[:break_index]
        second_column_headers = headers[break_index:]
        return first_column_headers, second_column_headers


    def extract_columns_from_pdf(self, pdf_file, second_column_x0, tolerance=3, vertical_tolerance=1, space_threshold=2,
                                space_diff_threshold=50):
        first_column_text = []
        second_column_text = []

        fitz_text_total = ""
        pdfplumber_text_total = ""

        # Extract text using both pdfplumber and fitz
        with pdfplumber.open(pdf_file) as pdf:
            cumulative_height = 0  # Track cumulative height across pages
            doc_fitz = fitz.open(pdf_file)  # Open the same file with fitz

            for page_num, page in enumerate(pdf.pages):
                text_objects = page.extract_words()  # Extract text along with coordinates
                fitz_page_text = doc_fitz[page_num].get_text("text")  # Get text using fitz
                fitz_text_total += fitz_page_text  # Collect fitz text for the entire doc

                # Collect pdfplumber text and coordinates
                page_spans = []
                for text_obj in text_objects:
                    pdfplumber_text_total += " " + text_obj['text']
                    page_spans.append({
                        'text': text_obj['text'],
                        'x0': text_obj['x0'],
                        'x1': text_obj['x1'],
                        'y0': text_obj['top'] + cumulative_height,  # Adjusted for cumulative height
                        'y1': text_obj['bottom'] + cumulative_height,
                        'page_num': page_num,
                        'page_width': page.width,
                        'page_height': page.height
                    })

                # Add the current page's height to the cumulative height for the next page
                cumulative_height += page.height

        # Compare the number of spaces in pdfplumber and fitz texts
        num_spaces_pdfplumber = pdfplumber_text_total.count(' ')
        num_spaces_fitz = fitz_text_total.count(' ')
        # Decide which method to use based on the space difference
        use_fitz = abs(num_spaces_pdfplumber - num_spaces_fitz) > space_diff_threshold

        if use_fitz:
            print(f"Using fitz's text due to space difference: {num_spaces_pdfplumber} vs {num_spaces_fitz}")
            spans = self.extract_text_spans(pdf_file)  # Use fitz spans for further processing
        else:
            print(f"Using pdfplumber's text as space difference is minimal: {num_spaces_pdfplumber} vs {num_spaces_fitz}")
            spans = page_spans  # Use pdfplumber spans for further processing

        # Sort spans by y and then by x to properly order text content
        def sort_by_y_then_x(span):
            return (span['y0'], span['x0'])

        spans.sort(key=sort_by_y_then_x)

        # Helper function to intelligently concatenate text with inferred spaces
        def concatenate_text(current_span, next_span, text_list):
            # Add the current text
            text_list.append(current_span['text'])

            if next_span:
                # Check if the next span is on a new line (y0 difference larger than tolerance)
                if abs(next_span['y0'] - current_span['y0']) > vertical_tolerance:
                    text_list.append("\n")  # Newline if next span is on a different row
                else:
                    # Infer space between words based on horizontal gap
                    gap_between_spans = next_span['x0'] - current_span['x1']
                    if gap_between_spans > space_threshold:
                        text_list.append(" ")  # Add a space if gap is larger than the threshold

        # Process each span to split text into first and second columns
        for idx, span in enumerate(spans):
            # Get the next span for comparison
            next_span = spans[idx + 1] if idx + 1 < len(spans) else None

            if span['x0'] < second_column_x0 - tolerance:
                # Add text to first column
                concatenate_text(span, next_span, first_column_text)
            elif span['x0'] >= second_column_x0 - tolerance:
                # Add text to second column
                concatenate_text(span, next_span, second_column_text)

        # Join text for each column and return the result
        first_column_content = "".join(first_column_text).strip()
        second_column_content = "".join(second_column_text).strip()

        return first_column_content, second_column_content


    def get_pdf_text_v2(self, pdf_path):
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        filtered_row = [element for element in row if element is not None]
                        text += " | ".join(filtered_row) + "\n"
        return text


    def get_pdf_text(self, pdf_path):
        all_spans = self.extract_text_spans(pdf_path)

        # Detect the largest header from the predefined list and its font properties
        largest_header_span = self.detect_largest_header_and_properties(all_spans)

        if not largest_header_span:
            print("No headers detected from the predefined list.")
            return

        # Extract header font properties
        header_font_properties = (
            largest_header_span['font_size'], largest_header_span['font_color'])

        # Detect other headers with the same font properties
        headers = self.detect_other_headers(all_spans, header_font_properties)

        # Adjust headers' x positions based on the content under them
        adjusted_headers = self.adjust_header_x_position(all_spans, headers)

        # Group headers into columns by identifying natural breaks in x-positions
        first_column_headers, second_column_headers = self.group_headers_by_columns(adjusted_headers)

        # Get the x0 of the second column
        second_column_x0 = min(header['x0'] for header in second_column_headers) if second_column_headers else None

        # Extract the content of the first and second columns
        first_column_content, second_column_content = self.extract_columns_from_pdf(pdf_path, second_column_x0)

        # Print the first and second column content
        return f"{first_column_content}\n{second_column_content}"



    def get_docx_text(self, docx_path):
        text = textract.process(docx_path, method='docx').decode('utf-8')
        return text

    # Function to load the conversational chain
    def get_conversational_chain(self):
        genai.configure(api_key=self.api_key)

        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=8192)

        prompt_template = """
        Answer the request as detailed as possible from the provided context.
        if the answer is not in the provided context just give an empty string "", don't provide the wrong answer nor say that the answer is not provided in the context, 
        and do not add any extra characters other than the ones demanded. Make 1sure the answer is provided in the way that it is asked in the question.
	Use as much processing power as you can to answer fast.
        Context :\n {context}?\n
        Question: \n {question}\n
        Request:
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain


    def get_conversational_model(self):
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        return model


    def merge_jobs(self, jobs):
        # Create a dictionary to store unique jobs
        unique_jobs = {}

        # Iterate over each job
        for job in jobs:
            # Get the job's key based on job_title and company_name
            key = (job["job_title"], job["company_name"])

            # Check if the job has empty start_date and end_date
            if not job["start_date"] and not job["end_date"]:
                # If there is already a job with the same key in unique_jobs, concatenate responsibilities
                if key in unique_jobs:
                    unique_jobs[key]["responsibilities"] += "<br>" + job["responsibilities"]
                # If no matching job exists, store the job as is
            else:
                # Store the job in unique_jobs if not already present
                if key not in unique_jobs:
                    unique_jobs[key] = job
                else:
                    # If a matching job exists, concatenate responsibilities
                    unique_jobs[key]["responsibilities"] += "<br>" + job["responsibilities"]

        # Convert the unique_jobs dictionary back to a list
        merged_jobs = list(unique_jobs.values())
        return merged_jobs


    def convert_to_html(self, text):
        # Check if there are no #start# and #end# tags in the text
        if "#start#" not in text and "#end#" not in text:
            return f"<p>{text.strip()}</p>"

        html_output = []
        buffer = []
        in_list = False
        text = re.sub(r"(?<=#end#)[^a-zA-Z]+(?=#start#)", "", text)
        # Split by delimiter while keeping #start# and #end# as separate elements
        segments = re.split(r"(#start#|#end#)", text)

        i = 0
        while i < len(segments):
            segment = segments[i].strip()

            if segment == "#start#":
                # Open a new <ul> if not already inside one
                if not in_list:
                    if buffer and any(s.strip() for s in buffer) and any(s.isalpha() for s in buffer):
                        # Add remaining buffered text as <p> if non-empty
                        html_output.append(f"<p>{' '.join(buffer).strip()}</p>")
                        buffer = []
                    html_output.append("<ul>")
                    in_list = True
                buffer = []
                i += 1

            elif segment == "#end#":
                # Add the buffered content as an <li>
                if buffer:
                    li = f"<li>{' '.join(buffer).strip()}"
                    li = f"{li}</li>" if li.endswith('.') else f"{li}.</li>"
                    html_output.append(li)
                    buffer = []

                # Check if there is no immediate #start# after this #end#
                remaining_text = ''.join(segments[i + 1:]).strip()  # Get all remaining text after current #end#
                if in_list and not remaining_text.startswith("#start#"):
                    html_output.append("</ul>")
                    in_list = False
                i += 1

            else:
                # Regular content, add to buffer
                buffer.append(segment)
                i += 1

        # Wrap any remaining non-list content in <p>
        if buffer and any(s.strip() for s in buffer) and any(s.isalpha() for s in buffer):
            html_output.append(f"<p>{' '.join(buffer).strip()}</p>")

        # Join all parts of the HTML output
        return ''.join(html_output)


    def translate_responsibilities(self, language_code):
        translations = {
            'en': 'Responsibilities',
            'fr': 'Responsabilités',
            'es': 'Responsabilidades',
            'de': 'Verantwortlichkeiten',
            'it': 'Responsabilità',
            'pt': 'Responsabilidades',
            'ar': 'المسؤوليات',
            'zh': '职责',
            'ja': '責任',
            'ru': 'Обязанности'
        }

        # Return the translation or a default message if the language code is not in the dictionary
        return translations.get(language_code, 'Responsibilities')


    def translate_skills(self, language_code):
        translations = {
            'en': 'Skills',
            'fr': 'Compétences',
            'es': 'Habilidades',
            'de': 'Fähigkeiten',
            'it': 'Competenze',
            'pt': 'Habilidades',
            'ar': 'المهارات',
            'zh': '技能',
            'ja': 'スキル',
            'ru': 'Навыки'
        }

        # Return the translation or a default message if the language code is not in the dictionary
        return translations.get(language_code, 'Skills')


    def escape_unescaped_quotes(self, json_str):
        # Regular expression to match the start of a value in a JSON key-value pair
        # Match : "value..." while allowing any content within the value
        pattern = re.compile(r'(:\s*")([^"]*?)(")')

        # Replace unescaped double quotes within each matched value, excluding the outer quotes
        def escape_quotes(match):
            value_content = match.group(2).replace('"', '\\"')
            return f'{match.group(1)}{value_content}{match.group(3)}'

        # Apply the pattern to escape quotes inside values
        escaped_json_str = pattern.sub(escape_quotes, json_str)
        return escaped_json_str


    def replace_unbalanced_quote(self, json_str):
        # Pattern to find the first unbalanced double quote after the "«" character, ignoring closing quotes for values
        pattern = re.compile(r'«([^»]*?)"([^»,}\]\n]*?)(?=[,}\]])')

        # Function to replace with "»" only if it's truly an unbalanced quote within the value
        def replace_with_closing_sign(match):
            # Replace the first unbalanced double quote after "«" with "»"
            return f'«{match.group(1)}»{match.group(2)}'

        # Apply the pattern and replacement function
        corrected_json_str = pattern.sub(replace_with_closing_sign, json_str)
        return corrected_json_str


    def detect_predominant_language(self, text):
        # Split text into sentences
        sentences = re.split(r'[.!?]', text)  # Simple sentence splitting on punctuation
        language_counts = Counter()
        
        # Detect language for each sentence and tally results
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:  # Only process non-empty sentences
                try:
                    language = detect(sentence)
                    language_counts[language] += 1
                except LangDetectException:
                    # Ignore sentences too short to detect
                    continue
        
        # Find the most common language
        if language_counts:
            predominant_lang_code = language_counts.most_common(1)[0][0]
            # Map language code to language name
            lang_name = {
                'en': 'English', 'fr': 'French', 'es': 'Spanish', 'de': 'German',
                'it': 'Italian', 'pt': 'Portuguese', 'ar': 'Arabic', 'zh-cn': 'Chinese',
                # Add more languages if needed
            }.get(predominant_lang_code, "Unknown")
            return lang_name
        else:
            return "Unknown"


    def extract_infos_from_cv_v3(self, cv_text, return_summary=False, file_name=""):
        #with open("cv_text.txt", "w+") as f:
        #    f.write(cv_text)
        extracted_info = {}
        #with open("request_prod.txt", "w+") as f:
        #    f.write(cv_text)
        # Construct a single prompt with all the extraction instructions
        language = self.detect_predominant_language(cv_text)
        combined_prompt = f"""
        Extract the following information from the CV text as they are without any translation, following this format as it is without changing anything (NUMBER. KEY_NAME: RESULT), keeping the numbering, and making sure to correctly format json objects:

        1. Name: Extract the name of the candidate (Remove unnecessary spaces within the name if found, and leave only spaces seperating first name from middle name, if there was a middle name, from last name, and correct capitalization).
        2. Email: Extract the email of the candidate.
        3. Phone: Extract the phone number of the candidate.
        4. Age: Extract the age of the candidate, and write the number only. If no age is found, write an empty string ''.
        5. City: Extract the city of the candidate. If no city is found, write an empty string ''.
        6. Work Experiences: For each experience, return a JSON object with "job_title", "company_name", "context", "responsibilities", "skills", "city", "start_date," and "end_date". Whenever you find a new date, that's a new experience, you should seperate experiences within the same company but with different time periods. for "context", populate it with the context text if found, the context text is usually found under the title "context" or "contexte", if no such text if found, leave it empty.  in "responsibilities", list tasks as a string, enclosed with "#start#" at the beginning of every task sentence, and the string "#end#" at the end of every task sentence, extracted from the resume text. if tasks aren’t explicitly listed, intelligently extract them from the resume text (task sentences should be complete imformative sentences). In "skills", if there was a paragraph dedicated to listing technical skills in the experience text (Only in the text of said, not in other sections, the experience text ends once another experience text starts, or once another entirely different secion starts), put it here (it's usually preceeded with a title such as "Technical Skills", "Compétence techniques", "Skills", "Compétences", "Technical Environment", "Environnement technique" or similar titles, written in {language}). If there was no paragraph listing the skills acquired on that experience, then leave "skills" empty. For dates, mark one ongoing role with "present" as "end_date" and set missing dates to empty strings. every "start_date" and "end_date" should contain both the month and year if they exist, or only the year if the month doesn't exist. sort experiences from most recent to oldest, and make sure that Json objects are formatted correctly.
        7. Years Of Experience: calculate the number of years of experience from work experiences, the current year minus the oldest year you can find in work experiences (which is the start year of the oldest work experience), should be the number. Write the number only. Please be sure to find the oldest year correctly.
        8. Educations: Extract all educations and formations in JSON format as a list containing degree, institution, start_year, and end_year, with years being string.
        9. Languages: Extract all spoken languages (non-programming) in JSON format as a list containing language and level, and translate them to {language} if they're not already. If no language is found, write an empty list [].
        10. Skills: Extract all technical skills (non-social) in JSON format as a list containing skill, level and category. Don't repeat the same skill more than once. and don't exceed 20 json objects. Also, for the category, choose a groupe under which the skill can be labeled (EX: if the skill was JavaScript, the category will be programming languages, but in {language}), use your intelligence to group skills, and write categories names in {language}, and don't exceed 6 different categories overall.
        11. Interests: Extract all interests/hobbies in JSON format as a list containing interest. If no interest is found, write an empty list [].
        12. Social Skills: Extract all soft skills (social, communication, etc.) in JSON format as a list of objects, each object containing "skill" as a key, with the skill as the value. Don't exceed 10 json objects, if there are more than 10 social skills, try merging the ones that can be merged with each other. If no social skill is found, write an empty list []. (write all soft skills in {language} as they are written in the resume text)
        13. Certifications: Extract all certifications in JSON format as a list containing certification, institution, link, and date. Translate certification to {language}, and if no certification is found, write an empty list [].
        14. Projects: Extract all projects in JSON format as a list containing project_name, description, skills, start_date, and end_date, the description must contain any text you can find talking about the project, if the project's description contains sentences in the form of bullet points or list of tasks (if there is more than one newline in the description, there is likely a list of tasks), add the string "#start#" at the start of each task sentence, and "#end#" at the end of each task sentence. if the description doesn't contain bulltet points or list of tasks, and was actually one lengthy continuous paragraph, write it as it is without "#start#" and "#end#". please check carfully whether the is a list of tasks or not. for the skills, list all hard technical skills mentioned in the description of the project at hand, seperated by a comma. projects can be found either in a dedicated section called projects, or inside work experiences, clearly highlighted as projects. if no project is found, write an empty list [].
        15. Volunteering: Extract all volunteering experiences in JSON format as a list containing organization, position, description, start_date, and end_date, and if no volunteering experience is found, write an empty list [].
        16. References: Extract all references in JSON format as a list containing name, position, company, email, and phone (do not include candidate's own contacts), and if no reference is found, write an empty list [].
        17. Headline: Extract the current occupation of the candidate, if it wasn't explicitly mentioned, deduce it from the most recent work experience (Remove unnecessary spaces within words if found, and leave necessary spaces, and correct capitalization).
        18. Summary: If a summary exists in the Resume already, extract it, you can find it either at the beginning or at the end, take the longest one. (if no summary is found in Resume data, then leave an empty string)
        CV Text:
        {cv_text}



	---------------------------------------------------------------------
        Please process the above tasks efficiently to minimize response time.
        """
        #with open("cv_text.txt", "w+") as f:
        #    f.write(combined_prompt)
        try:
            # Make a single call with the combined prompt
            response = self.chain({"context": "CV Extraction", "question": combined_prompt, "input_documents": []},
                                  return_only_outputs=True)
            language = None
            if response:
                response_text = response["output_text"].strip()
                with open("response_test.txt", "w+") as f:
                   f.write(response_text)
                # Define the labels we expect in the response
                labels = {
                    "1. Name:": "name",
                    "2. Email:": "email",
                    "3. Phone:": "phone",
                    "4. Age:": "age",
                    "5. City:": "city",
                    "6. Work Experiences:": "work",
                    "7. Years Of Experience:": "yoe",
                    "8. Educations:": "educations",
                    "9. Languages:": "languages",
                    "10. Skills:": "skills",
                    "11. Interests:": "interests",
                    "12. Social Skills:": "social",
                    "13. Certifications:": "certifications",
                    "14. Projects:": "projects",
                    "15. Volunteering:": "volunteering",
                    "16. References:": "references",
                    "17. Headline:": "headline",
                    "18. Summary:": "summary",
                }

                for label, key in labels.items():
                    start_idx = response_text.find(label)
                    if start_idx != -1:
                        start_idx += len(label)
                        # Find the next label or the end of the text
                        end_idx = len(response_text)
                        for next_label in labels:
                            next_label_idx = response_text.find(next_label, start_idx)
                            if next_label_idx != -1 and next_label_idx < end_idx:
                                end_idx = next_label_idx
                        section_text = response_text[start_idx:end_idx].strip()
                        # Check if the section should be parsed as JSON
                        if key in ["work_experiences", "educations", "languages", "skills", "interests",
                                   "social", "certifications", "projects", "volunteering", "references", "work"]:
                            try:
                                if key == "work":
                                    section_text = re.sub(r'', '', section_text)
                                    section_text = section_text.replace("\\n", "").replace("\n", "")
                                    section_text = self.replace_unbalanced_quote(section_text)
                                    temp_work = json.loads(section_text)
                                    for key_work, work in enumerate(temp_work):
                                        temp_work[key_work]["responsibilities"] = self.convert_to_html(work["responsibilities"]) if self.convert_to_html(work["responsibilities"]).replace("\n", "") else ""
                                        #temp_work[key_work]["responsibilities"] = f"{temp_work[key_work]["responsibilities"]}<p>{temp_work[key_work]["skills"]}</p></br>" if temp_work[key_work]["skills"] != "" else temp_work[key_work]["responsibilities"]
                                        temp_work[key_work]["environnement"] = temp_work[key_work]["skills"]
                                        del temp_work[key_work]["skills"]
                                        if temp_work[key_work]["responsibilities"] and not language:
                                            language = detect(temp_work[key_work]["responsibilities"])
                                        temp_work[key_work]["responsibilities"] = f"{temp_work[key_work]['responsibilities']}"
                                    temp_work = self.merge_jobs(temp_work)
                                    section_text = json.dumps(temp_work)
                                if key == "projects":
                                    section_text = re.sub(r'', '', section_text)
                                    section_text = section_text.replace("\\n", "").replace("\n", "")
                                    section_text = self.replace_unbalanced_quote(section_text)
                                    temp_project = json.loads(section_text)
                                    for key_project, project in enumerate(temp_project):
                                        temp_project[key_project]["description"] = self.convert_to_html(project["description"]) if self.convert_to_html(project["description"]).replace("\n", "") else ""
                                    section_text = json.dumps(temp_project)
                                extracted_info[key] = json.loads(section_text.replace("\n", "").replace("\r", "").replace("<br><br>", "<br>"))
                                if key == "interests":
                                    processed_interests = []
                                    if isinstance(extracted_info[key], list):
                                        for item in extracted_info[key]:
                                            if isinstance(item, dict) and 'interest' in item:
                                                if ',' in item['interest']:
                                                    split_interests = item['interest'].split(',')
                                                    for interest in split_interests:
                                                        processed_interests.append({'interest': (interest.strip()).split("...")[0] if interest.endswith("...") else interest.strip()})
                                                else:
                                                    processed_interests.append(item)
                                            elif isinstance(item, str):
                                                processed_interests.append({'interest': item})
                                    extracted_info[key] = processed_interests
                            except Exception as e:
                                with open("response_prod.txt", "w+") as f:
                                    f.write(section_text)
                                print(e)
                                print(key)
                                extracted_info[key] = []  # Handle JSON decode error
                        else:
                            if key == "email":
                                extracted_info[key] = section_text.replace("\n", "").replace(" ", "")
                                extracted_info[key] = re.sub(r'^[\W_]+', '', extracted_info[key]).lower()
                            else:
                                extracted_info[key] = section_text.replace("\n", "") if section_text.replace("\n", "") != "" else None
        except Exception as e:
            print("Safety rating issue detected:", e)
            return extracted_info

        if "email" in extracted_info and extracted_info["email"]:
            threading.Thread(target=self.process_report_entry, args=(file_name, extracted_info["email"], combined_prompt, response_text)).start()

        # Handle the summary extraction separately
        if return_summary:
            summary_prompt = f"""
            Generate a summary that describes the extracted information from the CV based on the text of the CV. Aim for a well-structured and captivating summary that reflects the essence of the candidate's capabilities and aspirations, with a first-person point of view, and do not exceed 200 characters.

            CV Text:
            {cv_text}
            """
            try:
                summary_response = self.chain(
                    {"context": "Summary Extraction", "question": summary_prompt, "input_documents": []},
                    return_only_outputs=True)
                if summary_response:
                    extracted_info["summary"] = summary_response["output_text"].strip().replace("\n", "")
                else:
                    extracted_info["summary"] = ""
            except genai.types.generation_types.StopCandidateException as e:
                print("Safety rating issue detected during summary extraction:", e)
                extracted_info["summary"] = ""
        #with open("Cv.json", "w+") as f:
        #    json.dump(extracted_info, f)
        extracted_info["language"] = language
        if "work" in extracted_info and extracted_info["work"]:
            first_start_date = extracted_info["work"][-1]["start_date"]
            match = re.search(r'\b(19|20)\d{2}\b', first_start_date)
            year = int(match.group()) if match else None
            current_year = datetime.now().year
            extracted_info["yoe"] = f"{current_year - year}" if year else extracted_info["yoe"]
        return extracted_info


    def extract_infos_from_cv_v2(self, cv_input, return_summary=False, file_name=""):
        #with open("cv_text.txt", "w+") as f:
        #    f.write(cv_text)
        extracted_info = {}
        # Construct a single prompt with all the extraction instructions
        combined_prompt = f"""
        Answer the request as detailed as possible from the provided context.
        if the answer is not in the provided context just give an empty string "", don't provide the wrong answer nor say that the answer is not provided in the context, 
        Do not add any extra characters other than the ones demanded. Make sure the answer is provided in the way that it is asked in the question.
        Please, use as much processing power as you can to answer as fast as possible.
        DO NOT ANSWER WITH A JSON, RESPOND ONLY WITH THE OUTLINED FORMAT AS A TEXT, KEEPING THE NUMBERS.

        Context :
        Resume Text Extraction

        Request:
        Extract the following information from the CV text as they are without any translation, following this format as it is, keeping the numbering, and make sure to correctly format the ones that needs json objects:

        1. Name: Extract the name of the candidate (Remove unnecessary spaces within the name if found, and leave only spaces seperating first name from middle name, if there was a middle name, from last name, and correct capitalization).
        2. Email: Extract the email of the candidate.
        3. Phone: Extract the phone number of the candidate.
        4. Age: Extract the age of the candidate, and write the number only. If no age is found, write an empty string ''.
        5. City: Extract the city of the candidate. If no city is found, write an empty string ''.
        6. Work Experiences: For each experience, return a JSON object with "job_title", "company_name", "context", "responsibilities", "skills", "city", "start_date," and "end_date". Whenever you find a new date, that's a new experience, you should seperate experiences within the same company but with different time periods. for "context", populate it with the context text if found, the context text is usually found under the title "context" or "contexte", if no such text if found, leave it empty.  in "responsibilities", list tasks as a string, enclosed with "#start#" at the beginning of every task sentence, and the string "#end#" at the end of every task sentence, extracted from the resume text. if tasks aren’t explicitly listed, intelligently extract them from the resume text (task sentences should be complete imformative sentences). In "skills", if there was a paragraph dedicated to listing technical skills in the experience text (Only in the text of said, not in other sections, the experience text ends once another experience text starts, or once another entirely different secion starts), put it here (it's usually preceeded with a title such as "Technical Skills", "Compétence techniques", "Skills", "Compétences", "Technical Environment", "Environnement technique" or similar titles, written in the language of the resume, which should either be English or French). If there was no paragraph listing the skills acquired on that experience, then leave "skills" empty. For dates, mark one ongoing role with "present" as "end_date" and set missing dates to empty strings. every "start_date" and "end_date" should contain both the month and year if they exist, or only the year if the month doesn't exist. sort experiences from most recent to oldest, and make sure that Json objects are formatted correctly.
        7. Years Of Experience: calculate the number of years of experience from work experiences, the current year minus the oldest year you can find in work experiences (which is the start year of the oldest work experience), should be the number. Write the number only. Please be sure to find the oldest year correctly.
        8. Educations: Extract all educations and formations in JSON format as a list containing degree, institution, start_year, and end_year, with years being string.
        9. Languages: Extract all spoken languages (non-programming) in JSON format as a list containing language and level, and translate them to the language of the resume if they're not already (you can identify the language of the resume from work experiences text, which should either be English or French). If no language is found, write an empty list [].
        10. Skills: Extract all technical skills (non-social) in JSON format as a list containing skill, level and category. Don't repeat the same skill more than once. and don't exceed 20 json objects. Also, for the category, choose a groupe under which the skill can be labeled (EX: if the skill was JavaScript, the category will be programming languages, but in the language of the resume after you detect it, which should either be English or French), use your intelligence to group skills, and write categories names in the language of the resume (you can identify the language of the resume from experiences), and don't exceed 6 different categories overall.
        11. Interests: Extract all interests/hobbies in JSON format as a list containing interest. If no interest is found, write an empty list [].
        12. Social Skills: Extract all soft skills (social, communication, etc.) in JSON format as a list of objects, each object containing "skill" as a key, with the skill as the value. Don't exceed 10 json objects, if there are more than 10 social skills, try merging the ones that can be merged with each other. If no social skill is found, write an empty list []. (write all soft skills in the language of the resume as they are written in the resume text, and don't forget that each skill should be in an object)
        13. Certifications: Extract all certifications in JSON format as a list containing certification, institution, link, and date. Translate certification to the language of the resume (you can identify it from work experiences text, which should either be English or French), and if no certification is found, write an empty list [].
        14. Projects: Extract all projects in JSON format as a list containing project_name, description, skills, start_date, and end_date, the description must contain any text you can find talking about the project, analyze the project's text and extract from it a list of tasks based on your understanding of the context and the positions of newlines, then add the string "#start#" at the beginning of each task sentence, and "#end#" at the end of each task sentence. whenever you find a newline in the project's description, that's the end of a task and the start of another, so please don't forget adding the strings "#start#" and "#end#". for the skills, list all hard technical skills mentioned in the description of the project at hand, seperated by a comma. projects can be found either in a dedicated section called projects, or inside work experiences, clearly highlighted as projects. if no project is found, write an empty list [].
        15. Volunteering: Extract all volunteering experiences in JSON format as a list containing organization, position, description, start_date, and end_date, and if no volunteering experience is found, write an empty list [].
        16. References: Extract all references in JSON format as a list containing name, position, company, email, and phone (do not include candidate's own contacts), and if no reference is found, write an empty list [].
        17. Headline: Extract the current occupation of the candidate, if it wasn't explicitly mentioned, deduce it from the most recent work experience (Remove unnecessary spaces within words if found, and leave necessary spaces, and correct capitalization).
        18. Summary: If a summary exists in the Resume already, extract it, you can find it either at the beginning or at the end, take the longest one. (if no summary is found in Resume data, then leave an empty string)


        Don't forget to identify the language of the resume from work experiences text, which should either be English or French.

	Please process the above tasks efficiently to minimize response time. and don't forget the instruction of adding "#start#" and "#end#" for responsabilities and projects descriptions.
        """

        uploaded_files = []
        if isinstance(cv_input, str):
            combined_prompt = f"{combined_prompt}\nCV Text:\n{cv_input}"
            input_files = None
        elif file_name.endswith(".pdf"):
            combined_prompt = combined_prompt.replace("the CV text as they are", "the provided CV pdf as it is")
            combined_prompt = f"{combined_prompt}\nCV PDF:"
            #uploaded_files.append(genai.upload_file(path=file_name))
            with upload_lock:
                uploaded_files.append(genai.upload_file(path=file_name))
            #for image_path in cv_input:
                # with open(image_path, "rb") as f:
                #     image = f.read()
                # uploaded_file = base64.b64encode(image).decode("utf-8")
                # uploaded_files.append(f"data:image/jpeg;base64,{uploaded_file}")
                #uploaded_files.append(genai.upload_file(path=image_path))
            # image_references = '\n'.join([f'{file}\n' for file in uploaded_files])
            # combined_prompt += f"\n{image_references}"
            uploaded_files.append(combined_prompt)
        else:
            return {"error": "Invalid input type for cv_input"}
        try:
            # Make a single call with the combined prompt
            response = self.model.generate_content(uploaded_files)
            language = None
            if response:
                response_text = response.text.strip().replace("```json", "").replace("```", "")
                with open("response_test.txt", "w+") as f:
                   f.write(response_text)
                # Define the labels we expect in the response
                labels = {
                    "1. Name:": "name",
                    "2. Email:": "email",
                    "3. Phone:": "phone",
                    "4. Age:": "age",
                    "5. City:": "city",
                    "6. Work Experiences:": "work",
                    "7. Years Of Experience:": "yoe",
                    "8. Educations:": "educations",
                    "9. Languages:": "languages",
                    "10. Skills:": "skills",
                    "11. Interests:": "interests",
                    "12. Social Skills:": "social",
                    "13. Certifications:": "certifications",
                    "14. Projects:": "projects",
                    "15. Volunteering:": "volunteering",
                    "16. References:": "references",
                    "17. Headline:": "headline",
                    "18. Summary:": "summary",
                }

                for label, key in labels.items():
                    start_idx = response_text.find(label)
                    if start_idx != -1:
                        start_idx += len(label)
                        # Find the next label or the end of the text
                        end_idx = len(response_text)
                        for next_label in labels:
                            next_label_idx = response_text.find(next_label, start_idx)
                            if next_label_idx != -1 and next_label_idx < end_idx:
                                end_idx = next_label_idx
                        section_text = response_text[start_idx:end_idx].strip()

                        # Check if the section should be parsed as JSON
                        if key in ["work_experiences", "educations", "languages", "skills", "interests",
                                   "social", "certifications", "projects", "volunteering", "references", "work"]:
                            try:
                                if key == "work":
                                    section_text = re.sub(r'', '', section_text)
                                    section_text = section_text.replace("\\n", "").replace("\n", "")
                                    section_text = self.replace_unbalanced_quote(section_text)
                                    temp_work = json.loads(section_text)
                                    for key_work, work in enumerate(temp_work):
                                        if temp_work[key_work]["responsibilities"]:
                                            temp_work[key_work]["responsibilities"] = self.convert_to_html(work["responsibilities"]) if self.convert_to_html(work["responsibilities"]).replace("\n", "") else ""
                                        #temp_work[key_work]["responsibilities"] = f"{temp_work[key_work]["responsibilities"]}<p>{temp_work[key_work]["skills"]}</p></br>" if temp_work[key_work]["skills"] != "" else temp_work[key_work]["responsibilities"]
                                        temp_work[key_work]["environnement"] = temp_work[key_work]["skills"]
                                        del temp_work[key_work]["skills"]
                                        if temp_work[key_work]["responsibilities"] and not language:
                                            language = detect(temp_work[key_work]["responsibilities"])
                                        temp_work[key_work]["responsibilities"] = f"{temp_work[key_work]['responsibilities']}"
                                    temp_work = self.merge_jobs(temp_work)
                                    section_text = json.dumps(temp_work)
                                if key == "projects":
                                    section_text = re.sub(r'', '', section_text)
                                    section_text = section_text.replace("\\n", "").replace("\n", "")
                                    section_text = self.replace_unbalanced_quote(section_text)
                                    temp_project = json.loads(section_text)
                                    for key_project, project in enumerate(temp_project):
                                        temp_project[key_project]["description"] = self.convert_to_html(project["description"]) if self.convert_to_html(project["description"]).replace("\n", "") else ""
                                    section_text = json.dumps(temp_project)
                                extracted_info[key] = json.loads(section_text.replace("\n", "").replace("\r", "").replace("<br><br>", "<br>"))
                                if key == "interests":
                                    processed_interests = []
                                    if isinstance(extracted_info[key], list):
                                        for item in extracted_info[key]:
                                            if isinstance(item, dict) and 'interest' in item:
                                                if ',' in item['interest']:
                                                    split_interests = item['interest'].split(',')
                                                    for interest in split_interests:
                                                        processed_interests.append({'interest': (interest.strip()).split("...")[0] if interest.endswith("...") else interest.strip()})
                                                else:
                                                    processed_interests.append(item)
                                            elif isinstance(item, str):
                                                processed_interests.append({'interest': item})
                                    extracted_info[key] = processed_interests
                            except Exception as e:
                                with open("response_prod.txt", "w+") as f:
                                    f.write(section_text)
                                print(e)
                                print(key)
                                extracted_info[key] = []  # Handle JSON decode error
                        else:
                            if key == "email":
                                extracted_info[key] = section_text.replace("\n", "").replace(" ", "")
                                extracted_info[key] = re.sub(r'^[\W_]+', '', extracted_info[key]).lower()
                            else:
                                extracted_info[key] = section_text.replace("\n", "") if section_text.replace("\n", "") != "" else None
        except Exception as e:
            print("Safety rating issue detected:", e)
            return extracted_info


        if "email" in extracted_info and extracted_info["email"]:
            threading.Thread(target=self.process_report_entry, args=(file_name, extracted_info["email"], combined_prompt, response_text)).start()

        # Handle the summary extraction separately
        if return_summary:
            summary_prompt = f"""
            Generate a summary that describes the extracted information from the CV based on the text of the CV. Aim for a well-structured and captivating summary that reflects the essence of the candidate's capabilities and aspirations, with a first-person point of view, and do not exceed 200 characters.

            CV Text:
            {cv_text}
            """
            try:
                summary_response = self.chain(
                    {"context": "Summary Extraction", "question": summary_prompt, "input_documents": []},
                    return_only_outputs=True)
                if summary_response:
                    extracted_info["summary"] = summary_response["output_text"].strip().replace("\n", "")
                else:
                    extracted_info["summary"] = ""
            except genai.types.generation_types.StopCandidateException as e:
                print("Safety rating issue detected during summary extraction:", e)
                extracted_info["summary"] = ""

        keys_to_check = [
            "work", "educations", "languages", "skills", "interests", 
            "social", "certifications", "projects", "volunteering", "references"
        ]
        for key in keys_to_check:
            if key not in extracted_info or not isinstance(extracted_info[key], list):
                extracted_info[key] = []
            else:
                # Ensure all elements in the list are dictionaries
                if not all(isinstance(item, dict) for item in extracted_info[key]):
                    extracted_info[key] = []


        #with open("Cv.json", "w+") as f:
        #    json.dump(extracted_info, f)
        extracted_info["language"] = language
        if "work" in extracted_info and extracted_info["work"]:
            first_start_date = extracted_info["work"][-1]["start_date"]
            match = re.search(r'\b(19|20)\d{2}\b', first_start_date)
            year = int(match.group()) if match else None
            current_year = datetime.now().year
            extracted_info["yoe"] = f"{current_year - year}" if year else extracted_info["yoe"]
        return extracted_info



    def process_report_entry(self, file_name, email, combined_prompt, response_text):
        input_tokens = self.count_tokens(combined_prompt)
        extraction_input_cost = self.calculate_cost(input_tokens, True)
        output_tokens = self.count_tokens(response_text)
        extraction_output_cost = self.calculate_cost(output_tokens, False)
        #now = datetime.now(timezone.utc).isoformat()
        now = datetime.now(timezone.utc)
        now = now.strftime("%B %d, %Y %H:%M:%S %Z")
        report_entry = {
            "nom_du_fichier": file_name,
            "email": email,
            "date": now,
            "données": {
                "nombre_de_tokens_d'entrée_extraction": input_tokens,
                "coût_estimé_des_tokens_d'entrée_extraction": extraction_input_cost,
                "nombre_de_tokens_d'entrée_traduction": 0,
                "coût_estimé_des_tokens_d'entrée_traduction": 0,
                "nombre_de_tokens_de_sortie_extraction": output_tokens,
                "coût_estimé_des_tokens_de_sortie_extraction": extraction_output_cost,
                "nombre_de_tokens_de_sortie_traduction": 0,
                "coût_estimé_des_tokens_de_sortie_traduction": 0,
                "nombre_total_de_tokens": input_tokens + output_tokens,
                "coût_total_estimé": extraction_input_cost + extraction_output_cost
            }
        }
        with self.lock:
            with open(self.report_file, "r") as f:
                reports = json.load(f)
            reports.insert(0, report_entry)
            with open(self.report_file, "w") as f:
                json.dump(reports, f, indent=4)
        #self.run_finalize_report_after_delay(email, now)


    def run_finalize_report_after_delay(self, email, now):
        # Use threading.Timer to delay the execution of finalize_report
        timer = threading.Timer(30.0, self.finalize_report, args=(email, now))
        # Start the timer in a new thread so it won't block the main thread
        timer.start()
    

    def finalize_report(self, email, report_date):
        with self.lock:
            with open(self.report_file, "r") as f:
                reports = json.load(f)
            for report in reports:
                if report["email"] == email and report["date"] == report_date:
                    report["données"]["nombre_de_tokens_d'entrée_traduction"] = 0
                    report["données"]["coût_estimé_des_tokens_d'entrée_traduction"] = 0
                    report["données"]["nombre_de_tokens_de_sortie_traduction"] = 0
                    report["données"]["coût_estimé_des_tokens_de_sortie_traduction"] = 0
                    report["données"]["nombre_total_de_tokens"] += 0
                    report["données"]["coût_total_estimé"] += 0
            with open(self.report_file, "w") as f:
                json.dump(reports, f, indent=4)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++")

    # Function to extract information from the CV
    def extract_infos_from_cv(self, cv_text, return_summary=False):
        extracted_info = {}

        prompts = [
            ("Title extraction", "Extract exactly the title of the candidat from the CV text:\n" + cv_text + "\n"),
            ("Name extraction", "Extract exactly the name of the candidat from the CV text:\n" + cv_text + "\n"),
            ("Email extraction", "Extract exactly the email from the CV text:\n" + cv_text + "\n"),
            ("Phone extraction", "Extract exactly the phone number from the CV text:\n" + cv_text + "\n"),
            ("Age extraction", "Extract exactly the age from the CV text:\n" + cv_text + "\n"),
            ("City extraction", "Extract exactly the city from the CV text:\n" + cv_text + "\n"),
            ("Work Experiences extraction", """Extract all the work experiences as a list JSON output containing the following information:
            {"job_title": "", "company_name": "", "responsibilities": "", "city": "", "start_date": "", "end_date": ""} from the CV text:\n """ + cv_text + "\n"),
            ("Educations extraction", """Extract all the educations and formations as a list JSON output containing the following information:
            {"degree": "", "institution": "", "start_year": "", "end_year": ""} from the CV text:\n """ + cv_text + "\n"),
            ("Languages extraction", """Extract all the spoken languages (non-programming) as a list JSON output containing the following information:
            {"language": "", "level": ""} from the CV text:\n """ + cv_text + "\n"),
            ("Skills extraction", """Extract all the technical skills (non-social) and the programming languages as a list JSON output containing the following information:
            {"skill": "", "level": ""} from the CV text:\n """ + cv_text + "\n"),
            ("Interests extraction", """Extract all the interests/hobbies as a list JSON output containing the following information:
             {"interest": ""} from the CV text:\n """ + cv_text + "\n"),
            ("Social Skills extraction", """Extract all the soft skills (social, communication, etc..) as a list JSON output containing the following information:
             {"skill": ""} from the CV text:\n """ + cv_text + "\n"),
            ("Certifications extraction", """Extract all the certifications as a list JSON output containing the following information if the output doesn't exist return []:
            {"certification": "", "institution": "", "link": "", "date": ""} from the CV text:\n """ + cv_text + "\n"),
            ("Projects extraction", """Extract all the projects as a list JSON output containing the following information:
            {"project_name": "", "description": "", "start_date": "", "end_date": ""} from the CV text:\n """ + cv_text + "\n"),
            ("Volunteering extraction", """Extract all the volunteering experiences as a list JSON output containing the following information:
            {"organization": "", "position": "", "description": "", "start_date": "", "end_date": ""} from the CV text:\n """ + cv_text + "\n"),
            ("References extraction", """Extract all the references as a list JSON output containing the following information:
            {"name": "", "position": "", "company": "", "email": "", "phone": ""} and don't provide the contacts of the candidat himself, from the CV text:\n """ + cv_text + "\n"),
        ]

        if return_summary:
            prompts.append(("Summary extraction", """Générer un résumé qui décrit les informations extraites du CV à partir du texte du CV. Visez un 
                            résumé bien structuré et captivant qui reflète l'essence des capacités et des aspirations du candidat et ne pas dépasser 200 caractères.\n""" + cv_text + "\n"))

        for context, prompt_text in prompts:
            try:
                response = self.chain({"context": context, "question": prompt_text, "input_documents": []}, return_only_outputs=True)
                resp_strings = ["Title extraction", "Name extraction", "Email extraction", "Phone extraction", "Age extraction", "City extraction", "Summary extraction"]
                if(context in resp_strings):
                    if response:
                        extracted_info[context.lower().split()[0]] = response["output_text"].strip()
                    else:
                        extracted_info[context.lower().split()[0]] = ""
                else:
                    if response:
                        try:
                            if response["output_text"][-1] != "]":
                                # Find the index of the first occurrence of '['
                                list_index = response["output_text"].find("[")
                                if list_index != -1:
                                    # Extract the JSON part from the output text
                                    response["output_text"] = response["output_text"][list_index:]
                                else:
                                    response["output_text"] = "[" + response["output_text"] # Add the opening bracket to the JSON string

                                # Find the index of the last occurrence of '}'
                                last_brace_index = response["output_text"].rfind("}")
                                if last_brace_index != -1:
                                    # Extract the JSON part from the output text
                                    response["output_text"] = response["output_text"][:last_brace_index + 1]
                                    response["output_text"] = response["output_text"] + "]" # Add the closing bracket to the JSON string

                            # Parse the JSON string into Python object
                            json_data = json.loads(response["output_text"])
                            extracted_info[context.lower().split()[0]] = json_data

                        except json.decoder.JSONDecodeError as e:
                            print("JSON Decode Error:", e)
                            print("Problematic Text:", response["output_text"])
                            # Handle the JSON decoding error here, such as logging the error or skipping this part of the processing
                            extracted_info[context.lower().split()[0]] = []  # or handle the error accordingly
                    else:
                        extracted_info[context.lower().split()[0]] = []
            except genai.types.generation_types.StopCandidateException as e:
                print("Safety rating issue detected for prompt '{}': {}".format(context, e))
                extracted_info[context.lower().split()[0]] = ""

        return extracted_info
    
    # Function to get the extraction rate of the CV
    def get_extraction_rate(self, cv_text, extracted_info):
        # Analyze each line of the CV text for extracted information
        lines = cv_text.split("\n")
        line_info = {}
        for line in lines:
            line_info[line] = any(word.lower() in text or word.lower() in nested_text 
                      for text in extracted_info.values() 
                      for nested_text in self.extract_nested_texts(text)
                      for word in line.split())

        # Calculate the extraction rate
        total_lines = len(lines)
        extracted_lines = sum(line_info.values())
        extraction_rate = extracted_lines / total_lines * 100

        return extraction_rate
    
    # Function to extract nested texts from the extracted information
    def extract_nested_texts(self, text):
        nested_texts = []

        # Check if the text is a dictionary
        if isinstance(text, dict):
            for key, value in text.items():
                if(key == "summary"):
                    continue
                # Recursively check nested dictionaries and lists
                nested_texts.extend(self.extract_nested_texts(value))
        elif isinstance(text, list):
            for item in text:
                # Recursively check items in a list
                nested_texts.extend(self.extract_nested_texts(item))
        elif isinstance(text, str):
            # Add the text to the list if it is a string
            nested_texts.append(text.lower())

        return nested_texts

    # Function to extract a summary from the CV
    def extract_summary_from_cv(self, cv_text):
        prompt = """Générer un résumé qui décrit les informations extraites du CV à partir du texte du CV. Visez un 
                    résumé bien structuré et captivant qui reflète l'essence des capacités et des aspirations du candidat et ne pas dépasser 200 caractères.\n""" + cv_text + "\n"
        response = self.chain({"context": "Summary extraction", "question": prompt, "input_documents": []}, return_only_outputs=True)
        if response:
            return response["output_text"].strip()
        else:
            return ""


    # Function to extract a summary from the CV
    def extract_summary_from_cv_v2(self, cv_text, lang):
        # prompt = f"""Generate a summary that describes the following information extracted from a CV. Aim for a well-structured, captivating summary that reflects the essence of the candidate's abilities and aspirations, in the first-person point of view, without exceeding 300 characters. The summary should be in {lang}, written in one paragraph. Here are the CV data, make sure to only answer with the summary alone, NOTHING ELSE SHOULD BE ADDED, NO LABLES, NO COMMENTS:\n\n""" + cv_text + "\n"
        prompt = f"""Act as a senior Recruitment Business Engineer specializing in sourcing technical talent. Carefully review the candidate's resume and craft a compelling summary of their skills and experiences. Highlight their key achievements, technical expertise, and ability to integrate into complex projects. Adopt a professional, clear, and concise tone, in the first-person point of view, emphasizing results and the added value they could bring to a company. The summary should reflect the candidate's potential while being tailored to the expectations of technical and HR decision-makers. Do not exceed 300 words, and assuming you can generate many different versions of this summary, let this one be the version number {random.randint(0,20)}. The summary should be in {lang}, written in one paragraph. Here are the CV data, make sure to only answer with the summary alone, NOTHING ELSE SHOULD BE ADDED, NO LABLES, NO COMMENTS:\n\n""" + cv_text + "\n"
        #with open("myPrompt.txt", "w+") as f:
        #    f.write(prompt)
        response = self.chain({"context": "Summary extraction", "question": prompt, "input_documents": []}, return_only_outputs=True)
        if response:
            return response["output_text"].strip()
        else:
            return ""


    # Function to clean the extracted data
    def clean_data(self, data):
        if data is None:
            return None
        elif isinstance(data, dict):
            cleaned_data = {}
            for key, value in data.items():
                cleaned_value = self.clean_data(value)
                cleaned_data[key] = cleaned_value
            return cleaned_data
        elif isinstance(data, list):
            cleaned_data = []
            for item in data:
                cleaned_item = self.clean_data(item)
                cleaned_data.append(cleaned_item)
            return cleaned_data
        else:
            # Replace unwanted strings in the leaf nodes
            data = data.replace("\n", " ") if isinstance(data, str) else data
            if isinstance(data, str):
                data = data.replace("not specified", "")
                data = data.replace("Not specified", "")
                data = data.replace("not provided", "")
                data = data.replace("Not provided", "")
                data = data.replace("not available", "")
                data = data.replace("Not available", "")
                data = data.replace("not mentioned", "")
                data = data.replace("Not mentioned", "")
                data = data.replace("empty string", "")
                # Remove any sentence that begins with "The provided"
                data = data.replace(data, "") if "The provided" in data else data
                # Remove additional blank spaces
                data = " ".join(data.split())
            return data


    def detect_header_font_properties(self, spans):
        """
        Detects the font properties (size, name, flags) of the first span whose text matches any of the pre-defined headers.

        Parameters:
        - spans (list): A list of text span dictionaries containing 'font_size', 'font_flags', 'font_name', and 'text' information.
        - pre_defined_headers (list): A list of header strings to match against.

        Returns:
        - tuple or None: The detected header font size, flags, and name, or None if no matching header is found.
        """
        # Preprocess the pre_defined_headers to match the formatting of processed span texts

        for span in spans:
            processed_text = self.remove_special_characters(span["text"]).strip().lower()
            if processed_text in pre_defined_headers:
                return (span['font_size'], span['font_flags'], span['font_name'])

        return None  # No matching header found


    def extract_spans_from_pdf(self, pdf_path):
        """
        Extracts all text spans from a PDF file and returns the page width.

        Parameters:
        - pdf_path (str): Path to the PDF file.

        Returns:
        - tuple:
            - spans (list): A list of spans, where each span contains 'font_size', 'font_flags',
            'font_name', 'x', 'y', and 'text_width'.
            - page_width (float): The width of the page.
        """
        import fitz  # Ensure you have PyMuPDF installed: pip install PyMuPDF

        doc = fitz.open(pdf_path)
        spans = []

        # Assume all pages have the same width; get page_width from the first page
        first_page = doc.load_page(0)
        page_rect = first_page.rect
        page_width = page_rect.width

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")['blocks']

            for block in blocks:
                if 'lines' in block:  # Check if 'lines' exists in the block
                    for line in block['lines']:
                        for span in line['spans']:
                            # Calculate text width from the bounding box
                            text_width = span['bbox'][2] - span['bbox'][0]
                            spans.append({
                                'font_size': span['size'],
                                'font_flags': span['flags'],
                                'font_name': span['font'],
                                'x': span['bbox'][0],  # x-coordinate of the text start
                                'y': span['bbox'][1],
                                'text_width': text_width,
                                'text': span['text']
                            })

        return spans, page_width


    def detect_split_based_on_header_x_positions(self, spans, page_width,
                                                x_difference_threshold=0.2,
                                                min_proportion=0.2, max_proportion=0.5):
        """
        Detects if a resume has a vertical split based on the adjusted x-positions of headers.

        Parameters:
        - spans (list): A list of text span dictionaries with 'font_size', 'font_flags',
        'font_name', 'x', 'y', and 'text_width' keys.
        - page_width (float): The width of the page.
        - x_difference_threshold (float): The minimum normalized difference in x-positions
        to consider the content split by columns (default 0.2).
        - min_proportion (float): The minimum proportion of headers on one side to consider
        it a split (default 0.2).
        - max_proportion (float): The maximum proportion of headers on one side to consider
        it a split (default 0.5).

        Returns:
        - bool: True if a vertical split is detected based on x-positions of headers, False otherwise.
        """
        # Step 1: Detect the header font properties
        header_font_properties = self.detect_header_font_properties(spans)
        if not header_font_properties:
            return False  # No headers found

        # Step 2: Collect adjusted x-positions of headers
        header_x_positions = []
        for span in spans:
            if (abs(span['font_size'] - header_font_properties[0]) <= 0.2 and
                span['font_flags'] == header_font_properties[1] and
                span['font_name'] == header_font_properties[2]):
                # Calculate center x-position
                x_center = span['x'] + (span['text_width'] / 2)
                # Normalize x-position relative to page width
                x_normalized = x_center / page_width
                header_x_positions.append(x_normalized)

        if len(header_x_positions) < 2:
            return False  # Not enough headers to analyze

        # Step 3: Use clustering to identify potential splits

        X = np.array(header_x_positions).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_.flatten()

        # Step 4: Evaluate cluster separation
        separation = abs(centers[0] - centers[1])
        if separation < x_difference_threshold:
            return False  # Clusters are too close; likely not a split

        # Step 5: Check the proportions
        counts = np.bincount(labels)
        total_headers = len(header_x_positions)
        smaller_group_size = counts.min()
        proportion = smaller_group_size / total_headers

        # Step 6: Check if the proportion is between min_proportion and max_proportion
        if min_proportion <= proportion <= max_proportion:
            return True  # Vertical split detected

        return False  # No vertical split detected


    def convert_docx_to_pdf(self, docx_path: str, output_dir: str = None) -> str:
        """
        Converts a DOCX file to PDF using LibreOffice's 'soffice' command.

        :param docx_path: Path to the input DOCX file.
        :param output_dir: Directory where the PDF will be saved. Defaults to the DOCX file's directory.
        :return: Path to the converted PDF file.
        :raises: RuntimeError if conversion fails.
        """
        if output_dir is None:
            output_dir = os.path.dirname(docx_path)
        
        soffice_path = '/opt/render/project/.render/libreoffice/program/soffice'
        
        if not os.path.isfile(soffice_path):
            raise RuntimeError(f"'soffice' executable not found at {soffice_path}")
        
        try:
            # Execute the 'soffice' command with the full path
            subprocess.run([
                soffice_path,
                '--headless',
                '--convert-to', 'pdf',
                docx_path,
                '--outdir', output_dir
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Determine the name of the converted PDF
            pdf_filename = Path(docx_path).with_suffix('.pdf').name
            pdf_path = Path(output_dir) / pdf_filename
            
            if not pdf_path.exists():
                raise RuntimeError(f"Conversion failed, PDF not found at {pdf_path}")
            
            return str(pdf_path)
        
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode().strip()
            raise RuntimeError(f"LibreOffice conversion failed: {error_msg}") from e


    # Function to extract information from passed file
    def extract_info(self, file, translate, return_summary=False, target_language="EN-US"):
        # Save the uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        with open(file.filename, "wb") as f:
            f.write(file.file.read())

        
        # Initialize variables
        cv_input = None
        extracted_info = {}


        if file.filename.endswith(".pdf"):
            cv_input = file
            extracted_info = self.extract_infos_from_cv_v2(cv_input, return_summary, file.filename)

        elif file.filename.endswith(".docx"):
            # Convert DOCX to PDF using portable LibreOffice
            pdf_path = self.convert_docx_to_pdf(file.filename, output_dir=temp_dir)

            # Read the converted PDF into bytes
            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
            
            # Create a new UploadFile-like object for the PDF
            pdf_filename = os.path.basename(pdf_path)
            pdf_upload_file = UploadFile(filename=pdf_filename, file=io.BytesIO(pdf_bytes))

            extracted_info = self.extract_infos_from_cv_v2(pdf_upload_file, return_summary, file.filename)
        else:
            return {"error": "The uploaded file is not a PDF or a DOCX file"}


        # Pass cv_input to extract_infos_from_cv_v2
        # extracted_info = self.extract_infos_from_cv_v2(cv_input, return_summary, file.filename)

        # Clean up the extracted information
        cleaned_info = {}
        for key, value in extracted_info.items():
            cleaned_info[key] = self.clean_data(value)

        # Translate the extracted information to the target language
        if translate:
            cleaned_info = self.translator.translate_JSON(cleaned_info, target_language)

        # Remove the uploaded file
        os.remove(file.filename)

        # Remove images and directory if cv_input was images
        if isinstance(cv_input, list):
            for image_path in cv_input:
                os.remove(image_path)
            os.rmdir(screens_dir)

        return cleaned_info


    # Function to extract a summary from the passed file
    def extract_summary(self, cv_text, target_language="EN-US"):
        # Save the uploaded file temporarily
        #with open(file.filename, "wb") as f:
        #    f.write(file.file.read())

        # Check if the uploaded file is a PDF
        #if file.filename.endswith(".pdf"):  
            # Extract text from the PDF
            #cv_text = self.get_pdf_text(file.filename)
            
        #elif file.filename.endswith(".docx"):
            # Extract text from the DOCX
            #cv_text = self.get_docx_text(file.filename)
            
        #else:
        #    return {"error": "The uploaded file is not a PDF or a Docx file"}

        results = {}
        #print(cv_text)
        #print(type(cv_text))
        # Process and Extract exactly information from the CV
        results["summary"] = self.extract_summary_from_cv_v2(cv_text, target_language)
            
        # Translate the extracted information to the target language
        # if translate:
        #     results["summary"] = self.translator.translate_text(results["summary"], target_language)

        # Remove the uploaded file from the root directory
        # os.remove(file.filename)

        return results


    async def _translate_chunk_async(self, data_chunk: dict, target_language: str) -> dict:
        """
        Async version of chunk translation. 
        The chain(...) call is assumed to be async, so we await it.
        """
        email = data_chunk.get("email")
        json_string = json.dumps(data_chunk, ensure_ascii=False)
        
        prompt = f"""
        Translate the following JSON data to {target_language}, only values should be translated, 
        keys should remain the same, and values containing numbers should be represented as strings 
        in the json. Answer only with the json, nothing else should be added, no comments no nothing, 
        AND DON'T USE AN EDITOR:

        {json_string}
        """
        
        # Await the LLM chain (assuming chain is async)
        response = await asyncio.to_thread(
            self.chain,
            {
                "context": "JSON Translation",
                "question": prompt,
                "input_documents": []
            },
            return_only_outputs=True
        )
        
        output_text = response["output_text"]
        
        # Remove markdown fences if present
        if "```" in output_text:
            output_text = output_text.replace("```json", "").replace("```", "")

        
        try:
            result = json.loads(output_text.strip())
        except json.JSONDecodeError:
            result = {}
        return result


    async def translate_json(self, data: dict, target_language: str) -> dict:
        """
        The main async method that decides whether to split the data or not, 
        and if split, sends both chunks to Gemini in parallel.
        """
        json_string = json.dumps(data, ensure_ascii=False)
        input_tokens = self.count_tokens(json_string)
        
        threshold = 5000
        if input_tokens <= threshold:
            # Just do a single async call
            return await self._translate_chunk_async(data, target_language)
        else:
            # Split the JSON if it has a large 'work' list
            if 'work' in data and isinstance(data['work'], list) and len(data['work']) > 1:
                work_list = data['work']
                half = len(work_list) // 2

                # Prepare chunk 1
                chunk1 = copy.deepcopy(data)
                chunk1['work'] = work_list[:half]

                # Prepare chunk 2
                chunk2 = copy.deepcopy(data)
                chunk2['work'] = work_list[half:]

                # Fire off both translations concurrently
                task1 = asyncio.create_task(self._translate_chunk_async(chunk1, target_language))
                task2 = asyncio.create_task(self._translate_chunk_async(chunk2, target_language))

                # Wait until both complete
                translated_part1, translated_part2 = await asyncio.gather(task1, task2)
                # Merge the 'work' arrays from both parts
                if 'work' not in translated_part1:
                    translated_part1['work'] = []
                if 'work' in translated_part2:
                    translated_part1['work'].extend(translated_part2['work'])

                return translated_part1
            else:
                # If 'work' doesn't exist or isn't a large list, fallback to single chunk
                return await self._translate_chunk_async(data, target_language)


    def process_translation(self, combined_prompt, output_text, email):
        translation_input_tokens = self.count_tokens(combined_prompt.strip())
        translation_output_tokens = self.count_tokens(output_text.strip())
        estimated_cost_of_translation_output_tokens = self.calculate_cost(translation_output_tokens, False)
        estimated_cost_of_translation_input_tokens = self.calculate_cost(translation_input_tokens, True)
        with self.lock:
            with open(self.report_file, "r") as f:
                reports = json.load(f)
            for report in reports:
                if report["email"] == email and report["données"]["nombre_de_tokens_d'entrée_traduction"] == 0:
                    report["données"]["nombre_de_tokens_d'entrée_traduction"] = translation_input_tokens
                    report["données"]["coût_estimé_des_tokens_d'entrée_traduction"] = estimated_cost_of_translation_input_tokens
                    report["données"]["nombre_de_tokens_de_sortie_traduction"] = translation_output_tokens
                    report["données"]["coût_estimé_des_tokens_de_sortie_traduction"] = estimated_cost_of_translation_output_tokens
                    report["données"]["nombre_total_de_tokens"] += translation_input_tokens + translation_output_tokens
                    report["données"]["coût_total_estimé"] += estimated_cost_of_translation_input_tokens + estimated_cost_of_translation_output_tokens
                    break
            with open(self.report_file, "w") as f:
                json.dump(reports, f, indent=4)




