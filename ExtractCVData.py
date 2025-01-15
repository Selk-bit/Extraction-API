import google.generativeai as genai
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
from langdetect import detect
# Load the environment variables
load_dotenv()

class ExtractCVInfos:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.chain = self.get_conversational_chain()
        self.translator = Translator()

    def get_pdf_text(self, pdf_path):
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

    def get_docx_text(self, docx_path):
        text = textract.process(docx_path, method='docx').decode('utf-8')
        return text

    # Function to load the conversational chain
    def get_conversational_chain(self):
        genai.configure(api_key=self.api_key)

        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=8192)

        prompt_template = """
        Answer the question as detailed as possible from the provided context, and make sure to provide all the details,
        if the answer is not in the provided context just give an empty string "", don't provide the wrong answer nor say that the answer is not provided in the context, 
        and do not add any extra characters other than the ones demanded. Make sure the answer is provided in the way that it is asked in the question.
	Use as much processing power as you can to answer fast.
        Context :\n {context}?\n
        Question: \n {question}\n
	DON'T ANSWER WITH A JSON FORMAT IN A TEXT EDITOR, BUT RATHER, ANSWER WITH THE FOLLOWING FORMAT, AND KEEP TITLES
        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

    import json

    def extract_infos_from_cv_v2(self, cv_text, return_summary=False):
        extracted_info = {}
        # Construct a single prompt with all the extraction instructions
        combined_prompt = f"""
        Extract the following information from the CV text as they are without any translation, following this format as it is:

        1. Title: Extract the current occupation of the candidate (Remove unnecessary spaces within words if found, and leave necessary spaces, and correct capitalization).
        2. Name: Extract the name of the candidate (Remove unnecessary spaces within the name if found, and leave only spaces seperating first name from middle name, if there was a middle name, from last name, and correct capitalization).
        3. Email: Extract the email of the candidate.
        4. Phone: Extract the phone number of the candidate.
        5. Age: Extract the age of the candidate, and write the number only.
        6. City: Extract the city of the candidate.
        7. Work Experiences: Extract all work experiences in JSON format as a list containing job_title, company_name, responsibilities, city, start_date, and end_date. (Make responsibilities a string always, and remove any bullet points symbols, dots symbols, or numbering symbols, from the string, and replace them with commas or full stops as you see fit)
        8. Educations: Extract all educations and formations in JSON format as a list containing degree, institution, start_year, and end_year.
        9. Languages: Extract all spoken languages (non-programming) in JSON format as a list containing language and level, and translate them to language of the CV (use work responsibilities to detect which language the cv is written in).
        10. Skills: Extract all technical skills (non-social) and programming languages in JSON format as a list containing skill and level.
        11. Interests: Extract all interests/hobbies in JSON format as a list containing interest.
        12. Social Skills: Extract all soft skills (social, communication, etc.) in JSON format as a list of objects, each object containing "skill" as a key, with the skill as the value.
        13. Certifications: Extract all certifications in JSON format as a list containing certification, institution, link, and date, and translate certification to language of the CV (use work responsibilities to detect which language the cv is written in).
        14. Projects: Extract all projects in JSON format as a list containing project_name, description, start_date, and end_date.
        15. Volunteering: Extract all volunteering experiences in JSON format as a list containing organization, position, description, start_date, and end_date.
        16. References: Extract all references in JSON format as a list containing name, position, company, email, and phone (do not include candidate's own contacts).
        17. Headline: Extract the current occupation of the candidate, if there wasn't, deduce it from other presented info (an example of the oppucation would be "web developer"). 
        18. Summary: If a summary exists in the CV already, extract it, you can find it either at the beginning or at the end, take the longest one. (if no summary is found is CV data, then generate one yourself based on data you have, it should be written in the same language)
        CV Text:
        {cv_text}
        """

        try:
            # Make a single call with the combined prompt
            response = self.chain({"context": "CV Extraction", "question": combined_prompt, "input_documents": []},
                                  return_only_outputs=True)
            language = None
            if response:
                response_text = response["output_text"].strip()
                # Define the labels we expect in the response
                labels = {
                    "1. Title:": "title",
                    "2. Name:": "name",
                    "3. Email:": "email",
                    "4. Phone:": "phone",
                    "5. Age:": "age",
                    "6. City:": "city",
                    "7. Work Experiences:": "work",
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

                # Extract data based on the expected labels
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
                                    temp_work = json.loads(section_text, strict=False)
                                    for key_work, work in enumerate(temp_work):
                                        temp_work[key_work]["responsibilities"] = work["responsibilities"].replace("\n", ", ").replace("-", "").replace("#RESP#", ".\n\n-") if work["responsibilities"] else ""
                                        if temp_work[key_work]["responsibilities"] and not language:
                                            language = detect(temp_work[key_work]["responsibilities"])
                                    section_text = json.dumps(temp_work)
                                extracted_info[key] = json.loads(section_text.replace("\n", ""))
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
                            except json.decoder.JSONDecodeError as e:
                                print(e)
                                print(key)
                                extracted_info[key] = []  # Handle JSON decode error
                        else:
                            if key == "email":
                                extracted_info[key] = section_text.replace("\n", "").replace(" ", "")
                                extracted_info[key] = re.sub(r'^[\W_]+', '', extracted_info[key]).lower()
                            else:
                                extracted_info[key] = section_text.replace("\n", "") if section_text.replace("\n", "") != "" else None
        except genai.types.generation_types.StopCandidateException as e:
            print("Safety rating issue detected:", e)
            return extracted_info

        # Handle the summary extraction separately
        if return_summary:
            summary_prompt = f"""
            Generate a summary that describes the extracted information from the CV based on the text of the CV. Aim for a well-structured and captivating summary that reflects the essence of the candidate's capabilities and aspirations, and do not exceed 200 characters.

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
        return extracted_info

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
        prompt = f"""Generate a summary that describes the following information extracted from a CV. Aim for a well-structured, captivating summary that reflects the essence of the candidate's abilities and aspirations, without exceeding 300 characters. The summary should be in {lang}, written in one paragraph. Here are the CV data, make sure to only answer with the summary alone, NOTHING ELSE SHOULD BE ADDED, NO LABLES, NO COMMENTS:\n\n""" + cv_text + "\n"
        with open("myPrompt.txt", "w+") as f:
            f.write(prompt)
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

    # Function to extract information from passed file
    def extract_info(self, file, translate, return_summary=False, target_language="EN-US"):
        # Save the uploaded file temporarily
        with open(file.filename, "wb") as f:
            f.write(file.file.read())

        # Check if the uploaded file is a PDF
        if file.filename.endswith(".pdf"):
            # Extract text from the PDF
            cv_text = self.get_pdf_text(file.filename)

        elif file.filename.endswith(".docx"):
            # Extract text from the DOCX
            cv_text = self.get_docx_text(file.filename)

        else:
            return {"error": "The uploaded file is not a PDF or a Docx file"}

        # Process and Extract information from the CV
        extracted_info = self.extract_infos_from_cv_v2(cv_text, return_summary)

        # Get the extraction rate of the CV
        extraction_rate = self.get_extraction_rate(cv_text, extracted_info)

        # Clean up the extracted information
        cleaned_info = {}
        for key, value in extracted_info.items():
            cleaned_value = self.clean_data(value)
            cleaned_info[key] = cleaned_value

        # Translate the extracted information to the target language
        if translate:
            cleaned_info = self.translator.translate_JSON(extracted_info, target_language)

        # Remove the uploaded file from the root directory
        os.remove(file.filename)

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



    def translate_json(self, data, target_language):
        json_string = json.dumps(data)
        prompt = f"""
        Translate the following JSON data to {target_language}, only values should be translated, keys should remain the same, and values containing numbers should be represented as strings in the json. Answer only with the json, nothing else should be added, no comments no nothing, AND DON'T USE USING AN EDITOR:
        {json_string}
        """
        response = self.chain({"context": "JSON Translation", "question": prompt, "input_documents": []}, return_only_outputs=True)
        output_text = response["output_text"]
        if "```" in output_text:
            output_text = output_text.split("```")[1]
            output_text = output_text.split("json", 1)[1] if output_text.startswith("json") else output_text
        #with open("Cv.txt", "w+") as f:
        #    f.write(output_text)
        if response:
            return json.loads(output_text.strip())
        else:
            return {}
