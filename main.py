from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Request, Body, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
import os
from ExtractCVDataV3 import ExtractCVInfos
import uvicorn
import asyncio
from concurrent.futures import ProcessPoolExecutor
import logging
import sys
import langdetect
import json
import uuid
import shutil
import tempfile
import magic
from starlette.background import BackgroundTask
from datetime import datetime
import uuid
from fastapi import Depends
from pdf2docx import Converter
from starlette.background import BackgroundTask
import magic  # For reliable file type detection
from concurrent.futures import ThreadPoolExecutor

# Configure the logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI()
TEMP_DIR = "./tmp/"

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Content-Type"],
)
    
    
# Initialize the ExtractCVInfos class
cv = ExtractCVInfos()

# Define supported MIME types
SUPPORTED_MIME_TYPES = {
    "application/pdf": "pdf",
}

# def process_file(file, translate, target_language):
#    return cv.extract_info(file, translate, return_summary=False, target_language=target_language)

def process_file(file, translate, target_language):
    extracted_info = cv.extract_info(file, translate, return_summary=False, target_language=target_language)
    
    # Check if "name", "email", and "work" keys are present and not empty
    required_keys = ["name", "email", "work"]
    if not all(key in extracted_info for key in required_keys):
        # Return an object with only the "filename" and "error" keys
        return {
            "filename": file.filename,
            "error": (
                f"An error occurred while extracting information from '{file.filename}'. "
                "Please ensure the file is a valid CV and try again."
            )
        }
    else:
        return extracted_info


def process_translation(data, target_language):
    return cv.translate_json(data, target_language)

# Custom exception handler for internal server errors
@app.exception_handler(Exception)
async def internal_server_error_handler(request: Request, exc: Exception):
    logger.error(f"Internal Server Error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )

# Custom exception handler for HTTP exceptions
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(f"HTTP Exception: {exc.detail}", exc_info=True)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

# Custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation Error: {exc.errors()}", exc_info=True)
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

# Endpoint to handle file upload and extraction with summary for multiple files
@app.post("/extract-with-summary")
async def extract_with(files: List[UploadFile] = File(...), translate: bool = False, target_language: str = "EN-US"):
    try:
        # List to store extracted information from each CV
        extracted_info_list = []

        for file in files:

            # Extract information from the CV
            extracted_info = cv.extract_info(file, translate, return_summary=True, target_language=target_language)

            # Append cleaned info to the list
            extracted_info_list.append(extracted_info)
        
        # Return the extracted information for all files as JSON
        return JSONResponse(content=extracted_info_list)

    except Exception as e:
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)



# Endpoint to handle file upload and extraction without summary for multiple files
@app.post("/extract")
async def extract_without(files: List[UploadFile] = File(...), translate: bool = False, target_language: str = "EN-US", return_summary: bool = False):
    try:
        extracted_info_list = []

        # Create a ProcessPoolExecutor with a maximum of 10 processes
        with ThreadPoolExecutor(max_workers=10) as executor:
            loop = asyncio.get_event_loop()

            # Process files in batches of 10
            for i in range(0, len(files), 10):
                batch = files[i:i+10]
                tasks = [loop.run_in_executor(executor, process_file, file, translate, target_language) for file in batch]
                results = await asyncio.gather(*tasks)
                extracted_info_list.extend(results)

        # Return the extracted information for all files as JSON
        return JSONResponse(content=extracted_info_list)

    except Exception as e:
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)


# Endpoint to handle summary extraction
#@app.post("/extract-only-summary")
#async def extract_summary(files: List[UploadFile] = File(...), translate: bool = False, lang: str = Form("EN-US")):
#    try:
        # List to store extracted information from each CV
#        extracted_info_list = []

#        for file in files:
            # Extract information from the CV
#            extracted_info = cv.extract_summary(file, translate, target_language=lang)

            # Append cleaned info to the list
#            extracted_info_list.append(extracted_info)

        # Return the extracted information for all files as JSON
#        return JSONResponse(content=extracted_info_list)

#    except Exception as e:
#        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)
    
    
@app.post("/extract-only-summary")
async def extract_summary(data: str = Form(...), lang: str = Form("EN-US")):
    try:
        # Load the stringified JSON
        json_data = json.loads(data)
        # Remove the "summary" key if it exists
        if "summary" in json_data:
            del json_data["summary"]
        # Stringify the JSON again
        json_data_str = json.dumps(json_data)

        # Pass the stringified JSON to the extract_summary method
        extracted_info = cv.extract_summary(json_data_str, target_language=lang)

        # Return the extracted summary as JSON
        return JSONResponse(content=extracted_info)

    except Exception as e:
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)


# Endpoint to handle translation of JSON data
@app.post("/translate")
async def translate_data(data: dict, target_language: str = "EN-US"):
    try:
        # Translate the data
        translated_data = cv.translator.translate_JSON(data, target_language)

        # Return the translated data as JSON
        return JSONResponse(content=translated_data)

    except Exception as e:
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)


#@app.post("/translate-users")
#async def translate_users(data: Dict[str, List[dict]] = Body(...)):
#    try:
#        translated_results = {}
#        loop = asyncio.get_event_loop()
#        tasks = []
#
#        for user, details in data.items():
#            json_string = json.dumps(details)
#            detected_language = langdetect.detect(json_string)
#            target_language = "EN-US" if detected_language == "fr" else "fr"
#
#            tasks.append(loop.run_in_executor(None, process_translation, details, target_language))
#
#        results = await asyncio.gather(*tasks)
#
#        for idx, (user, _) in enumerate(data.items()):
#            translated_results[user] = {"fr": results[idx] if detected_language == "en" else data[user], 
#                                        "en": results[idx] if detected_language == "fr" else data[user]}
#
#        return JSONResponse(content=translated_results)
#    except Exception as e:
#        logger.error(f"An error occurred: {str(e)}", exc_info=True)
#        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)


class UserData(BaseModel):
    id: str
    resumeData: Dict[str, Any]


@app.post("/translate-user")
async def translate_user(data: dict):
    try:
        # Extract resumeData from the received object
        resume_data = data.get("data")
        # Convert resumeData to a JSON string
        json_string = json.dumps(resume_data)

        # Detect the language of the JSON data
        detected_language = langdetect.detect(json_string)

        # Set the target language
        target_language = "EN-US" if detected_language == "fr" else "fr"

        # Directly call the translation method without using asyncio tasks
        translated_data = cv.translate_json(resume_data, target_language)

        # Prepare the response with the translated data and detected language
        result = {
            "translated": {
                "data": translated_data,
                "lang": "en" if target_language == "EN-US" else "fr"
            }
        }

        # Return the result as a JSON response
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)


@app.post("/translate-users")
async def translate_users(data: List[UserData]):
    try:
        translated_results = []
        loop = asyncio.get_event_loop()
        tasks = []
        results_mapping = {}

        for user_data in data:
            user_id = user_data.id
            resume_data = user_data.resumeData

            file_path = f"{user_id}.json"

            # Check if the file already exists
            if os.path.exists(file_path):
                # Load data from the existing file
                with open(file_path, 'r') as file:
                    translated_results.append(json.load(file))
            else:
                json_string = json.dumps(resume_data)
                detected_language = langdetect.detect(json_string)
                target_language = "EN-US" if detected_language == "fr" else "fr"

                # Add the translation task
                task = loop.run_in_executor(None, process_translation, resume_data, target_language)
                tasks.append(task)
                results_mapping[task] = {
                    "id": user_id,
                    "original_data": resume_data,
                    "detected_language": detected_language
                }

        # Process translations in parallel
        results = await asyncio.gather(*tasks)

        # Update translated_results with the translations and save to files
        for task, translated_data in zip(tasks, results):
            user_info = results_mapping[task]
            user_id = user_info["id"]
            file_path = f"{user_id}.json"

            if user_info["detected_language"] == "fr":
                result = {
                    "id": user_id,
                    "fr": user_info["original_data"],
                    "en": translated_data
                }
            else:
                result = {
                    "id": user_id,
                    "en": user_info["original_data"],
                    "fr": translated_data
                }

            translated_results.append(result)
            #with open("traduction.txt", 'w') as file:
            #    file.write(result)

            # Save the result to a file
            #with open(file_path, 'w') as file:
            #    json.dump(result, file)

        return JSONResponse(content=translated_results)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)


def cleanup_files(file_paths: list):
    """Background task to clean up temporary files."""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass  # Optionally, log the exception

@app.post("/convert/pdf-to-word")
async def convert_pdf_to_word(file: UploadFile = File(...)):
    # Validate MIME type
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are supported.")
    
    # Read a small portion of the file to confirm MIME type
    file_contents = await file.read(2048)
    file_type = magic.from_buffer(file_contents, mime=True)
    
    if file_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File content does not match PDF format.")
    
    # Reset the file pointer to the beginning after reading
    await file.seek(0)
    
    # Generate unique filenames to avoid collisions
    unique_id = str(uuid.uuid4())
    input_filename = f"{unique_id}.pdf"
    output_filename = f"{unique_id}.docx"
    
    # Create temporary file paths
    temp_dir = tempfile.gettempdir()
    input_path = os.path.join(temp_dir, input_filename)
    output_path = os.path.join(temp_dir, output_filename)
    
    try:
        # Save the uploaded PDF to the temporary directory
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Convert PDF to DOCX
        cv = Converter(input_path)
        cv.convert(output_path, start=0, end=None)
        cv.close()
        
        # Prepare the response with a background task to clean up files
        task = BackgroundTask(cleanup_files, [input_path, output_path])
        
        # Determine the filename for the output
        original_filename = os.path.splitext(file.filename)[0]
        converted_filename = f"{original_filename}.docx"
        
        return FileResponse(
            path=output_path,
            filename=converted_filename,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            background=task
        )
    
    except Exception as e:
        # Clean up the input file in case of an error
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")



def cleanup_files(file_paths: list):
    """Background task to clean up temporary files."""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            # Log the exception if needed
            pass


@app.post("/convert/pdf-to-word")
async def convert_pdf_to_word(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are supported.")
    
    # Generate unique filenames to avoid collisions
    unique_id = str(uuid.uuid4())
    input_path = os.path.join(TEMP_DIR, f"{unique_id}.pdf")
    output_path = os.path.join(TEMP_DIR, f"{unique_id}.docx")
    
    try:
        # Save the uploaded PDF to the temporary directory
        print(input_path)
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Convert PDF to DOCX
        cv = Converter(input_path)
        cv.convert(output_path, start=0, end=None)
        cv.close()
        
        # Return the converted DOCX file
        return FileResponse(
            path=output_path,
            filename=f"{os.path.splitext(file.filename)[0]}.docx",
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")
    finally:
        # Clean up the temporary files
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)


@app.get("/report")
async def get_report():
    try:
        with open("report.json", "r") as f:
            reports = json.load(f)
        return JSONResponse(content=reports)
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to retrieve report: {str(e)}"}, status_code=500)



if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
