# TalentProCV-back

## Setup Instructions


1. Run the following commands in your terminal:

    ```shell
    conda create -p venv python==3.10 (has to be 3.9 or greater)
    conda activate venv/
    ```


2. Obtain a Google API Key and set it as the value for the GOOGLE_API_KEY variable in a ".env" file:

    ```GOOGLE_API_KEY="YOUR_API_KEY"```
    

2. Obtain a DeepL API Key and set it as the value for the DEEPL_API_KEY variable in a ".env" file:

    ```DEEPL_API_KEY="YOUR_API_KEY"```


3. Install the required libraries by running the following command:

    ```shell
    pip install -r requirements.txt
    ```


4. Download your Firebase service account private key as a JSON named ```"serviceAccount.json"``` from the Google Cloud Console and put the JSON file in the root folder.


5. Run the following command to run the API and communicate with it using the Frontend:

    ```shell
    uvicorn main:app --reload
    ```
