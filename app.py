import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
import tempfile
from utils import get_model_response

def main():
    st.title('Data Analysis Bot')
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(uploaded_file.getvalue())
            temp_file_path = temp.name

            csv_loader = CSVLoader(file_path = temp_file_path, encoding = 'utf-8', delimiter = ',')

            data = csv_loader.load()

            user_input = st.text_input('Your Message: ')
            print(user_input)

            if user_input:
                response = get_model_response(data, user_input)
                st.write(response)


if __name__ == '__main__':
    main()