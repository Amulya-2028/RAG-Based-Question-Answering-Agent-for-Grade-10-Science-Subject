import os
import PyPDF2
import pdfplumber

def extract_text_with_fallback(pdf_path):
    """Extracts text from a PDF using pdfplumber, falling back to PyPDF2."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text.strip() + "\n"
            return text.strip()
    except Exception as e:
        print(f"Error with pdfplumber on {pdf_path}: {e}")
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text.strip() + "\n"
                return text.strip()
        except Exception as fallback_error:
            print(f"PyPDF2 also failed on {pdf_path}: {fallback_error}")
            return ""

def extract_text_from_pdfs(pdf_folder, output_file):
    """Processes all PDFs in a folder and writes extracted text to a file."""
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for pdf_file in sorted(os.listdir(pdf_folder)):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join(pdf_folder, pdf_file)
                print(f"Processing: {pdf_file}")
                text = extract_text_with_fallback(pdf_path)
                if text:
                    outfile.write(text + "\n")
                else:
                    print(f"Skipping {pdf_file} due to extraction issues.")

if __name__ == "__main__":
    pdf_folder = r"Your_pdfs_path"
    output_file = r"Your_processed_file_path"
    extract_text_from_pdfs(pdf_folder, output_file)
    print("Text extraction completed!")
