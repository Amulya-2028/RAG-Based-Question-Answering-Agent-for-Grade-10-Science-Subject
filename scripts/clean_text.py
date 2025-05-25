import re

def clean_extracted_text(input_file, output_file):
    """Clean up extracted text by removing repetitions, fixing formatting, and cleaning noise."""
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        lines = infile.readlines()
        cleaned_lines = []

        for line in lines:
            # Reduce repeated characters
            line = re.sub(r'(.)\1{2,}', r'\1', line)

            # Remove excessive dots, symbols, or blank lines
            line = re.sub(r'[.]{2,}', '.', line)
            line = re.sub(r'[^a-zA-Z0-9\s.,]', '', line)

            # Filter out short, meaningless lines
            if len(line.strip()) > 2:
                cleaned_lines.append(line.strip())

        outfile.write("\n".join(cleaned_lines))

    print("Text cleaning completed!")

if __name__ == "__main__":
    input_file = r"path_of_t1.txt"
    output_file = r"Cpath_of_cleaned_t1.txt"
    clean_extracted_text(input_file, output_file)
