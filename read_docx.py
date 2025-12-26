import sys
from docx import Document

# Set UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8')

doc = Document('Moatter Project Piplines with Code.docx')

for para in doc.paragraphs:
    print(para.text)
