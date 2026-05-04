"""
Parse NLT Bible PDFs and prepare for Pinecone upload.
Extracts individual verses and groups them for embedding.
"""

import json
import re
from pathlib import Path
import PyPDF2

# Bible book abbreviations mapping
BOOK_ABBREV = {
    'Gen': 'Genesis', 'Ex': 'Exodus', 'Lev': 'Leviticus', 'Num': 'Numbers',
    'Deut': 'Deuteronomy', 'Josh': 'Joshua', 'Judg': 'Judges', 'Ruth': 'Ruth',
    '1 Sam': '1 Samuel', '2 Sam': '2 Samuel', '1 Ki': '1 Kings', '2 Ki': '2 Kings',
    '1 Chr': '1 Chronicles', '2 Chr': '2 Chronicles', 'Ezra': 'Ezra', 'Neh': 'Nehemiah',
    'Est': 'Esther', 'Job': 'Job', 'Ps': 'Psalms', 'Prov': 'Proverbs',
    'Eccl': 'Ecclesiastes', 'Song': 'Song of Solomon', 'Isa': 'Isaiah', 'Jer': 'Jeremiah',
    'Lam': 'Lamentations', 'Ezek': 'Ezekiel', 'Dan': 'Daniel', 'Hosea': 'Hosea',
    'Joel': 'Joel', 'Amos': 'Amos', 'Obad': 'Obadiah', 'Jonah': 'Jonah',
    'Mic': 'Micah', 'Nah': 'Nahum', 'Hab': 'Habakkuk', 'Zeph': 'Zephaniah',
    'Hag': 'Haggai', 'Zech': 'Zechariah', 'Mal': 'Malachi', 'Matt': 'Matthew',
    'Mark': 'Mark', 'Luke': 'Luke', 'John': 'John', 'Acts': 'Acts',
    'Rom': 'Romans', '1 Cor': '1 Corinthians', '2 Cor': '2 Corinthians',
    'Gal': 'Galatians', 'Eph': 'Ephesians', 'Phil': 'Philippians', 'Col': 'Colossians',
    '1 Thess': '1 Thessalonians', '2 Thess': '2 Thessalonians', '1 Tim': '1 Timothy',
    '2 Tim': '2 Timothy', 'Titus': 'Titus', 'Philem': 'Philemon', 'Heb': 'Hebrews',
    'James': 'James', '1 Pet': '1 Peter', '2 Pet': '2 Peter', '1 John': '1 John',
    '2 John': '2 John', '3 John': '3 John', 'Jude': 'Jude', 'Rev': 'Revelation',
    'Phi': 'Philippians'  # Handle typo in PDFs
}

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def parse_verses(text):
    """
    Parse verses from text.
    Format: "Gen 1:1 In the beginning God created..."
    """
    verses = []
    
    # Pattern to match: Book Chapter:Verse Text
    # This is a flexible regex that handles various book name formats
    pattern = r'([A-Z][a-z]+\s*\d*)\s+(\d+):(\d+)\s+(.+?)(?=\n[A-Z][a-z]+\s*\d*\s+\d+:\d+|\Z)'
    
    matches = re.finditer(pattern, text, re.DOTALL)
    
    for match in matches:
        book = match.group(1).strip()
        chapter = match.group(2)
        verse = match.group(3)
        text_content = match.group(4).strip()
        
        # Clean up the verse text
        text_content = ' '.join(text_content.split())
        
        if text_content:
            verses.append({
                'book': book,
                'chapter': int(chapter),
                'verse': int(verse),
                'text': text_content,
                'reference': f"{book} {chapter}:{verse}",
                'full_text': f"{book} {chapter}:{verse} {text_content}"
            })
    
    return verses

def group_verses_for_embedding(verses, group_size=5):
    """
    Group verses together for embedding.
    This helps with semantic search - verses on same topic grouped together.
    """
    grouped = []
    
    for i in range(0, len(verses), group_size):
        group = verses[i:i+group_size]
        combined_text = " ".join([v['full_text'] for v in group])
        
        grouped.append({
            'verses': group,
            'text': combined_text,
            'book': group[0]['book'],
            'start_ref': group[0]['reference'],
            'end_ref': group[-1]['reference'],
            'count': len(group)
        })
    
    return grouped

def create_embedding_data(grouped_verses):
    """
    Format verses for Pinecone embedding.
    Each entry is a chunk that will be embedded.
    """
    documents = []
    
    for group in grouped_verses:
        doc = {
            'text': group['text'],
            'reference': group['start_ref'],
            'book': group['book'],
            'type': 'bible',
            'verses': [v['reference'] for v in group['verses']]
        }
        documents.append(doc)
    
    return documents

def process_bible_files(pdf_directory='data/NLT_Bible'):
    """
    Process all NLT Bible PDFs in a directory.
    Run from the repo root: python ingestion/bible_parser.py
    """
    print("🚀 Starting Bible PDF parsing...")
    print("="*60)

    out_dir = Path(pdf_directory)
    all_verses = []
    pdf_files = list(out_dir.glob('NLT_*.pdf'))

    print(f"Found {len(pdf_files)} PDF files\n")

    for pdf_file in sorted(pdf_files):
        print(f"Processing: {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)
        verses = parse_verses(text)
        all_verses.extend(verses)
        print(f"  ✅ Extracted {len(verses)} verses")

    print(f"\n📊 Total verses extracted: {len(all_verses)}")
    print("="*60)

    print("\nGrouping verses for embedding...")
    grouped = group_verses_for_embedding(all_verses, group_size=5)
    print(f"✅ Created {len(grouped)} verse groups")

    print("\nPreparing embedding data...")
    documents = create_embedding_data(grouped)

    print("\nSaving to JSON files...")

    with open(out_dir / 'bible_verses.json', 'w', encoding='utf-8') as f:
        json.dump(all_verses, f, ensure_ascii=False, indent=2)
    print("✅ bible_verses.json saved")

    with open(out_dir / 'bible_grouped.json', 'w', encoding='utf-8') as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)
    print("✅ bible_grouped.json saved")

    with open(out_dir / 'bible_for_embedding.json', 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    print("✅ bible_for_embedding.json saved")

    print("\n" + "="*60)
    print("✅ BIBLE PARSING COMPLETE")
    print("="*60)
    print(f"Total verses: {len(all_verses)}")
    print(f"Verse groups: {len(grouped)}")
    print(f"Embedding documents: {len(documents)}")
    print("\nNext steps:")
    print("1. Run: python ingestion/upload_bible.py")

if __name__ == '__main__':
    process_bible_files()