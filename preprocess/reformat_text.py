import psycopg2
import pandas as pd
import sys
import os
import spacy
import re
import time
import scispacy
from tqdm import tqdm
from heuristic_tokenize import sent_tokenize_rules 
from spacy.language import Language


#setting sentence boundaries
@Language.component("component")
def sbd_component(doc):
    for i, token in enumerate(doc[:-2]):
        # define sentence start if period + titlecase token
        if token.text == '.' and doc[i+1].is_title:
            doc[i+1].sent_start = True
        if token.text == '-' and doc[i+1].text != '-':
            doc[i+1].sent_start = True
    return doc

#convert de-identification text into one token
def fix_deid_tokens(text):
    #text = re.sub(r"(\d+)\.(?!\d)", r"\1,", text)
    #text = re.sub(r"\. \.+", ".", text)
    text = re.sub(r"\.", ",", text)
    deid_regex  = r"\[\*\*.{0,15}.*?\*\*\]" 
    if text:
        indexes = [m.span() for m in re.finditer(deid_regex,text,flags=re.IGNORECASE)]
    else:
        indexes = []
    len1, len2 = len(text), len(indexes)
    res = ""
    i, j = 0, 0

    if len2 == 0:
        return text
    
    while i < len1 and j < len2:
        start, end = indexes[j][0], indexes[j][1]
        if i < start:
            res += text[i]
            i += 1
        elif start <= i < end:
            i += 1
            pass
        else:
            j += 1
    res += text[i:len1]
    return res
    

def process_section(section, note, processed_sections):
    # perform spacy processing on section
    # print(type(section['sections']))
    #print(section['sections'])
    #print('***************************************')
    processed_section = fix_deid_tokens(section['sections'])
    processed_section = nlp(processed_section)
    processed_sections.append(processed_section)

def process_note_helper(note):
    # split note into sections
    note_sections = sent_tokenize_rules(note)
    # print(len(note_sections))
    processed_sections = []
    section_frame = pd.DataFrame({'sections':note_sections})
    section_frame.apply(process_section, args=(note,processed_sections,), axis=1)
    return (processed_sections)

def process_text(sent, note):
    sent_text = sent['sents'].text
    if len(sent_text) > 0 and sent_text.strip() != '\n':
        if '\n' in sent_text:
            sent_text = sent_text.replace('\n', ' ')
        note['Text'] += sent_text + '\n'  

def get_sentences(processed_section, note):
    # get sentences from spacy processing
    sent_frame = pd.DataFrame({'sents': list(processed_section['sections'].sents)})
    sent_frame.apply(process_text, args=(note,), axis=1)

def process_note(note):
    try:
        note_text = note['Text'] #unicode(note['text'])
        note['Text'] = ''
        processed_sections = process_note_helper(note_text)
        ps = {'sections': processed_sections}
        ps = pd.DataFrame(ps)
        ps.apply(get_sentences, args=(note,), axis=1)
        return note 
    except Exception as e:
        print ('error', e)


if __name__ == '__main__':
    '''
    if len(sys.argv) < 2:
            print('Please specify the note category.')
            sys.exit()
    category = sys.argv[1]
    '''
    # update these constants to run this script
    ts_part = 'P18'
    # 'Discharge_summary'
    NAME = 'ECER'
    category = set(['Radiology', 'Echo', 'ECG'])

    OUTPUT_DIR = '../Multi_data/%s_%s/'%(ts_part, NAME) 
    #this is the path to mimic data if you're reading from a csv. Else uncomment the code to read from database below
    MIMIC_NOTES_FILE = '%srawdata/Noteevents.csv' % OUTPUT_DIR

    
    start = time.time()
    tqdm.pandas()

    print('Begin reading notes')


    # Uncomment this to use postgres to query mimic instead of reading from a file
    # con = psycopg2.connect(dbname='mimic', host="/var/run/postgresql")
    # notes_query = "(select * from mimiciii.noteevents);"
    # notes = pd.read_sql_query(notes_query, con)
    notes = pd.read_csv(MIMIC_NOTES_FILE)
    notes = notes[notes['category'].isin(category)]
    # notes = notes[:20]
   
    print('Number of notes: %d' %len(notes.index))
    notes['ind'] = list(range(len(notes.index)))

    nlp = spacy.load('en_core_sci_md', disable=['tagger','ner',"lemmatizer"])
    nlp.add_pipe("component", before='parser')

    formatted_notes = notes.progress_apply(process_note, axis=1)
    formatted_notes.sort_values(['RecordID','Time'], ascending=True, inplace=True)
    df_grouped = formatted_notes.groupby('RecordID')
    df = pd.DataFrame([])
    
    for (id, df_tmp) in df_grouped:
        text = ""
        for inex, row in df_tmp.iterrows():
            text += row['Text'] + "\n"
        df_tmp.iloc[0, 2] = text
        df = df.append(df_tmp.iloc[0])
        # df = pd.concat([df, df_tmp.iloc[0]], axis=0)
    df.reset_index(drop=True, inplace=True)
    print(df.columns)
    features = ['PatientID','RecordID','Time','Text']
    df = df[features]

    df['PatientID'] = df['PatientID'].astype(int)
    df['RecordID'] = df['RecordID'].astype(int)
    df['Time'] = df['Time'].astype(int)

    dirs = OUTPUT_DIR + 'rawdata/'
    if not os.path.exists(dirs):
        raise FileNotFoundError

    df.to_csv(dirs + NAME + '.csv', index=None)

    end = time.time()
    print (end-start)
    print ("Done formatting notes")
