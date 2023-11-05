Follow the below steps:

## Mysql

# 1. Select features from Mimic-III dataset, and save rows whose measure type is in these features to ../mimicdata/biomarks.csv
select SUBJECT_ID, HADM_ID, ITEMID, VALUE, VALUEUOM, CHARTTIME from CHARTEVENTS where ITEMID IN 
(791, 1525, 220615, 224643, 225310, 220180, 8555, 220051, 8368, 8441, 8440, 220621, 225664, 811, 807, 226537, 1529, 
211, 20045, 220228, 814, 456, 220181, 224, 225312, 220052, 52, 6702, 224322, 646, 834, 220177, 220227, 828, 227457, 833,
224422, 618, 220210, 224689, 614, 651, 224690, 615, 626, 442, 227243, 224167, 220179, 225309, 6701, 220050, 51, 455,
223761, 671, 676, 679, 678, 223762, 861, 1542, 220546, 1127, 789, 3748, 1524, 220603, 3385, 3512, 79, 224144,3799,3834) 
INTO OUTFILE '/home/kelon/code/CVD/mimicdata/biomarks.csv';

# 2. extract NOTEENENTS
select SUBJECT_ID, HADM_ID, CHARTDATE, CHARTTIME, STORETIME, TEXT from NOTEEVENTS INTO OUTFILE '/home/kelon/code/CVD/mimicdata/noteevents.csv';

# 3. select GENDER，DOB(date of birth) from PATIENTS table and left join them to ADMISSIONS table
SELECT ADMISSIONS.SUBJECT_ID, ADMISSIONS.HADM_ID, ADMISSIONS.ADMITTIME, ADMISSIONS.DISCHTIME, 
ADMISSIONS.HOSPITAL_EXPIRE_FLAG, ADMISSIONS.ETHNICITY, ADMISSIONS.MARITAL_STATUS, PATIENTS.GENDER, PATIENTS.DOB FROM ADMISSIONS
LEFT JOIN PATIENTS ON ADMISSIONS.SUBJECT_ID = PATIENTS.SUBJECT_ID
ORDER BY ADMISSIONS.HADM_ID
INTO OUTFILE '/home/kelon/code/CVD/mimicdata/admissions.csv';

# 4. copy these two files PROCEDURES_ICD and DIAGNOSES_ICD, to CVD/mimicdata/

After following steps 1, 2, 3, 4, the mimicdata director should look like this:

mimicdata
--P18
  --biomarks.csv
DIAGNOSES_ICD.csv
PROCESURES_ICD.csv
NOTEEVENTS.csv
ICD9_descriptions.csv
ICD9_cardiovascular.csv
admissions.csv


## Manual
Identify ICD-9 code of specify disease

## Juplyer file

# 1. run extract_mimiciii.ipynb
Pre-process time-series data and save them to 'Multi_data/Your_data_name/rawdata/'

# 2. run reformat_text.sh
Pre-process text data(tokenize text to pieces of sentences and only extract one kind of text data such as Discharge_summary)

# 3. run GenerateData.ipynb
Get input of time-series model and language model, labels and splits(train, dev, test)

After following steps 1, 2, 3, the Multi_data director should look like this:

Multi_data
--P18_ECER
  --rawdata
    --set
      100001.txt
      100002.txt
      ...
    biomarks.csv
    biomarks_plus.csv
    Noteevents.csv
    ECER.csv
    Outcomes.csv
  --splits
    1_fold.npy
    2_fold.npy
    ...
  static_params.npy
  ts_params.npy
  P_list.npy
  df_outcomes.npy
  ICD_code.csv
  ECER.csv
  PTdict_list.npy
