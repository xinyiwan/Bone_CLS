# Extract clinical information from both DICOM header and LLM results

llm_cli_csv = "/home/ext_xinwan/Bone_AI/BTRecordsLLM/output/clinical_variables/kira-0515-llm-cli.csv"
dcm_batch_1_csv = "/home/ext_xinwan/Bone_AI/output/DCM_Physics/batch_1/dicom_headers_labelled_Mar24.csv"
dcm_batch_2_csv = "/home/ext_xinwan/Bone_AI/output/DCM_Physics/batch_2/batch_2_headers_labelled.csv"


# for dcm header label, we extracted info from colum subject, birthdate,age,gender from both csv file
# we can take the first image of each subject instead of every row
# for llm cli csv, the info is already extracted

"""
pid,symptoms,history_of_neoplasm,suspected_metastasis,skeletal_location,location_within_bone
BONE_AI_1255,[],No,No,"Tibia, left",Not specified
BONE_AI_724,['Pain'],No,No,"Proximal tibia, right",Epiphysis and metaphysis
BONE_AI_858,[],No,No,"Pelvis, left",Not specified
BONE_AI_244,[],No,No,"Ilium, left",Not applicable
"""

# we want to combine all subjects from each file is included
