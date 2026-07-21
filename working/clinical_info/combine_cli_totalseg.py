import pandas as pd
import os

# Read files
df_seg = pd.read_csv('/mnt/rimp/PROJECTS/BONE-AI/tmp_data_totalseg/bone_segmentation_summary.csv')
df_info = pd.read_csv('/home/ext_xinwan/Bone_AI/output/clinical_info/combined_clinical_info_batches1_to_4.csv')

# Deduplicate and keep only subjects in both
df_info_unique = df_info.drop_duplicates(subset=['subject'], keep='first')
df_combined = df_seg.merge(df_info_unique, left_on='patient_id', right_on='subject', how='inner')

# Drop the redundant 'subject' column
df_combined = df_combined.drop('subject', axis=1)

# Define column order
subject_cols = ['birthdate', 'age', 'gender', 'symptoms', 'history_of_neoplasm', 
                'suspected_metastasis', 'skeletal_location', 'location_within_bone']
other_cols = [col for col in df_combined.columns if col not in ['patient_id'] + subject_cols]

# Reorder: patient_id, subject info, then everything else
df_combined = df_combined[['patient_id'] + subject_cols + other_cols]

# Save
df_combined.to_csv('/home/ext_xinwan/Bone_AI/output/clinical_info/combined_bone_data_anatomy_cli.csv', index=False)

print(f"Combined data saved. Rows: {len(df_combined)}")
print(f"Patients: {df_combined['patient_id'].nunique()}")
print(f"\nColumns in order:")
for i, col in enumerate(df_combined.columns, 1):
    print(f"  {i}. {col}")
print("\nPreview:")
print(df_combined.head())