import os
import json
import pandas as pd
from collections import Counter
import glob
from pathlib import Path

def extract_labels_with_frequencies(json_path):
    """
    Extract labels from a single JSON file.
    Returns a list of label values.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return list(data.values())
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return []

def process_subject_folder(subject_path):
    """
    Process a single subject folder and extract information from all sessions.
    """
    subject_results = []
    subject_id = os.path.basename(subject_path)
    
    # Find all session folders within the subject folder
    session_folders = [d for d in os.listdir(subject_path) 
                      if os.path.isdir(os.path.join(subject_path, d))]
    
    for session in session_folders:
        session_path = os.path.join(subject_path, session)
        
        # Look for all bone_seg_labels.json files in the session subfolders
        json_pattern = os.path.join(session_path, "*", "bone_seg_labels.json")
        json_files = glob.glob(json_pattern)
        
        if not json_files:
            print(f"No bone_seg_labels.json files found in {session_path}")
            continue
        
        # Collect labels from all JSON files in this session
        all_labels = []
        for json_file in json_files:
            labels = extract_labels_with_frequencies(json_file)
            all_labels.extend(labels)
        
        if not all_labels:
            print(f"No labels found in session {session} for subject {subject_id}")
            continue
        
        # Count frequencies
        label_counts = Counter(all_labels)
        
        # Get top 3 labels with their counts
        top_3_labels_with_counts = label_counts.most_common(3)
        
        # Initialize results with empty strings/zeros
        top1_label = top_3_labels_with_counts[0][0] if len(top_3_labels_with_counts) > 0 else ""
        top1_count = top_3_labels_with_counts[0][1] if len(top_3_labels_with_counts) > 0 else 0
        
        top2_label = top_3_labels_with_counts[1][0] if len(top_3_labels_with_counts) > 1 else ""
        top2_count = top_3_labels_with_counts[1][1] if len(top_3_labels_with_counts) > 1 else 0
        
        top3_label = top_3_labels_with_counts[2][0] if len(top_3_labels_with_counts) > 2 else ""
        top3_count = top_3_labels_with_counts[2][1] if len(top_3_labels_with_counts) > 2 else 0
        
        subject_results.append({
            'patient_id': subject_id,
            'session': session,
            'total_images': len(json_files),  # Total number of images in this session
            'top1_label': top1_label,
            'top1_count': top1_count,
            'top2_label': top2_label,
            'top2_count': top2_count,
            'top3_label': top3_label,
            'top3_count': top3_count,
            'total_label_instances': len(all_labels),
            'unique_labels': len(label_counts)
        })
    
    return subject_results

def process_all_subjects(base_path, output_path, output_csv='bone_segmentation_summary.csv'):
    """
    Process all subjects in the base directory and create a CSV summary.
    """
    all_results = []
    
    # Get all subject directories
    subject_dirs = [d for d in os.listdir(base_path) 
                   if os.path.isdir(os.path.join(base_path, d))]
    
    print(f"Found {len(subject_dirs)} subjects")
    
    for subject_dir in subject_dirs:
        subject_path = os.path.join(base_path, subject_dir)
        print(f"Processing subject: {subject_dir}")
        
        results = process_subject_folder(subject_path)
        all_results.extend(results)
    
    if not all_results:
        print("No results found. Please check your data structure.")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Reorder columns for better readability
    column_order = [
        'patient_id', 
        'session', 
        'total_images',
        'top1_label', 'top1_count',
        'top2_label', 'top2_count', 
        'top3_label', 'top3_count',
        'total_label_instances',
        'unique_labels'
    ]
    df = df[column_order]
    
    # Sort for better readability
    df = df.sort_values(['patient_id', 'session'])
    
    # Save to CSV
    output_path = os.path.join(output_path, output_csv)
    df.to_csv(output_path, index=False)
    print(f"\nSummary CSV saved to: {output_path}")
    print(f"Total entries: {len(df)}")
    
    return df

def main():
    # Base path to your data
    base_path = "/mnt/rimp/PROJECTS/BONE-AI/tmp_data_totalseg/sorted_data"
    output_path = "/mnt/rimp/PROJECTS/BONE-AI/tmp_data_totalseg/"
    
    # Process all subjects
    df = process_all_subjects(base_path, output_path)
    
    if df is not None:
        # Show a preview of the results
        print("\nPreview of the data:")
        print(df.head(10))
        
        # Display some statistics
        print(f"\nUnique patients: {df['patient_id'].nunique()}")
        print(f"Unique sessions: {df['session'].nunique()}")
        print(f"Total images processed: {df['total_images'].sum()}")
        
        # Show statistics about label frequencies
        print("\nLabel frequency statistics:")
        print(f"Average number of labels per session: {df['total_label_instances'].mean():.2f}")
        print(f"Average unique labels per session: {df['unique_labels'].mean():.2f}")
        
        # Show the most common labels across all sessions
        print("\nTop 10 most common labels across all sessions:")
        all_labels = []
        for col in ['top1_label', 'top2_label', 'top3_label']:
            all_labels.extend(df[col][df[col] != ""].tolist())
        label_counts = Counter(all_labels)
        for label, count in label_counts.most_common(10):
            print(f"  {label}: {count}")

if __name__ == "__main__":
    main()