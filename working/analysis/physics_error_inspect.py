import pandas as pd

reviewed = "/results/test/batch_1/Results/Review_Sequence_Classifier.csv" 
physics = "/results/DCM_Physics/dicom_headers_labelled_Mar24.csv"

phy_data = pd.read_csv(physics)
re_data = pd.read_csv(reviewed)
re_data = re_data[re_data["viewed"] == "X"]
re_data = re_data.rename(columns={"Paciente": "scan", 
                        "Estudio":  "session",
                        "Serie":    "subject"})

filtered = pd.merge(phy_data, re_data, on=["subject", "session", "scan"], how="left").fillna(0)
filtered = filtered[filtered["viewed"] == "X"]

# Check unknow
