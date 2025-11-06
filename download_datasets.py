import os
import zipfile
import subprocess

# Kaggle Auto Downloader

# Create datasets directory
BASE_DIR = os.path.join(os.getcwd(), "datasets")
os.makedirs(BASE_DIR, exist_ok=True)

# List of all datasets to download
DATASETS = {
    "fever_diagnosis": "ziya07/fever-diagnosis-and-medicine-dataset",
    "disease_symptom": "itachi9604/disease-symptom-description-dataset",
    "disease_prediction": "kaushil268/disease-prediction-using-machine-learning",
    "symptom2disease": "niyarrbarman/symptom2disease",
    "patient_profile": "uom190346a/disease-symptoms-and-patient-profile-dataset",
    "covid_symptom": "hemanthhari/symptoms-and-covid-presence",
    #"covid_radiography": "tawsifurrahman/covid19-radiography-database",
    "typhoid_detection": "m9pnvv2vpv",  # From Mendeley (manual download)
    #"coughvid": "nasrulhakim86/coughvid-wav"

}

def download_and_unzip(name, dataset):
    print(f"\n Downloading {name} ...")
    folder_path = os.path.join(BASE_DIR, name)
    os.makedirs(folder_path, exist_ok=True)
    
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", folder_path],
            check=True
        )
    except subprocess.CalledProcessError:
        print(f"⚠ Could not download {name}. Skipping.")
        return
    
    # Unzip any downloaded zip files
    for file in os.listdir(folder_path):
        if file.endswith(".zip"):
            zip_path = os.path.join(folder_path, file)
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(folder_path)
                os.remove(zip_path)
                print(f"Extracted and cleaned: {name}")
            except:
                print(f"⚠ Could not unzip {file}")
                continue

if __name__ == "__main__":
    print("\n Starting Kaggle dataset downloader...\n")
    for name, dataset in DATASETS.items():
        if dataset == "m9pnvv2vpv":
            print(f"{name}: Download this manually from Mendeley (browser link).")
            continue
        download_and_unzip(name, dataset)
    print("\n All available datasets downloaded and organized in 'datasets/' folder!")