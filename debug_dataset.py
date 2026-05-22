from data import CmrxReconInferenceSliceDataset

try:
    dataset = CmrxReconInferenceSliceDataset(
        data_path="/data/ChallengeData",
        challenge="multicoil"
    )

    print("Found files:", len(dataset.files))
    if len(dataset.files) > 0:
        for i in range(min(5, len(dataset.files))):
            print(dataset.files[i])
except Exception as e:
    print("Error initializing dataset:", e)
