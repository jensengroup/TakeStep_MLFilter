import tqdm
import requests
import os

def _download_file(url: str, filename: str, _dir: str) -> None:
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        pbar = tqdm.tqdm(
            total=total_size, desc=os.path.basename(filename), unit="B", unit_scale=True
        )
        with open(_dir + "/" + filename, "wb") as fileobj:
            for chunk in response.iter_content(chunk_size=1024):
                fileobj.write(chunk)
                pbar.update(len(chunk))
        pbar.close()


def download_processed_data():
    data_filenames = ["xtrain.npy", "ytrain.npy", "xtrain_smote.npy", "ytrain_smote.npy", "training_fragments_data.csv"]
    data_url = "https://sid.erda.dk/share_redirect/EylpneGCF3/processed_train_data/"
    data_urls = [data_url + fname for fname in data_filenames]

    for name, url in zip(data_filenames, data_urls):
        _download_file(url, name, "processed_data")


def download_models():
    """ """
    data_filenames = ["balanced_convergence_clf.pkl", "rf_smote_convergence_clf.pkl"]
    data_url = "https://sid.erda.dk/share_redirect/EylpneGCF3/models/"
    data_urls = [data_url + fname for fname in data_filenames]

    for name, url in zip(data_filenames, data_urls):
        _download_file(url, name, "models")


if __name__ == "__main__":
    
    download_processed_data()