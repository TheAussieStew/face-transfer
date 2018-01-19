import csv
import numpy as np
from scipy.misc import imsave


def save_statistics(log_path, line_to_add, create=False):
    log_filepath = "{}/summary_statistics.csv".format(log_path)
    if create:
        with open(log_filepath, 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(line_to_add)
    else:
        with open(log_filepath, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(line_to_add)

    return log_filepath

def unstack(np_array):
    new_list = []
    for i in range(np_array.shape[0]):
        temp_list = np_array[i]
        new_list.append(temp_list)
    return new_list

def save_images(batch_list, file_name, rows):
    #[out, b, h, w, c]
    output_image = []
    for image_batch in batch_list:
        im_shape = image_batch.shape
        grid_columns = int(im_shape[0]/rows)
        temp_image = np.reshape(image_batch, newshape=(grid_columns, rows, im_shape[1], im_shape[2], im_shape[3]))
        temp_image = unstack(temp_image)
        temp_image = np.concatenate(temp_image, axis=1)
        temp_image = unstack(temp_image)
        temp_image = np.concatenate(temp_image, axis=1)


        black_line = np.zeros(shape=(temp_image.shape[0], im_shape[1], temp_image.shape[2]))

        temp_image = np.concatenate([temp_image, black_line], axis=1)
        output_image.append(temp_image)

    output_image = np.array(output_image)
    output_image = np.concatenate(output_image, axis=1)
    output_image = ((output_image - np.min(output_image)) / (np.max(output_image) - np.min(output_image))) * 255.
    output_image = np.squeeze(output_image)
    imsave("{}".format(file_name), output_image)
    return "{}".format(file_name)


def load_statistics(experiment_name):
    data_dict = dict()
    with open("{}.csv".format(experiment_name), 'r') as f:
        lines = f.readlines()
        data_labels = lines[0].replace("\n","").split(",")
        del lines[0]

        for label in data_labels:
            data_dict[label] = []

        for line in lines:
            data = line.replace("\n","").split(",")
            for key, item in zip(data_labels, data):
                data_dict[key].append(item)
    return data_dict

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def get_folder_id(drive, f_id, folder_name):

    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(f_id)}).GetList()
    for file1 in file_list:
        if file1['title'] == folder_name:
            return file1['id']

    return None

def delete_file(drive, f_id, folder_name):

    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(f_id)}).GetList()
    for file1 in file_list:
        if file1['title'] == folder_name:
            file1.Delete()



def save_item_to_gdrive_folder(file_to_save_path, gdrive_folder):
    try:
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)
        f_id = "0B7GigvsqXXcyVDNWN1NUY2x5NW8"
        folder_id = get_folder_id(drive, f_id=f_id, folder_name=gdrive_folder)

        if folder_id == None:
            folder_id = create_folder_get_id(drive=drive, folder_name=gdrive_folder, f_id=f_id)

        delete_file(drive, f_id=folder_id, folder_name=file_to_save_path.split("/")[-1])

        f = drive.CreateFile({"parents": [{"id": folder_id}]})
        f.SetContentFile(file_to_save_path)
        f.Upload()
    except:
        print("Failed to upload file to GDRIVE")

def create_folder_get_id(drive, folder_name, f_id):

    fold = drive.CreateFile({'title': folder_name,
                             "parents": [{"id": f_id}],
                             "mimeType": "application/vnd.google-apps.folder"})
    fold.Upload()
    foldertitle = fold['title']
    folderid = fold['id']
    print("Created folder {} which has id {}".format(foldertitle, folderid))
    return folderid


def build_experiment_folder(experiment_name):
    saved_models_filepath = "{}/{}".format(experiment_name.replace("_", "/"), "saved_models")
    logs_filepath = "{}/{}".format(experiment_name.replace("_", "/"), "logs")
    samples_filepath = "{}/{}".format(experiment_name.replace("_", "/"), "visual_outputs")

    import os
    if not os.path.exists(experiment_name.replace("_", "/")):
        os.makedirs(experiment_name.replace("_", "/"))
    if not os.path.exists(logs_filepath):
        os.makedirs(logs_filepath)
    if not os.path.exists(samples_filepath):
        os.makedirs(samples_filepath)
    if not os.path.exists(saved_models_filepath):
        os.makedirs(saved_models_filepath)

    return saved_models_filepath, logs_filepath, samples_filepath
