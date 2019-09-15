import csv
import json
import os
import tempfile
import urllib
import zipfile

import cv2
import uuid
import numpy as np
import pandas as pd
import pydicom
import requests
import scipy.io as sio
from skimage.draw import polygon


class AuthenticationError(Exception):
    pass

class SaliencyClient:
    seq = {"train": [],"val": [],"test": []}
    batch_size = 16
    task = "classification"
    temp_dir = 'tempfiles'
    classes = 0

    def __init__(self, username, password, host="https://api.saliency.ai/"):
        self.username = username
        self.password = password
        self.host = host
        self.headers = self.login()
        if not self.headers:
            raise AuthenticationError('Failed to authenticate "{}"'.format(username))

    def login(self):
        auth_link = self.host + 'api-auth/token/'
        r = requests.post(auth_link, data={'username': self.username, 'password': self.password})

        if r.status_code != 200:
            raise Exception("Authentication error")

        headers = {'Authorization': 'Token ' + r.json()['token']}
        return headers

    def add_annotation_scheme(self, name, type, labels):
        # TODO: check labels formatting
        
        data = {
            "name": name,
            "type": type,
            "labels": labels
        }
        r = requests.post(self.host + 'schemes/', data=data, headers=self.headers)
        if r.status_code in (200, 201):
            print('Scheme has been created')
            return r.json()
        else:
            for error in r.json().values():
                print(error)

    def add_task(self, scheme, study):
        # TODO: check labels formatting

        data = {
            "scheme": scheme,
            "study": study,
        }
        r = requests.post(self.host + 'tasks/create_by_names/', data=data, headers=self.headers)
        if r.status_code in (200, 201):
            print('Task has been created')
            return r.json()
        else:
            print(r.content)

    def add_study(self, study, study_name):
        if type(study) == pd.DataFrame:
            filename = os.path.join(self.temp_dir, 'temp_study.csv')
            study.to_csv(filename)
        else:
            if os.path.isfile(study):
                filename = study
            else:
                print('File {} does not exist')
                return
        with open(filename, 'rb') as file:
            data = {'name': study_name}
            files = {'file': file}
            r = requests.post(self.host + 'studies/', data=data, files=files, headers=self.headers)
        if r.status_code in (200, 201):
            print('Study has been created')
            return r.json()
        else:
            for error in r.json().values():
                print(error)

    def get_mri(self, tmpfile):
        tmpmridir = tmpfile + "-dir"
        zip_ref = zipfile.ZipFile(tmpfile, 'r')
        zip_ref.extractall(tmpmridir)
        zip_ref.close()

        # list all files
        while True:
            # our mris have a lot of nested dirs
            mrifiles = os.listdir(tmpmridir)
            testfile = "%s/%s" % (tmpmridir, mrifiles[0])
            if (not os.path.isdir(testfile)):
                break
            tmpmridir = testfile

        # figure out the dimensions and create an array
        mrifiles = sorted(mrifiles)
        ds = pydicom.dcmread("%s/%s" % (tmpmridir, mrifiles[0]) )
        shape = ds.pixel_array.shape + (len(mrifiles),)
        X = np.zeros(shape)

        # copy slices
        for i, tmpfile in enumerate(mrifiles):
            ds = pydicom.dcmread("%s/%s" % (tmpmridir, tmpfile) )
            X[:,:,i] = ds.pixel_array

        return X

    def get_xray(self, tmpfile):
        X = cv2.imread(tmpfile,0)
        return X

    def get_xray_anno(self, tmpfile):
        X = cv2.imread(tmpfile,0)
        return X

    def get_zip_mri_anno(self, tmpfile, shape):
        Y = np.zeros(shape)
        mat = sio.loadmat(tmpfile)
        npoints = mat['FemoralCartilage'].shape[0]

        lx = 0
        ly = 0
        lz = None

        contour = []

        for i in range(npoints):
            py, px, pz = tuple(mat['FemoralCartilage'][i,:])
            py = py
            px = shape[0] - px

            dist = np.sqrt((lx - px)**2 + (ly - py)**2)
            if ((pz != lz) or (dist > 25)) and lz != None:
                acontour = np.array(contour)
                rr, cc = polygon(acontour[:,0], acontour[:,1], Y.shape[0:2])
                Y[rr,cc,int(pz)] = 1
                contour = []

            contour.append([px,py])
            lx,ly,lz = px,py,pz

        return Y

    def get_file(self, file):
        file_uid = str(uuid.uuid4())[:8]
        file_name = file_uid + file["filename"]  # creates unique filename
        tmpfile = "{filedir}/{filename}".format(filedir=tempfile.gettempdir(), filename=file_name)

        # check if cashed
        # filehash = self.get_file_hash(file)
        # if filehash in self.storage.keys():
        #     return np.array(self.storage[filehash])

        # check if already downloaded
        if not os.path.isfile(tmpfile):
            urllib.request.urlretrieve(file["file"], tmpfile)

        # process dicom MRI
        if file["datatype"] == "mri":
            X = self.get_mri(tmpfile)

        # process xray
        if file["datatype"] == "xray":
            X = self.get_xray(tmpfile)

        # cache the file
        # self.storage[filehash] = X

        return X

    def get_annotation(self, file, shape, headers):
        if file['annotation_set']:
            r = requests.get(file['annotation_set'][0], headers=headers)
            datatype = file['datatype']
            annotation = r.json()
            tmpfile = "%s/%s" % (tempfile.gettempdir(), annotation["filename"])
            #if not os.path.isfile(tmpfile):
            urllib.request.urlretrieve(annotation["file"], tmpfile)
            if datatype == 'mri':
                res = self.get_zip_mri_anno(tmpfile, shape)
            elif datatype == 'xray':
                res = self.get_xray_anno(tmpfile)
            return res
        return None


    def create_task(self, scheme_id, study_id, datatype="xray"):
        headers = self.login()
        if headers:
            datapoints = []
            # annotations = []
            url = '{}datapoints/?datatype={}&study={}'.format(self.host, datatype, study_id)
            while url is not None:
                r = requests.get(url, headers=headers)
                r = r.json()
                for item in r['results']:
                    datapoint_url = '/datapoints/{}/'.format(item['id'])
                    datapoints.append(datapoint_url)
                    # if item['annotation_set']:
                    #     for anno_url in item['annotation_set']:
                    #         annotations.append(anno_url)
                url = r['next']
            scheme_url = '/schemes/{}/'.format(scheme_id)
            data = {'scheme': scheme_url, 'datapoints': datapoints}
            r = requests.post(self.host + 'tasks/', data=data, headers=headers)
            return r.json()

    def download_datapoints(self, patient="", study="", datatype="xray"):
        headers = self.login()
        if headers:
            link = '%sdatapoints/?datatype=%s&patientid=%s&study=%s' % (self.host, datatype, patient, study)
            while link != None:
                r = requests.get(link, headers=headers)
                r = r.json()
                print(r)
                for datapoint in r['results']:
                    x = self.get_file(datapoint)
                    y = self.get_annotation(datapoint, x.shape, headers)
                    yield x, y
                link = r['next']

    def create_arrays(self, values, width, height):
        list_x = []
        list_y = []
        for x, y in values:
            x = self.resize_image(x, width, height)
            if y is not None:
                y = self.resize_image(y, width, height)
            list_x.append(x)
            list_y.append(y)
        list_x = np.asarray(list_x)
        list_y = np.asarray(list_y)
        return list_x, list_y

    def resize_image(self, image, width, height):
        im_height, im_width = image.shape
        ratio = im_width / im_height
        new_ratio = width / height
        if ratio > new_ratio:
            new_width = int(new_ratio * height)
            offset = (im_width - new_width) / 2
            resize = (im_width - offset, height)
            width_offset = int(offset)
            height_offset = 0
        else:
            new_height = int(width / new_ratio)
            offset = (im_height - new_height) / 2
            resize = (width, im_height - offset)
            width_offset = 0
            height_offset = int(offset)

        resize = (int(resize[0]), int(resize[1]))
        image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
        crop = image[height_offset:height + height_offset, width_offset:width + width_offset]
        return crop

    def _list_objects(self, obj, obj_id=None):
        """
        Method to cover all listings.

        param:obj: is string, that represents an object name in the API. (e.g. 'tasks')
        param:obj_id is actual id of the object in the API
        """
        url = self.host + '{obj}/'.format(obj=obj)
        if obj_id:
            url += '{obj_id}/'.format(obj_id=obj_id)
        response = requests.get(url=url, headers=self.headers)

        return response.json()

    def _delete_objects(self, obj, obj_id):
        """
        Method to delete object.

        param:obj: is string, that represents an object name in the API. (e.g. 'tasks')
        param:obj_id is actual id of the object in the API
        """
        url = self.host + '{obj}/{obj_id}/'.format(obj=obj, obj_id=obj_id)
        response = requests.delete(url=url, headers=self.headers)

        return response.json()

    def list_tasks(self, task_id=None):
        """
        Method to list all tasks.

        param:task_id if passed function would return information for specific task.
        """
        tasks = self._list_objects(obj='tasks', obj_id=task_id)

        return pd.read_json(json.dumps(tasks))

    def delete_task(self, task_id):
        """
        Method to delete specific task.

        param:task_id is actual id of the task in the API
        """
        return self._delete_objects(obj='tasks', obj_id=task_id) 
