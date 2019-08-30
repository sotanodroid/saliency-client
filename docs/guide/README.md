---
title: Getting started
lang: en-US
---

## What is Saliency.ai?

Saliency.ai is a set of tools that allow you to:
* Aggregate heterogeneous data and retrieve it in a consistent format
* Collect labels and annotations for machine learning
* Quickly train machine learning algorithms using models that have already learned from thousands of medical images
* Deploy your trained models as HIPAA-compliant web apps

**Spend more time innovating and less time cleaning heterogeneous data dumps.**

## Installation

Use `pip` to install our PyPI package in your environment

```bash
> pip install saliency-client
```

You are now ready to connect to the Saliency API

```python
from saliency import SaliencyClient

sapi = SaliencyClient("[username]", "[password]")
```

## Aggregate data from multiple sites

A *study* is a set of data that you want to analyze. You can upload a list of datapoints as a study with the following command

```python
sapi.add_study("study_name", "filename1.csv")
```

where `filename.csv` is a file with links to datapoints. `SaliencyClient` will automatically determine the column with links to datapoints. All other columns will be stored as metadata associated with the file.

Alternatively, if you have your dataframe with links already stored as a pandas DataFrame, you can simply add it with
```python
sapi.add_study("study_name", pandasDataFrame)
```
#### Example: 
You have knee MRIs for 1000 patients from Hospital A, along with the patients' age and disease severity. This is organized in HospitalA.csv which has three columns: age, disease_severity, and location_of_mri_on_your_server. You also have knee MRIs for 2000 patients from Hospital B, along with the patients' sex, state of residence, and disease severity. This is organized in a pandas dataframe called HospitalB which has four columns: sex, state, location_of_mri_on_your_server, and disease_severity. In both files, the location_of_mri_on_your_server column is a column of web addresses where your MRIs reside on your server. These datasets from Hospital A and Hospital B represent two different studies. You would call the sapi.add_study() function twice in order to upload both, even if you later want to merge the images and disease severitiy values from both for a single analysis.

## Collect annotations

In a typical machine learning workflow, data needs to be annotated. `SaliencyClient` provides a simple interface that allows you to define an annotation task (segmentation or labeling). After defining the task, you can send to your contractors a link to the annotation tool populated with images and the definition of the task.

You will first need to define what is to be annotated:

```python
sapi.add_annotation_scheme("Knee structures", "segmentation", [
    "Femoral Cartilage",
	"Lateral Meniscus",
	"Lateral Tibial Cartilage",
	"Medial Meniscus",
	"Medial Tibial Cartilage",
	"Patellar Cartilage"
])
```

Now, you can define the task by refering to the annotation scheme defined above by its name, and a list of studies to annotate

```python
sapi.add_task("Knee structures", ["oai", "nhs"])
```

After defining the task, you can log in to the [annotation tool](https://annotator.saliency.ai/) and start annotating.

[TODO make the screenshot more realistic.]

![Annotation Tool](/annotation-tool.png)

## Train models

After sufficient number of annotations came back, you start training your models. Get an iterator of the data with the labels. You can define the format in which you want to receive the data

```python
data_iter = sapi.get_data(["oai", "nhs"],
                          labels="Knee structures",
						  {"width": 500, "height": 500})
```

We provide wrappers for this data iterater into a typical data loaders in tensorflow, keras, and pytorch. Run

```python
loader = sapi.format_iterator(data_iter, "keras")
```
and start training your models! We provide a UNET training code as a baseline:

```python
model = sapi.train("U-Net", loader)
```

## Deploy

After your model is ready, use `SaliencyClient` to deploy it
```python
sapi.deploy("kltool",model)
```
Deployed model will run entirely in the browser -- no data leaves enduser's computer. 

See an example deployed model [here](http://demo.saliency.ai/kltool).
