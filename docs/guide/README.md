---
title: Getting started
lang: en-US
---

## What is Saliency.ai?

Saliency.ai is a set of tools that allow you to:
* Aggragate your data in consistent format
* Collect labels and annotations for your machine learning
* Quickly train machine learning algorithms using pretrained models
* Deploy your trained models as HIPAA-complient webapps

**Spend more time innovating and less time cleaning heterogenous data dumps.**

## Installation

Use `pip` to install our PyPI package in your environment

```bash
> pip install saliency-client
```

You are now ready to connect to Saliency API

```python
from saliency import SaliencyClient

sapi = SaliencyClient("[username]", "[password]")
```

## Aggregate data from multiple sites

You can upload a list of datapoints as a study with a following command

```python
sapi.add_study("oai", "filename.csv")
```

where `filename.csv` is a file with links to datapoints. `SaliencyClient` will automatically determine the column with links to datapoints. All other columns will be stored as metadata associated with the file.

Alternatively, if you have your dataframe with links already stored as a pandas DataFrame, you can simply add it with
```python
sapi.add_study("nhs", pandasDataFrame)
```

## Collect annotations

In a typical machine learning workflow data needs to be annotated. `SaliencyClient` provides a simple interface that allows you to define annotation task (segmentation or labeling). After defining the task, you can send to your contractors a link to the annotation tool populated with images and the definition of the task.

You will first need to define what is to be annotated

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

[TODO make a screenshot more realistic.]

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
loader = sapi.train("U-Net", loader)
```

## Deploy

After your model is ready, use `SaliencyClient` to deploy it
```python
sapi.deploy("kltool",model)
```
Deployed model will run entirely in the browser -- no data leaves enduser's computer. 

See an example deployed model [here](http://demo.saliency.ai/kltool).